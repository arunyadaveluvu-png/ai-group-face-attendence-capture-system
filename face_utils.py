import cv2
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple, Optional, Any

# Try loading InsightFace, with fallback indicator if missing
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

@st.cache_resource
def load_face_analysis_model():
    """
    Loads and caches the InsightFace FaceAnalysis model.
    """
    if not INSIGHTFACE_AVAILABLE:
        return None
    try:
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except Exception as e:
        # Fallback to default or lighter setup if buffalo_l download/init fails
        try:
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            return app
        except Exception as inner_e:
            st.error(f"Failed to load InsightFace model: {inner_e}")
            return None

def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two feature vectors."""
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))

def extract_single_face_embedding(image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str, str]:
    """
    Extract facial embedding from a single student registration image.
    Validates face count in the frame.
    
    Returns:
        (embedding, status_code, message)
    """
    model = load_face_analysis_model()
    if model is None:
        return None, "MODEL_ERROR", "InsightFace model is not initialized."
    
    faces = model.get(image_bgr)
    
    if len(faces) == 0:
        return None, "NO_FACE", "No face detected in the image. Please upload a clear photo showing your face."
    elif len(faces) > 1:
        return None, "MULTIPLE_FACES", f"Detected {len(faces)} faces! Please ensure only your face is present in the frame."
    
    # Single face successfully detected
    embedding = faces[0].embedding
    return embedding, "SUCCESS", "Face extracted successfully!"

def recognize_faces_in_group(
    group_image_bgr: np.ndarray,
    registered_students: List[Dict[str, Any]],
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, int]]:
    """
    Detect faces in group photo and match against registered student embeddings.
    
    Returns:
        (annotated_image_bgr, attendance_records, summary_metrics)
    """
    annotated_img = group_image_bgr.copy()
    model = load_face_analysis_model()
    
    if model is None:
        return annotated_img, [], {"total_registered": len(registered_students), "present": 0, "absent": len(registered_students)}
    
    detected_faces = model.get(group_image_bgr)
    
    # Track presence for each registered student
    present_reg_nos = set()
    matches_info = {} # reg_no -> best_confidence
    
    for face in detected_faces:
        bbox = face.bbox.astype(int)
        det_embedding = face.embedding
        
        best_match_reg = None
        best_sim = -1.0
        best_student = None
        
        # Compare against all registered students
        for student in registered_students:
            sim = compute_cosine_similarity(det_embedding, student["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match_reg = student["register_no"]
                best_student = student
        
        if best_sim >= threshold and best_match_reg is not None:
            present_reg_nos.add(best_match_reg)
            if best_match_reg not in matches_info or best_sim > matches_info[best_match_reg]:
                matches_info[best_match_reg] = best_sim
            
            # Draw green bounding box & student label
            label = f"{best_student['name']} ({best_match_reg}) [{best_sim:.2f}]"
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            # Label background banner
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (bbox[0], bbox[1] - 20), (bbox[0] + w, bbox[1]), (0, 255, 0), -1)
            cv2.putText(annotated_img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            # Unknown face - Draw yellow/red bounding box
            label = f"Unknown [{best_sim:.2f}]" if best_sim > 0 else "Unknown"
            cv2.rectangle(annotated_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
            cv2.putText(annotated_img, label, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
    # Build attendance list
    attendance_records = []
    for student in registered_students:
        reg = student["register_no"]
        is_present = reg in present_reg_nos
        attendance_records.append({
            "Register No": reg,
            "Name": student["name"],
            "Department": student["department"],
            "Status": "Present" if is_present else "Absent",
            "Match Confidence": f"{matches_info[reg]:.2f}" if reg in matches_info else "N/A"
        })
        
    metrics = {
        "total_registered": len(registered_students),
        "total_detected": len(detected_faces),
        "present": len(present_reg_nos),
        "absent": len(registered_students) - len(present_reg_nos)
    }
    
    return annotated_img, attendance_records, metrics
