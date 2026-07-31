import streamlit as st
import cv2
import numpy as np
import pandas as pd

import database
import face_utils

# Page Config
st.set_page_config(
    page_title="Registration Portal - AI Attendance System",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Unique Sky Blue & White Theme & Mobile Responsiveness
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Unique Sky Blue & White Gradient Canvas */
    .stApp {
        background: radial-gradient(ellipse at 50% -20%, #bae6fd 0%, #f0f9ff 45%, #ffffff 100%);
        color: #0f172a;
    }
    .stSidebar {
        background-color: rgba(240, 249, 255, 0.95) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid #7dd3fc;
    }
    
    /* Unique Hero Sky Blue Banner */
    .sky-hero-banner {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 60%, #0c4a6e 100%);
        padding: 1.5rem 2rem;
        border-radius: 20px;
        box-shadow: 0 12px 30px -5px rgba(2, 132, 199, 0.35);
        border-top: 4px solid #38bdf8;
        border-left: 1px solid rgba(255,255,255,0.2);
        border-right: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .sky-hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.25) 0%, transparent 70%);
        pointer-events: none;
    }
    .sky-brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.02em;
        margin: 0;
        text-shadow: 0 2px 8px rgba(2, 132, 199, 0.3);
    }
    .sky-brand-tagline {
        font-size: 0.92rem;
        color: #e0f2fe;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0369a1;
        margin-bottom: 0.4rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #334155;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    /* Unique Sky Blue & White Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #e0f2fe;
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #7dd3fc;
        box-shadow: inset 0 2px 6px rgba(2, 132, 199, 0.08);
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 12px;
        color: #0369a1;
        font-weight: 700;
        font-size: 0.95rem;
        padding: 0 24px;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 6px 18px rgba(2, 132, 199, 0.4);
    }

    /* Unique Card Containers & Metrics */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #7dd3fc;
        border-top: 3px solid #0284c7;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 8px 25px rgba(2, 132, 199, 0.12);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(2, 132, 199, 0.2);
    }
    [data-testid="stMetricValue"] {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        color: #0284c7 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        color: #475569 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Unique Sky Blue Action Buttons */
    .stButton button, .stDownloadButton button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 1.6rem !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton button:hover, .stDownloadButton button:hover {
        transform: translateY(-3px) !important;
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 100%) !important;
        box-shadow: 0 10px 30px rgba(2, 132, 199, 0.5) !important;
    }

    /* Unique Sky Chips */
    .reg-chip {
        background: #e0f2fe;
        border: 1.5 solid #0284c7;
        color: #0284c7;
        padding: 7px 16px;
        border-radius: 30px;
        display: inline-block;
        margin: 5px;
        font-weight: 700;
        font-size: 0.88rem;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.15);
    }

    /* Inputs Styling */
    .stTextInput input, .stSelectbox select, .stMultiselect {
        border-radius: 12px !important;
        border: 1.5px solid #7dd3fc !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.05) !important;
    }
    .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #0284c7 !important;
        box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.25) !important;
    }

    /* Mobile Responsive Rules */
    @media (max-width: 768px) {
        .sky-hero-banner { padding: 1.1rem; text-align: center; }
        .sky-brand-title { font-size: 1.6rem !important; }
        .main-header { font-size: 1.5rem !important; text-align: center; }
        .sub-header { font-size: 0.9rem !important; text-align: center; margin-bottom: 1.2rem; }
        .stTabs [data-baseweb="tab-list"] { display: flex; overflow-x: auto; width: 100%; }
        .stTabs [data-baseweb="tab"] { font-size: 0.85rem !important; height: 40px !important; padding: 0 14px !important; flex-shrink: 0; }
        [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important; }
        .stButton button, .stDownloadButton button { width: 100% !important; min-height: 48px !important; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database
database.init_db()

# Sidebar Info
st.sidebar.image("https://img.icons8.com/color/96/facial-recognition.png", width=65)
st.sidebar.title("AI Attendance System")
st.sidebar.markdown("**Registration Portal**")
st.sidebar.info(
    "📸 **AI Facial Attendance System**\n"
    "Central Registration System connected to SQLite DB."
)

# Unique Hero Header Banner
st.markdown("""
<div class="sky-hero-banner">
    <div class="sky-brand-title">📸 AI FACE RECOGNITION SYSTEM</div>
    <div class="sky-brand-tagline">Student Facial Registration & Faculty Account Portal</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="sub-header">Register student facial profiles and create faculty accounts for the AI attendance system.</div>', unsafe_allow_html=True)

# Tabs Navigation
tab1, tab2 = st.tabs([
    "🎓 Student Face Registration", 
    "👨‍🏫 Faculty Account Registration"
])

# Helper function to convert streamlit file/camera input to BGR numpy array
def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# TAB 1: STUDENT REGISTRATION
# ==============================================================================
with tab1:
    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:
        st.subheader("📋 Student Details & Face Photo Upload")
        with st.form("student_reg_form", clear_on_submit=False):
            reg_no = st.text_input("Register Number *", placeholder="e.g. REG2024001")
            full_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
            department = st.selectbox(
                "Department *",
                [
                    "Computer Science & Engineering", 
                    "Information Technology", 
                    "Electrical & Electronics", 
                    "Mechanical Engineering", 
                    "Civil Engineering", 
                    "Biotechnology", 
                    "Other"
                ]
            )
            
            st.markdown("### 📷 Face Photo Capture")
            input_source = st.radio("Choose Photo Input Method:", ["Upload Photo File", "Live Webcam Snapshot"], horizontal=True)
            
            photo_file = None
            if input_source == "Upload Photo File":
                photo_file = st.file_uploader("Upload Clear Face Photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
            else:
                photo_file = st.camera_input("Take a Snapshot")
            
            submit_btn = st.form_submit_button("Register Student Profile", use_container_width=True)

        if submit_btn:
            if not reg_no.strip() or not full_name.strip():
                st.error("⚠️ Please fill in all required fields (Register Number and Full Name).")
            elif photo_file is None:
                st.error("⚠️ Please upload or capture a photo.")
            elif database.student_exists(reg_no):
                st.error(f"⚠️ Student with Register Number '{reg_no.strip()}' is already registered in database!")
            else:
                with st.spinner("Processing facial embedding via InsightFace..."):
                    img_bgr = load_image_as_bgr(photo_file)
                    
                    if img_bgr is None:
                        st.error("❌ Failed to decode image. Please upload a valid image file.")
                    else:
                        embedding, status_code, message = face_utils.extract_single_face_embedding(img_bgr)
                        
                        if status_code == "NO_FACE":
                            st.error(f"❌ {message}")
                        elif status_code == "MULTIPLE_FACES":
                            st.warning(f"⚠️ {message}")
                        elif status_code == "SUCCESS":
                            success = database.add_student(reg_no, full_name, department, embedding)
                            if success:
                                st.balloons()
                                st.success(f"🎉 Success! Student '{full_name}' ({reg_no}) registered successfully into central database.")
                            else:
                                st.error("❌ Database error during insertion.")
                        else:
                            st.error(f"❌ {message}")

    with col2:
        st.subheader("💡 Registration Guidelines")
        st.markdown("""
        To ensure high attendance recognition accuracy, please follow these guidelines:
        
        - 👤 **Single Face**: Only one face must be present in the frame.
        - 💡 **Good Lighting**: Ensure your face is well-lit and clearly visible.
        - 👀 **Look at Camera**: Look directly into the camera with a neutral expression.
        - 👓 **No Obstructions**: Avoid sunglasses, heavy masks, or face-covering hats.
        """)
        
        st.markdown("---")
        st.subheader("📊 Registration Database Status")
        all_registered = database.get_all_students()
        st.metric("Total Registered Students in DB", len(all_registered))

# ==============================================================================
# TAB 2: FACULTY ACCOUNT REGISTRATION
# ==============================================================================
with tab2:
    f_col1, f_col2 = st.columns([1.2, 1], gap="large")
    
    with f_col1:
        st.subheader("🔑 New Faculty Account Registration")
        st.markdown("Create a new faculty user account to allow access to **Website 2 (Attendance Capture Portal)**.")
        with st.form("faculty_reg_form", clear_on_submit=True):
            f_name = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
            f_username = st.text_input("Username *", placeholder="alan_turing")
            f_password = st.text_input("Password *", type="password", placeholder="Set secure password")
            f_confirm_pw = st.text_input("Confirm Password *", type="password")
            
            f_submit = st.form_submit_button("Register Faculty Account", use_container_width=True)
            
        if f_submit:
            if not f_name.strip() or not f_username.strip() or not f_password.strip():
                st.error("⚠️ All fields are required.")
            elif f_password != f_confirm_pw:
                st.error("❌ Passwords do not match!")
            elif database.faculty_exists(f_username):
                st.error(f"⚠️ Username '{f_username.strip()}' is already taken.")
            else:
                success = database.add_faculty(f_username, f_password, f_name)
                if success:
                    st.success(f"🎉 Faculty account created for '{f_name}' (Username: `{f_username.strip()}`). You can now login on Website 2!")
                else:
                    st.error("❌ Database error while creating faculty account.")
                    
    with f_col2:
        st.subheader("👥 Registered Faculty Accounts")
        faculty_list = database.get_all_faculty()
        st.metric("Total Registered Faculty Accounts", len(faculty_list))
        
        if faculty_list:
            df_faculty = pd.DataFrame(faculty_list)
            st.dataframe(df_faculty, use_container_width=True, hide_index=True)
