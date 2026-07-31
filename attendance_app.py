import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime

import database
import face_utils
import export_utils

# Page Config
st.set_page_config(
    page_title="Attendance Capture Portal - AI System",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Toggle Control with Instant Re-render
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

st.sidebar.markdown("### 🎨 Theme Selector")
toggle_val = st.sidebar.toggle(
    "🌙 Enable Dark Mode",
    value=st.session_state["dark_mode"],
    key="att_dark_mode_toggle_key"
)

if toggle_val != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = toggle_val
    st.rerun()

# Apply CSS Overrides based on Theme State
if st.session_state["dark_mode"]:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Force Dark Background Everywhere */
        html, body, .stApp, section.main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #020617 !important;
            background: radial-gradient(ellipse at 50% -20%, #0c4a6e 0%, #0f172a 60%, #020617 100%) !important;
            color: #f8fafc !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

        /* Dark Sidebar Override */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"], section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(56, 189, 248, 0.3) !important;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }

        /* Hero Banner Dark */
        .sky-hero-banner {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 60%, #0f172a 100%) !important;
            padding: 1.5rem 2rem;
            border-radius: 20px;
            box-shadow: 0 12px 30px -5px rgba(56, 189, 248, 0.3) !important;
            border-top: 4px solid #38bdf8 !important;
            border-left: 1px solid rgba(255,255,255,0.1) !important;
            border-right: 1px solid rgba(255,255,255,0.1) !important;
            margin-bottom: 1.8rem;
        }
        .sky-brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin: 0;
            text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }
        .sky-brand-tagline {
            font-size: 0.92rem;
            color: #38bdf8 !important;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 4px;
        }

        .main-header, h1, h2, h3 {
            color: #38bdf8 !important;
        }
        .sub-header, p, span, label, div.stMarkdown {
            color: #cbd5e1 !important;
        }

        /* Dark Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: rgba(15, 23, 42, 0.8) !important;
            padding: 8px;
            border-radius: 16px;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background-color: transparent !important;
            border-radius: 12px;
            color: #94a3b8 !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0 24px;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 6px 18px rgba(2, 132, 199, 0.5) !important;
        }
        .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
            color: #ffffff !important;
        }

        /* Dark Metrics */
        [data-testid="stMetric"], .sky-card {
            background: #1e293b !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-top: 3px solid #38bdf8 !important;
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4) !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            font-size: 0.85rem !important;
            color: #cbd5e1 !important;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.7rem 1.6rem !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 30px rgba(56, 189, 248, 0.6) !important;
        }

        .reg-chip {
            background: rgba(2, 132, 199, 0.2) !important;
            border: 1.5px solid #38bdf8 !important;
            color: #38bdf8 !important;
            padding: 7px 16px;
            border-radius: 30px;
            display: inline-block;
            margin: 5px;
            font-weight: 700;
        }

        input, select, textarea, [data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1.5px solid rgba(56, 189, 248, 0.4) !important;
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* Force Light Background Everywhere */
        html, body, .stApp, section.main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #ffffff !important;
            background: radial-gradient(ellipse at 50% -20%, #bae6fd 0%, #f0f9ff 45%, #ffffff 100%) !important;
            color: #0f172a !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

        /* Light Sidebar Override */
        [data-testid="stSidebar"], [data-testid="stSidebarContent"], section[data-testid="stSidebar"] {
            background-color: #f0f9ff !important;
            border-right: 1px solid #7dd3fc !important;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #0f172a !important;
        }
        
        .sky-hero-banner {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 60%, #0c4a6e 100%) !important;
            padding: 1.5rem 2rem;
            border-radius: 20px;
            box-shadow: 0 12px 30px -5px rgba(2, 132, 199, 0.35) !important;
            border-top: 4px solid #38bdf8 !important;
            border-left: 1px solid rgba(255,255,255,0.2) !important;
            border-right: 1px solid rgba(255,255,255,0.2) !important;
            margin-bottom: 1.8rem;
        }
        .sky-brand-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .sky-brand-tagline {
            font-size: 0.92rem;
            color: #e0f2fe !important;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-top: 4px;
        }

        .main-header, h1, h2, h3 {
            color: #0369a1 !important;
        }
        .sub-header, p, span, label, div.stMarkdown {
            color: #334155 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background-color: #e0f2fe !important;
            padding: 8px;
            border-radius: 16px;
            border: 1px solid #7dd3fc !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            background-color: transparent !important;
            border-radius: 12px;
            color: #0369a1 !important;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 0 24px;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            box-shadow: 0 6px 18px rgba(2, 132, 199, 0.4) !important;
        }
        .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
            color: #ffffff !important;
        }

        [data-testid="stMetric"], .sky-card {
            background: #ffffff !important;
            border: 1px solid #7dd3fc !important;
            border-top: 3px solid #0284c7 !important;
            border-radius: 16px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 8px 25px rgba(2, 132, 199, 0.12) !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            font-size: 2.1rem !important;
            font-weight: 800 !important;
            color: #0284c7 !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            font-size: 0.85rem !important;
            color: #475569 !important;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 700 !important;
            padding: 0.7rem 1.6rem !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.35) !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 30px rgba(2, 132, 199, 0.5) !important;
        }

        .reg-chip {
            background: #e0f2fe !important;
            border: 1.5px solid #0284c7 !important;
            color: #0284c7 !important;
            padding: 7px 16px;
            border-radius: 30px;
            display: inline-block;
            margin: 5px;
            font-weight: 700;
        }

        input, select, textarea, [data-baseweb="select"] {
            border-radius: 12px !important;
            border: 1.5px solid #7dd3fc !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Initialize Database
database.init_db()

# Session State Initializations
if "faculty_logged_in" not in st.session_state:
    st.session_state["faculty_logged_in"] = False
if "faculty_user" not in st.session_state:
    st.session_state["faculty_user"] = None
if "cam_snapshots_list" not in st.session_state:
    st.session_state["cam_snapshots_list"] = []

# Sidebar Info
st.sidebar.image("https://img.icons8.com/color/96/facial-recognition.png", width=65)
st.sidebar.title("AI Attendance System")
st.sidebar.markdown("**Attendance Portal**")
st.sidebar.info(
    "📸 **AI Facial Attendance System**\n"
    "Classroom Attendance Capture & Database Management."
)

# Top Bar & Sidebar Theme Toggle Controls
top_col1, top_col2 = st.columns([3, 1])
with top_col2:
    top_dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="top_att_dark_toggle")
    if top_dark_toggle != st.session_state.get("dark_mode", False):
        st.session_state["dark_mode"] = top_dark_toggle
        st.rerun()

# Unique Hero Header Banner
st.markdown("""
<div class="sky-hero-banner">
    <div class="sky-brand-title">📸 AI FACE RECOGNITION SYSTEM</div>
    <div class="sky-brand-tagline">AI Classroom Attendance Capture & Database Management Portal</div>
</div>
""", unsafe_allow_html=True)

# Helper function to convert streamlit file input to BGR numpy array
def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# FACULTY AUTHENTICATION SCREEN
# ==============================================================================
if not st.session_state["faculty_logged_in"]:
    st.markdown('<div class="sub-header">Faculty Login Required to capture classroom attendance.</div>', unsafe_allow_html=True)
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("### 🔐 Faculty Login")
        with st.form("faculty_login_form"):
            username = st.text_input("Username", value="faculty")
            password = st.text_input("Password", type="password", value="password123")
            login_btn = st.form_submit_button("Login to Attendance Portal", use_container_width=True)
            
            if login_btn:
                faculty = database.verify_faculty(username, password)
                if faculty:
                    st.session_state["faculty_logged_in"] = True
                    st.session_state["faculty_user"] = faculty
                    st.success(f"Welcome back, {faculty['name']}!")
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Use any registered faculty account (Default: `faculty` / `password123`).")

# ==============================================================================
# AUTHENTICATED FACULTY DASHBOARD
# ==============================================================================
else:
    faculty_name = st.session_state["faculty_user"]["name"] if st.session_state["faculty_user"] else "Faculty Admin"
    
    header_col, logout_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"### 👋 Welcome back, **{faculty_name}**")
    with logout_col:
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["faculty_logged_in"] = False
            st.session_state["faculty_user"] = None
            st.rerun()

    st.markdown("---")
    
    # 3 Main Tabs Layout
    att_tab1, att_tab2, att_tab3 = st.tabs([
        "📸 Capture Attendance", 
        "✏️ Manage Attendance", 
        "👥 Manage Registered Students"
    ])
    
    # ==========================================================================
    # TAB 1: CAPTURE ATTENDANCE (Photo Capture, Present Table, PDF & Excel Export)
    # ==========================================================================
    with att_tab1:
        st.subheader("📸 Classroom Attendance Photo Capture")
        st.markdown("Upload or live capture **up to 10 classroom group photos**. Faces detected across all photos will be combined.")
        
        slot_name = st.text_input("📌 Class / Session / Slot Name *", value="Slot A - Morning Class", key="input_slot_name")
        confidence_threshold = 0.50


        input_source = st.radio(
            "Choose Input Method:", 
            ["Upload Classroom Photo Files (1 to 10)", "Live Camera Multi-Snapshot (Capture up to 10)"], 
            horizontal=True
        )
        
        uploaded_files = []
        if input_source == "Upload Classroom Photo Files (1 to 10)":
            raw_files = st.file_uploader(
                "Upload Classroom Group Photo(s) (Select 1 to 10 JPG/PNG files at once)", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True
            )
            if raw_files:
                uploaded_files = raw_files[:10]
                if len(raw_files) > 10:
                    st.warning("⚠️ Limit reached. Only the first 10 photos will be processed.")
                st.info(f"📸 {len(uploaded_files)} photo file(s) selected.")
        else:
            st.markdown("#### 📷 Live Webcam Multi-Snapshot Capture (Up to 10)")
            c_cnt = len(st.session_state["cam_snapshots_list"])
            
            c_col1, c_col2 = st.columns([3, 1])
            with c_col1:
                cam_snap = st.camera_input(f"Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "📷 Max 10 Snapshots Captured")
                if cam_snap is not None and c_cnt < 10:
                    if not st.session_state["cam_snapshots_list"] or st.session_state["cam_snapshots_list"][-1].getvalue() != cam_snap.getvalue():
                        st.session_state["cam_snapshots_list"].append(cam_snap)
                        st.rerun()
            with c_col2:
                st.metric("Captured Snapshots", f"{c_cnt} / 10")
                if st.session_state["cam_snapshots_list"]:
                    if st.button("🗑️ Clear All Snapshots", use_container_width=True):
                        st.session_state["cam_snapshots_list"] = []
                        st.rerun()

            if st.session_state["cam_snapshots_list"]:
                st.markdown(f"**Captured Snapshots List ({len(st.session_state['cam_snapshots_list'])} Photos):**")
                cols = st.columns(min(len(st.session_state["cam_snapshots_list"]), 5))
                for idx, snap in enumerate(st.session_state["cam_snapshots_list"]):
                    with cols[idx % 5]:
                        st.image(snap, caption=f"Photo #{idx+1}", use_container_width=True)
                        
                uploaded_files = st.session_state["cam_snapshots_list"]

        if uploaded_files:
            if st.button(f"🚀 Process Attendance Across {len(uploaded_files)} Photo(s)", use_container_width=True, type="primary"):
                registered_students = database.get_all_students()
                
                if len(registered_students) == 0:
                    st.warning("⚠️ No students registered in database! Register students in Website 1 (Registration Portal) first.")
                else:
                    with st.spinner(f"Analyzing {len(uploaded_files)} classroom photo(s) with InsightFace AI..."):
                        union_present_regs = set()
                        annotated_images = []
                        total_faces_detected_count = 0
                        
                        for img_file in uploaded_files:
                            img_bgr = load_image_as_bgr(img_file)
                            if img_bgr is not None:
                                annotated_bgr, records, metrics = face_utils.recognize_faces_in_group(
                                    img_bgr, registered_students, threshold=confidence_threshold
                                )
                                total_faces_detected_count += metrics.get("total_detected", 0)
                                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                annotated_images.append(annotated_rgb)
                                
                                for r in records:
                                    if r["Status"] == "Present":
                                        union_present_regs.add(r["Register No"])
                        
                        # Build combined final records
                        combined_records = []
                        for s in registered_students:
                            reg = s["register_no"]
                            is_present = reg in union_present_regs
                            combined_records.append({
                                "Register No": reg,
                                "Name": s["name"],
                                "Department": s["department"],
                                "Status": "Present" if is_present else "Absent"
                            })
                            # Force sync sliding toggle widget state for Tab 2
                            st.session_state[f"slide_toggle_{reg}"] = is_present
                            
                        p_count = len(union_present_regs)
                        a_count = len(registered_students) - p_count

                        
                        combined_metrics = {
                            "total_registered": len(registered_students),
                            "total_detected": total_faces_detected_count,
                            "present": p_count,
                            "absent": a_count,
                            "photos_processed": len(uploaded_files)
                        }
                        
                        now_str = datetime.datetime.now()
                        st.session_state["last_annotated_imgs"] = annotated_images
                        st.session_state["last_attendance_records"] = combined_records
                        st.session_state["last_metrics"] = combined_metrics
                        st.session_state["last_slot_name"] = slot_name.strip()
                        st.session_state["last_process_date"] = now_str.strftime("%Y-%m-%d")
                        st.session_state["last_process_time"] = now_str.strftime("%H:%M:%S")

        # Render Attendance Capture Results
        if "last_attendance_records" in st.session_state:
            st.markdown("---")
            st.subheader("📊 Captured Attendance Results")
            
            metrics = st.session_state["last_metrics"]
            records = st.session_state["last_attendance_records"]
            slot_title = st.session_state.get("last_slot_name", "Slot A")
            proc_date = st.session_state.get("last_process_date", datetime.datetime.now().strftime("%Y-%m-%d"))
            proc_time = st.session_state.get("last_process_time", datetime.datetime.now().strftime("%H:%M:%S"))
            
            df_records = pd.DataFrame(records)
            present_df = df_records[df_records["Status"] == "Present"].reset_index(drop=True)
            absent_df = df_records[df_records["Status"] == "Absent"].reset_index(drop=True)
            
            present_reg_list = present_df["Register No"].tolist() if not present_df.empty else []
            
            # Overview Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Registered Students (DB)", metrics["total_registered"])
            m2.metric("Detected Faces (Photos)", metrics["total_detected"])
            m3.metric("Present Count", len(present_df), delta=f"{(len(present_df)/max(metrics['total_registered'],1))*100:.1f}%")
            m4.metric("Absent Count", len(absent_df))
            
            # Captured Register Numbers Badges
            st.markdown("### 🎯 Captured Register Numbers (Present Students)")
            if present_reg_list:
                reg_html = "".join([f'<span class="reg-chip">✅ {reg}</span>' for reg in present_reg_list])
                st.markdown(reg_html, unsafe_allow_html=True)
            else:
                st.warning("No registered student faces matched in the uploaded photo(s).")

            # AI Recognition Images Display
            if "last_annotated_imgs" in st.session_state and st.session_state["last_annotated_imgs"]:
                st.markdown("### 🖼️ Visual AI Recognition Results")
                imgs = st.session_state["last_annotated_imgs"]
                if len(imgs) == 1:
                    st.image(imgs[0], caption="Green Bounding Box = Matched Student | Red = Unknown / Unmatched", use_container_width=True)
                else:
                    cols = st.columns(min(len(imgs), 3))
                    for idx, img in enumerate(imgs):
                        with cols[idx % len(cols)]:
                            st.image(img, caption=f"Photo #{idx+1} AI Recognition", use_container_width=True)

            # PRESENT STUDENTS TABLE ONLY (As requested: "here will only show the present students")
            st.markdown(f"### 📋 Present Students Roster ({len(present_df)} Present)")
            if not present_df.empty:
                display_present = present_df.copy()
                display_present.insert(0, "S.No", range(1, len(display_present) + 1))
                st.dataframe(display_present, use_container_width=True, hide_index=True)
            else:
                st.info("No present students to display.")

            st.markdown("---")

            # DUAL EXPORT OPTIONS (EXCEL AND PDF)
            st.subheader("📥 Export Official Attendance Reports")
            st.markdown("Download standard attendance reports with header metadata block (Slot, Date, Time, Total Strength, Present, Absent) and formatted table.")
            
            exp_p_count = len(df_records[df_records["Status"] == "Present"])
            exp_a_count = len(df_records[df_records["Status"] == "Absent"])
            exp_total = len(df_records)
            
            excel_bytes = export_utils.create_excel_report(
                slot_name=slot_title,
                faculty_name=faculty_name,
                date_str=proc_date,
                time_str=proc_time,
                total_strength=exp_total,
                present_count=exp_p_count,
                absent_count=exp_a_count,
                df_records=df_records
            )
            
            pdf_bytes = export_utils.create_pdf_report(
                slot_name=slot_title,
                faculty_name=faculty_name,
                date_str=proc_date,
                time_str=proc_time,
                total_strength=exp_total,
                present_count=exp_p_count,
                absent_count=exp_a_count,
                df_records=df_records,
                annotated_images=st.session_state.get("last_annotated_imgs", [])
            )


            
            date_file_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.download_button(
                    label="📊 Download Attendance Excel Sheet (.xlsx)",
                    data=excel_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            with btn_col2:
                st.download_button(
                    label="📄 Download Official Attendance PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    # ==========================================================================
    # TAB 2: MANAGE ATTENDANCE (Sliding Toggle Bar for All Registered Students)
    # ==========================================================================
    with att_tab2:
        st.subheader("✏️ Manage & Override Session Attendance")
        st.markdown(
            "View all registered students. Students recognized from the photo are marked **Present** by default. "
            "Slide the toggle switch next to any student to change their status between **Present** and **Absent**."
        )
        
        all_db_students = database.get_all_students()
        
        if not all_db_students:
            st.warning("📂 No students registered in the database. Please register students in Website 1 first.")
        else:
            # Ensure last_attendance_records exists in session state
            if "last_attendance_records" not in st.session_state:
                # Default initial state: all absent until photo processed
                st.session_state["last_attendance_records"] = [
                    {
                        "Register No": s["register_no"],
                        "Name": s["name"],
                        "Department": s["department"],
                        "Status": "Absent"
                    }
                    for s in all_db_students
                ]
                st.session_state["last_metrics"] = {
                    "total_registered": len(all_db_students),
                    "total_detected": 0,
                    "present": 0,
                    "absent": len(all_db_students)
                }

            # Map current session state records by register_no
            current_records = st.session_state["last_attendance_records"]
            record_dict = {r["Register No"]: r for r in current_records}
            
            # Ensure any newly registered student in DB is included
            updated_records = []
            for s in all_db_students:
                reg = s["register_no"]
                if reg in record_dict:
                    updated_records.append(record_dict[reg])
                else:
                    updated_records.append({
                        "Register No": reg,
                        "Name": s["name"],
                        "Department": s["department"],
                        "Status": "Absent"
                    })
            st.session_state["last_attendance_records"] = updated_records

            # Calculate metrics
            present_count = sum(1 for r in updated_records if r["Status"] == "Present")
            absent_count = len(updated_records) - present_count
            
            # Update metrics object
            st.session_state["last_metrics"]["present"] = present_count
            st.session_state["last_metrics"]["absent"] = absent_count
            st.session_state["last_metrics"]["total_registered"] = len(updated_records)

            # Metrics Row
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total Registered Class Strength", len(updated_records))
            sm2.metric("Currently Present (Slide ON)", present_count, delta=f"{(present_count/max(len(updated_records),1))*100:.1f}%")
            sm3.metric("Currently Absent (Slide OFF)", absent_count)

            st.markdown("---")

            # Quick Bulk Action Controls
            b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
            with b_col1:
                if st.button("✅ Mark All as Present", use_container_width=True):
                    for r in updated_records:
                        r["Status"] = "Present"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = True
                    st.session_state["last_attendance_records"] = updated_records
                    st.rerun()
            with b_col2:
                if st.button("❌ Mark All as Absent", use_container_width=True):
                    for r in updated_records:
                        r["Status"] = "Absent"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = False
                    st.session_state["last_attendance_records"] = updated_records
                    st.rerun()
            with b_col3:
                search_term = st.text_input("🔍 Search Student:", placeholder="Filter by Name or Register Number...", key="tab2_slide_search")


            st.markdown("---")
            st.markdown("#### 📜 Registered Students Attendance Sliding Toggles")

            # Table Header
            hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5 = st.columns([0.6, 1.5, 2, 2, 1.5])
            with hdr_col1:
                st.markdown("**S.No**")
            with hdr_col2:
                st.markdown("**Register No**")
            with hdr_col3:
                st.markdown("**Student Name**")
            with hdr_col4:
                st.markdown("**Department**")
            with hdr_col5:
                st.markdown("**Attendance (Slide Toggle)**")

            st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #334155;'/>", unsafe_allow_html=True)

            status_changed = False
            
            # Filter records if search query entered
            filtered_recs = updated_records
            if search_term.strip():
                st_q = search_term.strip().lower()
                filtered_recs = [r for r in updated_records if (st_q in r["Register No"].lower() or st_q in r["Name"].lower())]

            for idx, r in enumerate(filtered_recs):
                reg = r["Register No"]
                name = r["Name"]
                dept = r["Department"]
                curr_status = r["Status"]
                is_present = (curr_status == "Present")

                c1, c2, c3, c4, c5 = st.columns([0.6, 1.5, 2, 2, 1.5])
                with c1:
                    st.write(f"**{idx + 1}**")
                with c2:
                    st.code(reg, language="text")
                with c3:
                    st.write(f"**{name}**")
                with c4:
                    st.caption(dept)
                with c5:
                    # Sliding toggle bar widget
                    toggle_val = st.toggle(
                        "✅ Present" if is_present else "❌ Absent",
                        value=is_present,
                        key=f"slide_toggle_{reg}"
                    )
                    
                    if toggle_val != is_present:
                        r["Status"] = "Present" if toggle_val else "Absent"
                        status_changed = True

            if status_changed:
                st.session_state["last_attendance_records"] = updated_records
                st.rerun()


    # ==========================================================================
    # TAB 3: MANAGE REGISTERED STUDENTS (Database Edit Name/Dept & Delete)
    # ==========================================================================
    with att_tab3:
        st.subheader("👥 Registered Students Directory & Management")
        st.markdown("Search, view, modify student names/departments, or delete registered profiles in database.")
        
        all_students = database.get_all_students()
        
        if not all_students:
            st.warning("📂 No students registered yet in central database.")
        else:
            dept_counts = {}
            for s in all_students:
                dept = s.get("department", "Unknown")
                dept_counts[dept] = dept_counts.get(dept, 0) + 1

            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Total Registered", len(all_students))
            sm2.metric("Departments", len(dept_counts))
            sm3.metric("Largest Dept", max(dept_counts, key=dept_counts.get) if dept_counts else "N/A")

            st.markdown("---")
            
            # View / Search Section
            s_col1, s_col2 = st.columns([2, 1])
            with s_col1:
                search_q = st.text_input(
                    "🔍 Search by Name or Register Number:",
                    placeholder="Search by name or reg no...",
                    key="tab3_search_query"
                )
            with s_col2:
                available_depts = ["All Departments"] + sorted(list(dept_counts.keys()))
                selected_dept = st.selectbox(
                    "🏢 Filter by Department:",
                    available_depts,
                    key="tab3_dept_filter"
                )

            filtered_students = []
            for s in all_students:
                m_dept = (selected_dept == "All Departments") or (s["department"] == selected_dept)
                sq = search_q.strip().lower()
                m_q = (not sq) or (sq in s["name"].lower()) or (sq in s["register_no"].lower())
                if m_dept and m_q:
                    filtered_students.append({
                        "Register No": s["register_no"],
                        "Full Name": s["name"],
                        "Department": s["department"]
                    })

            st.markdown(f"#### 📜 Registered Roster ({len(filtered_students)} of {len(all_students)})")
            if filtered_students:
                df_filt = pd.DataFrame(filtered_students)
                st.dataframe(df_filt, use_container_width=True, hide_index=True)
                
                csv_bytes = df_filt.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Registered Students Roster (CSV)",
                    data=csv_bytes,
                    file_name="Registered_Students_Roster.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("ℹ️ No matching students found.")

            st.markdown("---")
            # FACULTY REGISTRATION MANAGEMENT: EDIT & DELETE
            st.subheader("🛠️ Faculty Student Profile Management: Edit or Delete")
            
            student_map = {f"{s['register_no']} - {s['name']} ({s['department']})": s for s in all_students}
            selected_option = st.selectbox(
                "Select a Student to Edit or Delete:",
                list(student_map.keys()),
                key="select_student_manage_tab3"
            )
            
            if selected_option:
                target_student = student_map[selected_option]
                target_reg = target_student["register_no"]
                
                edit_col, delete_col = st.columns([1.2, 1], gap="large")
                
                with edit_col:
                    st.markdown("#### ✏️ Modify Student Details")
                    with st.form(f"edit_form_tab3_{target_reg}"):
                        st.info(f"Editing Register Number: **{target_reg}**")
                        new_name = st.text_input("Full Name", value=target_student["name"])
                        
                        all_dept_options = [
                            "Computer Science & Engineering", 
                            "Information Technology", 
                            "Electrical & Electronics", 
                            "Mechanical Engineering", 
                            "Civil Engineering", 
                            "Biotechnology", 
                            "Other"
                        ]
                        curr_dept = target_student["department"]
                        default_idx = all_dept_options.index(curr_dept) if curr_dept in all_dept_options else len(all_dept_options) - 1
                        
                        new_dept = st.selectbox("Department", all_dept_options, index=default_idx)
                        save_btn = st.form_submit_button("💾 Save Updated Details", use_container_width=True)
                        
                        if save_btn:
                            if not new_name.strip():
                                st.error("⚠️ Full Name cannot be empty.")
                            else:
                                success = database.update_student(target_reg, new_name, new_dept)
                                if success:
                                    st.success(f"✅ Successfully updated '{new_name}' ({target_reg}).")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to update student record.")
                                    
                with delete_col:
                    st.markdown("#### 🗑️ Delete Student Profile")
                    st.warning(f"Warning: Deleting **{target_student['name']}** ({target_reg}) will remove their facial profile permanently.")
                    
                    confirm_del = st.checkbox(f"Confirm deletion for {target_reg}", key=f"del_confirm_tab3_{target_reg}")
                    if st.button(f"🗑️ Permanently Delete {target_reg}", use_container_width=True, type="secondary"):
                        if not confirm_del:
                            st.error("⚠️ Please check the confirmation box above before deleting.")
                        else:
                            success = database.delete_student(target_reg)
                            if success:
                                st.success(f"🗑️ Student '{target_student['name']}' ({target_reg}) deleted successfully.")
                                st.rerun()
                            else:
                                st.error("❌ Failed to delete student from database.")
