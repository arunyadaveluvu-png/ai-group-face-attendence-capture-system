import streamlit as st
import cv2
import numpy as np
import pandas as pd
import io
import datetime
import calendar

import database
import face_utils
import export_utils

# Page Config
st.set_page_config(
    page_title="AI Face Recognition Attendance Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Toggle Control with Instant Re-render
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

st.sidebar.markdown("### Theme Selector")
toggle_val = st.sidebar.toggle(
    "Enable Dark Mode",
    value=st.session_state["dark_mode"],
    key="app_dark_mode_toggle_key"
)

if toggle_val != st.session_state["dark_mode"]:
    st.session_state["dark_mode"] = toggle_val
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Database Control")
st.sidebar.metric("Registered Students", len(database.get_all_students()))
if st.sidebar.button("Sync Registered Students", use_container_width=True, key="app_sync_db_btn"):
    st.toast("Database synced!")
    st.rerun()

# Apply Clean Professional CSS Overrides
if st.session_state["dark_mode"]:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, section.main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #020617 !important;
            background: radial-gradient(ellipse at 50% -20%, #0c4a6e 0%, #0f172a 60%, #020617 100%) !important;
            color: #f8fafc !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

        [data-testid="stSidebar"], [data-testid="stSidebarContent"], section[data-testid="stSidebar"] {
            background-color: #0f172a !important;
            border-right: 1px solid rgba(56, 189, 248, 0.3) !important;
        }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }

        .sky-hero-banner {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 60%, #0f172a 100%) !important;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.2) !important;
            border-top: 3px solid #38bdf8 !important;
            margin-bottom: 1.8rem;
        }
        .sky-brand-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .sky-brand-tagline {
            font-size: 0.9rem;
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

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: rgba(15, 23, 42, 0.8) !important;
            padding: 6px;
            border-radius: 8px;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            background-color: transparent !important;
            border-radius: 6px;
            color: #94a3b8 !important;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0 20px;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0284c7 !important;
            color: #ffffff !important;
        }
        .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
            color: #ffffff !important;
        }

        [data-testid="stMetric"], .sky-card {
            background: #1e293b !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-top: 3px solid #38bdf8 !important;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #38bdf8 !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            font-size: 0.82rem !important;
            color: #cbd5e1 !important;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background-color: #0369a1 !important;
            box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5) !important;
        }

        input, select, textarea, [data-baseweb="select"] {
            border-radius: 6px !important;
            border: 1px solid rgba(56, 189, 248, 0.4) !important;
            background-color: #1e293b !important;
            color: #f8fafc !important;
        }

        /* Calendar Green/Red Date Buttons */
        .stButton button[key^="app_cal_btn_"] { background: #15803d !important; color: #fff !important; }
        .stButton button[key^="app_cal_btn_dis_"] { background: #7f1d1d !important; color: #fff !important; }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .sky-brand-title { font-size: 1.5rem; }
            .sky-brand-tagline { font-size: 0.75rem; }
            .stTabs [data-baseweb="tab"] { font-size: 0.72rem; padding: 0 10px; height: 36px; }
            .stButton button, .stDownloadButton button { padding: 0.4rem 0.8rem !important; font-size: 0.82rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
            .stButton button { min-width: 28px !important; }
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, .stApp, section.main, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: #ffffff !important;
            background: radial-gradient(ellipse at 50% -20%, #bae6fd 0%, #f0f9ff 45%, #ffffff 100%) !important;
            color: #0f172a !important;
        }

        #MainMenu, footer, header {visibility: hidden;}

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
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(2, 132, 199, 0.2) !important;
            border-top: 3px solid #38bdf8 !important;
            margin-bottom: 1.8rem;
        }
        .sky-brand-title {
            font-size: 2.1rem;
            font-weight: 800;
            color: #ffffff !important;
            letter-spacing: -0.02em;
            margin: 0;
        }
        .sky-brand-tagline {
            font-size: 0.9rem;
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
            gap: 8px;
            background-color: #e0f2fe !important;
            padding: 6px;
            border-radius: 8px;
            border: 1px solid #7dd3fc !important;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            background-color: transparent !important;
            border-radius: 6px;
            color: #0369a1 !important;
            font-weight: 600;
            font-size: 0.9rem;
            padding: 0 20px;
            border: none !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0284c7 !important;
            color: #ffffff !important;
        }
        .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
            color: #ffffff !important;
        }

        [data-testid="stMetric"], .sky-card {
            background: #ffffff !important;
            border: 1px solid #7dd3fc !important;
            border-top: 3px solid #0284c7 !important;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.1) !important;
        }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #0284c7 !important;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
            font-size: 0.82rem !important;
            color: #475569 !important;
            font-weight: 700;
            text-transform: uppercase;
        }

        .stButton button, .stDownloadButton button {
            background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.25) !important;
        }
        .stButton button:hover, .stDownloadButton button:hover {
            background-color: #0369a1 !important;
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.4) !important;
        }

        input, select, textarea, [data-baseweb="select"] {
            border-radius: 6px !important;
            border: 1px solid #7dd3fc !important;
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        /* Calendar Green/Red Date Buttons */
        .stButton button[key^="app_cal_btn_"] { background: #15803d !important; color: #fff !important; }
        .stButton button[key^="app_cal_btn_dis_"] { background: #7f1d1d !important; color: #fff !important; }

        /* Mobile Responsive */
        @media (max-width: 768px) {
            .sky-brand-title { font-size: 1.5rem; }
            .sky-brand-tagline { font-size: 0.75rem; }
            .stTabs [data-baseweb="tab"] { font-size: 0.72rem; padding: 0 10px; height: 36px; }
            .stButton button, .stDownloadButton button { padding: 0.4rem 0.8rem !important; font-size: 0.82rem !important; }
            [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
            .stButton button { min-width: 28px !important; }
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

# Sidebar Mode Selection
st.sidebar.title("AI Attendance System")

portal_choice = st.sidebar.radio(
    "Select Portal:",
    [
        "Website 1: Registration Portal",
        "Website 2: Attendance Portal"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("AI Facial Attendance System\nCentral Database Connected.")

# Unique Hero Header Banner
st.markdown("""
<div class="sky-hero-banner">
    <div class="sky-brand-title">AI FACE RECOGNITION SYSTEM</div>
    <div class="sky-brand-tagline">Official AI Facial Attendance & Database Management Portal</div>
</div>
""", unsafe_allow_html=True)

def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# WEBSITE 1: REGISTRATION PORTAL
# ==============================================================================
if portal_choice == "Website 1: Registration Portal":
    st.markdown('<div class="main-header">Website 1: Student & Faculty Registration Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Register student face profiles or create faculty accounts into the database.</div>', unsafe_allow_html=True)
    
    tab_student_reg, tab_faculty_reg = st.tabs(["Student Face Registration", "Faculty Account Registration"])

    with tab_student_reg:
        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.subheader("Student Information")
            with st.form("app_student_reg_form", clear_on_submit=False):
                reg_no = st.text_input("Register Number *", placeholder="e.g. 24BCA8011")
                full_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
                department = st.selectbox(
                    "Department *",
                    ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
                )
                
                st.markdown("### Face Photo Capture")
                input_source = st.radio("Choose Input Method:", ["Upload Photo", "Live Webcam Snapshot"], horizontal=True, key="app_reg_input_src")
                
                photo_file = None
                if input_source == "Upload Photo":
                    photo_file = st.file_uploader("Upload Clear Face Photo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="app_reg_file_up")
                else:
                    photo_file = st.camera_input("Take a Snapshot", key="app_reg_cam_snap")
                
                submit_btn = st.form_submit_button("Submit Registration", use_container_width=True)

            if submit_btn:
                if not reg_no.strip() or not full_name.strip():
                    st.error("Please fill in all required fields (Register Number and Full Name).")
                elif photo_file is None:
                    st.error("Please upload or capture a photo.")
                elif database.student_exists(reg_no):
                    st.error(f"Student with Register Number '{reg_no.strip()}' is already registered!")
                else:
                    with st.spinner("Processing image & extracting facial embeddings..."):
                        img_bgr = load_image_as_bgr(photo_file)
                        if img_bgr is None:
                            st.error("Failed to decode image.")
                        else:
                            embedding, status_code, message = face_utils.extract_single_face_embedding(img_bgr)
                            if status_code == "NO_FACE":
                                st.error(f"{message}")
                            elif status_code == "MULTIPLE_FACES":
                                st.warning(f"{message}")
                            elif status_code == "SUCCESS":
                                success = database.add_student(reg_no, full_name, department, embedding)
                                if success:
                                    st.success(f"Success! Student '{full_name}' ({reg_no}) has been registered.")
                                else:
                                    st.error("Database insertion failed.")
                            else:
                                st.error(f"{message}")

        with col2:
            st.subheader("Registration Guidelines")
            st.markdown("""
            - Single Face: Ensure only your face is present in the frame.
            - Good Lighting: Ensure face is clearly visible.
            - Direct View: Look straight into the camera.
            - No Obstructions: Avoid sunglasses or heavy masks.
            """)
            st.metric("Total Registered Students", len(database.get_all_students()))

    with tab_faculty_reg:
        f_col1, f_col2 = st.columns([1.2, 1], gap="large")
        
        with f_col1:
            st.subheader("New Faculty Account Registration")
            with st.form("app_fac_reg_form", clear_on_submit=True):
                f_name = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
                f_username = st.text_input("Username *", placeholder="alan_turing")
                f_password = st.text_input("Password *", type="password")
                f_confirm_pw = st.text_input("Confirm Password *", type="password")
                f_submit = st.form_submit_button("Register Faculty Account", use_container_width=True)
                
            if f_submit:
                if not f_name.strip() or not f_username.strip() or not f_password.strip():
                    st.error("All fields are required.")
                elif f_password != f_confirm_pw:
                    st.error("Passwords do not match!")
                elif database.faculty_exists(f_username):
                    st.error(f"Username '{f_username.strip()}' is already taken.")
                else:
                    success = database.add_faculty(f_username, f_password, f_name)
                    if success:
                        st.success(f"Faculty account created for '{f_name}'!")
                    else:
                        st.error("Database error creating faculty account.")
                        
        with f_col2:
            st.subheader("Registered Faculty Accounts")
            faculty_list = database.get_all_faculty()
            st.metric("Total Faculty Accounts", len(faculty_list))
            if faculty_list:
                st.dataframe(pd.DataFrame(faculty_list), use_container_width=True, hide_index=True)

# ==============================================================================
# WEBSITE 2: ATTENDANCE PORTAL
# ==============================================================================
else:
    st.markdown('<div class="main-header">Website 2: Classroom Attendance & Management</div>', unsafe_allow_html=True)
    
    if not st.session_state["faculty_logged_in"]:
        st.markdown('<div class="sub-header">Faculty Login Required to capture attendance and manage student database.</div>', unsafe_allow_html=True)
        col_center = st.columns([1, 2, 1])[1]
        with col_center:
            st.markdown("### Faculty Login")
            with st.form("app_fac_login"):
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
                        st.error("Invalid Username or Password. (Default: faculty / password123).")
    else:
        faculty_name = st.session_state["faculty_user"]["name"] if st.session_state["faculty_user"] else "Faculty Admin"
        header_col, logout_col = st.columns([4, 1])
        with header_col:
            st.markdown(f"### Welcome back, **{faculty_name}**")
        with logout_col:
            if st.button("Logout", use_container_width=True):
                st.session_state["faculty_logged_in"] = False
                st.session_state["faculty_user"] = None
                st.rerun()

        st.markdown("---")
        
        att_tab1, att_tab2, att_tab3, att_tab4 = st.tabs([
            "Capture Attendance",
            "Edit Attendance",
            "All-Time Attendance History",
            "Manage Registered Students"
        ])

        # TAB 1: CAPTURE ATTENDANCE ONLY
        with att_tab1:
            st.subheader("Classroom Attendance Photo Capture & Processing")
            slot_name = st.text_input("Class / Session / Slot Name *", value="Slot A - Morning Class", key="app_slot_name")
            confidence_threshold = 0.50

            input_source = st.radio("Choose Input Method:", ["Upload Classroom Photo(s) (1 to 10)", "Live Camera Multi-Snapshot (Capture up to 10)"], horizontal=True, key="app_att_input_src")
            
            uploaded_files = []
            if input_source == "Upload Classroom Photo(s) (1 to 10)":
                raw_files = st.file_uploader("Upload Classroom Group Photo(s) (1 to 10 JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="app_att_up_files")
                if raw_files:
                    uploaded_files = raw_files[:10]
            else:
                c_cnt = len(st.session_state["cam_snapshots_list"])
                c_col1, c_col2 = st.columns([3, 1])
                with c_col1:
                    cam_snap = st.camera_input(f"Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "Max 10 Snapshots Captured", key="app_att_cam_input")
                    if cam_snap is not None and c_cnt < 10:
                        if not st.session_state["cam_snapshots_list"] or st.session_state["cam_snapshots_list"][-1].getvalue() != cam_snap.getvalue():
                            st.session_state["cam_snapshots_list"].append(cam_snap)
                            st.rerun()
                with c_col2:
                    st.metric("Captured Snapshots", f"{c_cnt} / 10")
                    if st.session_state["cam_snapshots_list"]:
                        if st.button("Clear Snapshots", use_container_width=True, key="app_clear_snaps"):
                            st.session_state["cam_snapshots_list"] = []
                            st.rerun()
                uploaded_files = st.session_state["cam_snapshots_list"]

            if uploaded_files:
                if st.button(f"Process Attendance Across {len(uploaded_files)} Photo(s)", use_container_width=True, type="primary", key="app_btn_proc"):
                    registered_students = database.get_all_students()
                    if len(registered_students) == 0:
                        st.warning("No students registered in database!")
                    else:
                        with st.spinner(f"Analyzing {len(uploaded_files)} classroom photo(s)..."):
                            union_present_regs = set()
                            annotated_images = []
                            total_faces_detected_count = 0
                            
                            for img_file in uploaded_files:
                                img_bgr = load_image_as_bgr(img_file)
                                if img_bgr is not None:
                                    annotated_bgr, records, metrics = face_utils.recognize_faces_in_group(img_bgr, registered_students, threshold=confidence_threshold)
                                    total_faces_detected_count += metrics.get("total_detected", 0)
                                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                    annotated_images.append(annotated_rgb)
                                    for r in records:
                                        if r["Status"] == "Present":
                                            union_present_regs.add(r["Register No"])
                            
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
                                st.session_state[f"slide_toggle_{reg}"] = is_present
                                
                            p_count = len(union_present_regs)
                            a_count = len(registered_students) - p_count
                            
                            now_dt = datetime.datetime.now()
                            date_str = now_dt.strftime("%Y-%m-%d")
                            time_str = now_dt.strftime("%H:%M:%S")

                            database.save_attendance_session(
                                slot_name=slot_name.strip(),
                                faculty_name=faculty_name,
                                date_str=date_str,
                                time_str=time_str,
                                total_students=len(registered_students),
                                present_count=p_count,
                                absent_count=a_count,
                                records=combined_records
                            )

                            st.session_state["last_annotated_imgs"] = annotated_images
                            st.session_state["last_attendance_records"] = combined_records
                            st.session_state["last_metrics"] = {
                                "total_registered": len(registered_students),
                                "total_detected": total_faces_detected_count,
                                "present": p_count,
                                "absent": a_count
                            }
                            st.session_state["last_slot_name"] = slot_name.strip()
                            st.session_state["last_process_date"] = date_str
                            st.session_state["last_process_time"] = time_str

                            st.success(f"Attendance processed successfully! Identified {p_count} present student(s). Switch to 'Edit Attendance' tab to view or adjust roster.")

            if "last_attendance_records" in st.session_state and "last_metrics" in st.session_state:
                st.markdown("---")
                st.subheader("Attendance Processing Results Summary")
                metrics = st.session_state.get("last_metrics", {})
                records = st.session_state.get("last_attendance_records", [])
                if metrics and records:
                    present_df = pd.DataFrame([r for r in records if r["Status"] == "Present"])

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Registered Students", metrics["total_registered"])
                m2.metric("Detected Faces", metrics["total_detected"])
                m3.metric("Present Count", metrics["present"], delta=f"{(metrics['present']/max(metrics['total_registered'],1))*100:.1f}%")
                m4.metric("Absent Count", metrics["absent"])

                if "last_annotated_imgs" in st.session_state and st.session_state["last_annotated_imgs"]:
                    st.markdown("### Visual Recognition Output")
                    imgs = st.session_state["last_annotated_imgs"]
                    if len(imgs) == 1:
                        st.image(imgs[0], caption="Green Bounding Box = Matched Student | Red = Unknown", use_container_width=True)
                    else:
                        cols = st.columns(min(len(imgs), 3))
                        for idx, img in enumerate(imgs):
                            with cols[idx % len(cols)]:
                                st.image(img, caption=f"Photo #{idx+1}", use_container_width=True)

                st.markdown("### Present Students Identified")
                if not present_df.empty:
                    disp_df = present_df[["Register No", "Name", "Department", "Status"]].copy()
                    disp_df.insert(0, "S.No", range(1, len(disp_df) + 1))
                    st.dataframe(disp_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No present students identified in photo data.")

        # TAB 2: EDIT ATTENDANCE (SLIDING TOGGLES & EXPORTS AT BOTTOM)
        with att_tab2:
            st.subheader("Edit Attendance Roster & Manual Toggle Overrides")
            st.markdown("Captured student face detections default to **Present**. Slide any toggle switch to adjust status between Present and Absent.")

            all_db_students = database.get_all_students()
            if not all_db_students:
                st.warning("No students registered in database.")
            else:
                existing_records = st.session_state.get("last_attendance_records", [])
                existing_map = {r["Register No"]: r for r in existing_records}
                
                records = []
                for s in all_db_students:
                    reg = s["register_no"]
                    if reg in existing_map:
                        records.append(existing_map[reg])
                    else:
                        records.append({
                            "Register No": reg,
                            "Name": s["name"],
                            "Department": s["department"],
                            "Status": "Absent"
                        })
                st.session_state["last_attendance_records"] = records
                present_records = [r for r in records if r["Status"] == "Present"]
                absent_records = [r for r in records if r["Status"] == "Absent"]

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Class Strength", len(records))
                m2.metric("Currently Present (Toggle ON)", len(present_records), delta=f"{(len(present_records)/max(len(records),1))*100:.1f}%")
                m3.metric("Currently Absent (Toggle OFF)", len(absent_records))

                st.markdown("---")
                b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
                with b_col1:
                    if st.button("Mark All as Present", use_container_width=True, key="app_tab2_all_p"):
                        for r in records:
                            r["Status"] = "Present"
                            st.session_state[f"slide_toggle_{r['Register No']}"] = True
                        st.session_state["last_attendance_records"] = records
                        st.rerun()
                with b_col2:
                    if st.button("Mark All as Absent", use_container_width=True, key="app_tab2_all_a"):
                        for r in records:
                            r["Status"] = "Absent"
                            st.session_state[f"slide_toggle_{r['Register No']}"] = False
                        st.session_state["last_attendance_records"] = records
                        st.rerun()
                with b_col3:
                    search_term = st.text_input("Search Student:", placeholder="Filter by Name or Reg No...", key="app_tab2_search")

                st.markdown("<hr style='margin: 10px 0; border-color: #334155;'/>", unsafe_allow_html=True)
                hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5 = st.columns([0.6, 1.5, 2, 2, 1.5])
                with hdr_col1: st.markdown("**S.No**")
                with hdr_col2: st.markdown("**Register No**")
                with hdr_col3: st.markdown("**Student Name**")
                with hdr_col4: st.markdown("**Department**")
                with hdr_col5: st.markdown("**Attendance Status (Slide Toggle)**")
                st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #334155;'/>", unsafe_allow_html=True)

                filtered_recs = records
                if search_term.strip():
                    st_q = search_term.strip().lower()
                    filtered_recs = [r for r in records if (st_q in r["Register No"].lower() or st_q in r["Name"].lower())]

                status_changed = False
                for idx, r in enumerate(filtered_recs):
                    reg = r["Register No"]
                    name = r["Name"]
                    dept = r["Department"]
                    curr_status = r["Status"]
                    is_present = (curr_status == "Present")

                    c1, c2, c3, c4, c5 = st.columns([0.6, 1.5, 2, 2, 1.5])
                    with c1: st.write(f"**{idx + 1}**")
                    with c2: st.code(reg, language="text")
                    with c3: st.write(f"**{name}**")
                    with c4: st.caption(dept)
                    with c5:
                        toggle_val = st.toggle("Present" if is_present else "Absent", value=is_present, key=f"slide_toggle_{reg}")
                        if toggle_val != is_present:
                            r["Status"] = "Present" if toggle_val else "Absent"
                            status_changed = True

                if status_changed:
                    st.session_state["last_attendance_records"] = records
                    st.rerun()

                st.markdown("---")
                st.subheader("Export Official Attendance Reports")
                st.markdown("Official PDF report contains **ONLY Present Students** along with Header metadata (Faculty Name, Slot Name, Total Strength, Present Count, Absent Count, Date & Time).")

                cur_slot = st.session_state.get("last_slot_name", "Slot A")
                cur_date = st.session_state.get("last_process_date", datetime.datetime.now().strftime("%Y-%m-%d"))
                cur_time = st.session_state.get("last_process_time", datetime.datetime.now().strftime("%H:%M:%S"))

                p_cnt = len(present_records)
                a_cnt = len(absent_records)
                tot_cnt = len(records)

                excel_bytes = export_utils.create_excel_report(
                    slot_name=cur_slot,
                    faculty_name=faculty_name,
                    date_str=cur_date,
                    time_str=cur_time,
                    total_strength=tot_cnt,
                    present_count=p_cnt,
                    absent_count=a_cnt,
                    df_records=pd.DataFrame(records)
                )

                pdf_bytes = export_utils.create_pdf_report(
                    slot_name=cur_slot,
                    faculty_name=faculty_name,
                    date_str=cur_date,
                    time_str=cur_time,
                    total_strength=tot_cnt,
                    present_count=p_cnt,
                    absent_count=a_cnt,
                    df_records=pd.DataFrame(records),
                    annotated_images=st.session_state.get("last_annotated_imgs", [])
                )

                date_file_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    st.download_button(
                        label="Download Official Attendance PDF (.pdf)",
                        data=pdf_bytes,
                        file_name=f"Attendance_Report_{date_file_str}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        key="app_down_pdf_edit"
                    )
                with btn_col2:
                    st.download_button(
                        label="Download Attendance Excel Sheet (.xlsx)",
                        data=excel_bytes,
                        file_name=f"Attendance_Report_{date_file_str}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        key="app_down_excel_edit"
                    )

        # TAB 3: ALL-TIME HISTORY (INTERACTIVE CALENDAR VIEW BY YEAR & MONTH)
        with att_tab3:
            st.subheader("Attendance History & Calendar View")
            st.markdown(f"Displaying attendance records captured by **{faculty_name}**.")

            saved_sessions = database.get_all_attendance_sessions(faculty_name=faculty_name)

            if not saved_sessions:
                st.info(f"No attendance sessions recorded for {faculty_name} yet.")
            else:
                sessions_by_date = {}
                for s in saved_sessions:
                    d = s["date_str"]
                    if d not in sessions_by_date:
                        sessions_by_date[d] = []
                    sessions_by_date[d].append(s)

                now_dt = datetime.datetime.now()
                current_yr = now_dt.year
                current_mo = now_dt.month

                db_years = sorted(list(set([int(s["date_str"].split("-")[0]) for s in saved_sessions if "-" in s["date_str"]] + [current_yr])), reverse=True)
                month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

                col_y, col_m, col_summary = st.columns([1.5, 1.5, 2])
                with col_y:
                    sel_year = st.selectbox("Select Year:", db_years, index=0, key="app_cal_sel_year")
                with col_m:
                    default_m_idx = current_mo - 1
                    sel_month_name = st.selectbox("Select Month:", month_names, index=default_m_idx, key="app_cal_sel_month")
                    sel_month_num = month_names.index(sel_month_name) + 1
                with col_summary:
                    st.metric("Total Saved Sessions", len(saved_sessions))

                st.markdown("---")
                st.markdown(f"### Interactive Calendar: {sel_month_name} {sel_year}")
                st.markdown("Click any active date button to open attendance records for that day.")

                days_header = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                hdr_cols = st.columns(7)
                for idx, day_name in enumerate(days_header):
                    with hdr_cols[idx]:
                        st.markdown(f"<div style='text-align:center; font-weight:700; color:#38bdf8; font-size:0.85rem;'>{day_name}</div>", unsafe_allow_html=True)

                month_matrix = calendar.monthcalendar(sel_year, sel_month_num)

                for week in month_matrix:
                    week_cols = st.columns(7)
                    for day_idx, day_num in enumerate(week):
                        with week_cols[day_idx]:
                            if day_num == 0:
                                st.markdown("<div style='height:44px;'></div>", unsafe_allow_html=True)
                            else:
                                date_fmt = f"{sel_year:04d}-{sel_month_num:02d}-{day_num:02d}"
                                has_recs = date_fmt in sessions_by_date
                                
                                if has_recs:
                                    sess_cnt = len(sessions_by_date[date_fmt])
                                    lbl = f"{day_num}"
                                    st.markdown(f"<div style='background:#15803d; border:2px solid #22c55e; border-radius:8px; text-align:center; padding:8px 4px; color:#fff; font-weight:700; font-size:0.95rem; margin-bottom:2px;'>{day_num}<br><span style='font-size:0.65rem; opacity:0.9;'>{sess_cnt} rec</span></div>", unsafe_allow_html=True)
                                    if st.button("View", key=f"app_cal_btn_{date_fmt}", use_container_width=True):
                                        st.session_state["app_selected_cal_date"] = date_fmt
                                        st.rerun()
                                else:
                                    st.markdown(f"<div style='background:#7f1d1d; border:2px solid #ef4444; border-radius:8px; text-align:center; padding:8px 4px; color:#fca5a5; font-weight:600; font-size:0.95rem; margin-bottom:2px; opacity:0.7;'>{day_num}</div>", unsafe_allow_html=True)

                st.markdown("---")
                active_date = st.session_state.get("app_selected_cal_date", None)

                if active_date:
                    st.markdown(f"### Attendance Records for Date: **{active_date}**")
                    date_sessions = sessions_by_date.get(active_date, [])

                    if not date_sessions:
                        st.info(f"No records stored for date {active_date}.")
                    else:
                        for s in date_sessions:
                            s_id = s["id"]
                            s_title = f"Slot: {s['slot_name']} (Time: {s['time_str']} | Faculty: {s['faculty_name']}) — Present: {s['present_count']}/{s['total_students']}"
                            
                            with st.expander(s_title, expanded=True):
                                info, logs = database.get_attendance_session_details(s_id)
                                if info:
                                    inf_c1, inf_c2, inf_c3, inf_c4 = st.columns(4)
                                    inf_c1.metric("Date & Time", f"{info['date_str']} {info['time_str']}")
                                    inf_c2.metric("Total Strength", info['total_students'])
                                    inf_c3.metric("Present Count", info['present_count'])
                                    inf_c4.metric("Absent Count", info['absent_count'])

                                    df_logs = pd.DataFrame(logs)
                                    st.markdown("#### Present Students Roster")
                                    df_logs_present = df_logs[df_logs["Status"] == "Present"].reset_index(drop=True)
                                    if not df_logs_present.empty:
                                        df_logs_present.insert(0, "S.No", range(1, len(df_logs_present) + 1))
                                        st.dataframe(df_logs_present, use_container_width=True, hide_index=True)
                                    else:
                                        st.info("No present students recorded for this session.")

                                    h_pdf_bytes = export_utils.create_pdf_report(
                                        slot_name=info["slot_name"],
                                        faculty_name=info["faculty_name"],
                                        date_str=info["date_str"],
                                        time_str=info["time_str"],
                                        total_strength=info["total_students"],
                                        present_count=info["present_count"],
                                        absent_count=info["absent_count"],
                                        df_records=df_logs
                                    )

                                    h_excel_bytes = export_utils.create_excel_report(
                                        slot_name=info["slot_name"],
                                        faculty_name=info["faculty_name"],
                                        date_str=info["date_str"],
                                        time_str=info["time_str"],
                                        total_strength=info["total_students"],
                                        present_count=info["present_count"],
                                        absent_count=info["absent_count"],
                                        df_records=df_logs
                                    )

                                    h_col1, h_col2, h_col3 = st.columns([2, 2, 1])
                                    with h_col1:
                                        st.download_button(
                                            label=f"Download PDF Report ({info['date_str']})",
                                            data=h_pdf_bytes,
                                            file_name=f"Attendance_{info['date_str']}_{s_id}.pdf",
                                            mime="application/pdf",
                                            use_container_width=True,
                                            key=f"app_cal_pdf_{s_id}"
                                        )
                                    with h_col2:
                                        st.download_button(
                                            label=f"Download Excel Sheet ({info['date_str']})",
                                            data=h_excel_bytes,
                                            file_name=f"Attendance_{info['date_str']}_{s_id}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                            use_container_width=True,
                                            key=f"app_cal_excel_{s_id}"
                                        )
                                    with h_col3:
                                        if st.button("Delete Session", key=f"app_cal_del_{s_id}", use_container_width=True):
                                            database.delete_attendance_session(s_id)
                                            st.session_state["app_selected_cal_date"] = None
                                            st.success("Session record deleted successfully.")
                                            st.rerun()
                else:
                    st.info("Click on any active date in the calendar above to view detailed attendance logs and export PDF/Excel reports.")

        # TAB 4: MANAGE REGISTERED STUDENTS DIRECTORY
        with att_tab4:
            st.subheader("Registered Students Directory & Management")
            all_students = database.get_all_students()
            if not all_students:
                st.warning("No students registered yet.")
            else:
                sm1, sm2 = st.columns(2)
                sm1.metric("Total Registered", len(all_students))
                sm2.metric("Central Database File", "attendance_system.db")
                st.markdown("---")
                
                filtered_students = [{
                    "Register No": s["register_no"],
                    "Full Name": s["name"],
                    "Department": s["department"]
                } for s in all_students]
                
                st.dataframe(pd.DataFrame(filtered_students), use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("Faculty Student Profile Management: Edit or Delete")
                student_map = {f"{s['register_no']} - {s['name']} ({s['department']})": s for s in all_students}
                selected_option = st.selectbox("Select a Student to Edit or Delete:", list(student_map.keys()), key="app_tab4_select")
                
                if selected_option:
                    target_student = student_map[selected_option]
                    target_reg = target_student["register_no"]
                    edit_col, delete_col = st.columns([1.2, 1], gap="large")
                    
                    with edit_col:
                        st.markdown("#### Modify Student Details")
                        with st.form(f"app_edit_form_{target_reg}"):
                            new_name = st.text_input("Full Name", value=target_student["name"])
                            all_dept_options = ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
                            curr_dept = target_student["department"]
                            default_idx = all_dept_options.index(curr_dept) if curr_dept in all_dept_options else len(all_dept_options) - 1
                            new_dept = st.selectbox("Department", all_dept_options, index=default_idx)
                            save_btn = st.form_submit_button("Save Updated Details", use_container_width=True)
                            
                            if save_btn:
                                if not new_name.strip():
                                    st.error("Full Name cannot be empty.")
                                else:
                                    success = database.update_student(target_reg, new_name, new_dept)
                                    if success:
                                        st.success(f"Updated details for {target_reg}!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update student.")
                    
                    with delete_col:
                        st.markdown("#### Delete Student Profile")
                        st.warning(f"Delete student {target_student['name']} ({target_reg})?")
                        if st.button(f"Delete Student {target_reg}", use_container_width=True, key="app_del_{target_reg}"):
                            success = database.delete_student(target_reg)
                            if success:
                                st.success(f"Student {target_reg} deleted.")
                                st.rerun()
                            else:
                                st.error("Failed to delete student.")
