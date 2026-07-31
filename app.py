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
    page_title="AI Face Recognition Attendance Portal",
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
    key="app_dark_mode_toggle_key"
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
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
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
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
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

# Sidebar Mode Selection
st.sidebar.image("https://img.icons8.com/color/96/facial-recognition.png", width=65)
st.sidebar.title("AI Attendance System")

portal_choice = st.sidebar.radio(
    "Select Website Portal:",
    [
        "🌐 Website 1: Registration Portal",
        "📸 Website 2: Attendance Portal"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("📸 **AI Facial Attendance System**\nCentral Database Connected.")

# Top Bar & Sidebar Theme Toggle Controls
top_col1, top_col2 = st.columns([3, 1])
with top_col2:
    top_dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False), key="top_app_dark_toggle")
    if top_dark_toggle != st.session_state.get("dark_mode", False):
        st.session_state["dark_mode"] = top_dark_toggle
        st.rerun()

# Unique Hero Header Banner
st.markdown("""
<div class="sky-hero-banner">
    <div class="sky-brand-title">📸 AI FACE RECOGNITION SYSTEM</div>
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
if portal_choice == "🌐 Website 1: Registration Portal":
    st.markdown('<div class="main-header">Website 1: Student & Faculty Registration Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Register student face profiles or create faculty accounts into the database.</div>', unsafe_allow_html=True)
    
    tab_student_reg, tab_faculty_reg = st.tabs(["🎓 Student Face Registration", "👨‍🏫 Faculty Account Registration"])

    with tab_student_reg:
        col1, col2 = st.columns([1.2, 1], gap="large")

        with col1:
            st.subheader("📋 Student Information")
            with st.form("app_student_reg_form", clear_on_submit=False):
                reg_no = st.text_input("Register Number *", placeholder="e.g. 24BCA8011")
                full_name = st.text_input("Full Name *", placeholder="e.g. Jane Doe")
                department = st.selectbox(
                    "Department *",
                    ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
                )
                
                st.markdown("### 📷 Face Photo Capture")
                input_source = st.radio("Choose Input Method:", ["Upload Photo File", "Live Webcam Snapshot"], horizontal=True, key="app_reg_input_src")
                
                photo_file = None
                if input_source == "Upload Photo File":
                    photo_file = st.file_uploader("Upload Clear Face Photo (JPG, PNG)", type=["jpg", "jpeg", "png"], key="app_reg_file_up")
                else:
                    photo_file = st.camera_input("Take a Snapshot", key="app_reg_cam_snap")
                
                submit_btn = st.form_submit_button("Submit Registration", use_container_width=True)

            if submit_btn:
                if not reg_no.strip() or not full_name.strip():
                    st.error("⚠️ Please fill in all required fields (Register Number and Full Name).")
                elif photo_file is None:
                    st.error("⚠️ Please upload or capture a photo.")
                elif database.student_exists(reg_no):
                    st.error(f"⚠️ Student with Register Number '{reg_no.strip()}' is already registered!")
                else:
                    with st.spinner("Processing image & extracting facial embeddings..."):
                        img_bgr = load_image_as_bgr(photo_file)
                        if img_bgr is None:
                            st.error("❌ Failed to decode image.")
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
                                    st.success(f"🎉 Success! Student '{full_name}' ({reg_no}) has been registered.")
                                else:
                                    st.error("❌ Database insertion failed.")
                            else:
                                st.error(f"❌ {message}")

        with col2:
            st.subheader("💡 Registration Guidelines")
            st.markdown("""
            - 👤 **Single Face**: Ensure only your face is present in the frame.
            - 💡 **Good Lighting**: Ensure face is clearly visible.
            - 👀 **Direct View**: Look straight into the camera.
            - 👓 **No Obstructions**: Avoid sunglasses or heavy masks.
            """)
            st.metric("Total Registered Students", len(database.get_all_students()))

    with tab_faculty_reg:
        f_col1, f_col2 = st.columns([1.2, 1], gap="large")
        
        with f_col1:
            st.subheader("🔑 New Faculty Account Registration")
            with st.form("app_fac_reg_form", clear_on_submit=True):
                f_name = st.text_input("Full Name *", placeholder="Dr. Alan Turing")
                f_username = st.text_input("Username *", placeholder="alan_turing")
                f_password = st.text_input("Password *", type="password")
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
                        st.success(f"🎉 Faculty account created for '{f_name}'!")
                    else:
                        st.error("❌ Database error creating faculty account.")
                        
        with f_col2:
            st.subheader("👥 Registered Faculty Accounts")
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
            st.markdown("### 🔐 Faculty Login")
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
                        st.error("❌ Invalid Username or Password. (Default: `faculty` / `password123`).")
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
        
        att_tab1, att_tab2, att_tab3 = st.tabs(["📸 Capture Attendance", "✏️ Manage Attendance", "👥 Manage Registered Students"])

        with att_tab1:
            st.subheader("📸 Classroom Attendance Photo Capture")
            slot_name = st.text_input("📌 Class / Session / Slot Name *", value="Slot A - Morning Class", key="app_slot_name")
            confidence_threshold = 0.50


            input_source = st.radio("Choose Input Method:", ["Upload Classroom Photo Files (1 to 10)", "Live Camera Multi-Snapshot (Capture up to 10)"], horizontal=True, key="app_att_input_src")
            
            uploaded_files = []
            if input_source == "Upload Classroom Photo Files (1 to 10)":
                raw_files = st.file_uploader("Upload Classroom Group Photo(s) (1 to 10 JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="app_att_up_files")
                if raw_files:
                    uploaded_files = raw_files[:10]
            else:
                c_cnt = len(st.session_state["cam_snapshots_list"])
                c_col1, c_col2 = st.columns([3, 1])
                with c_col1:
                    cam_snap = st.camera_input(f"Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "📷 Max 10 Snapshots Captured", key="app_att_cam_input")
                    if cam_snap is not None and c_cnt < 10:
                        if not st.session_state["cam_snapshots_list"] or st.session_state["cam_snapshots_list"][-1].getvalue() != cam_snap.getvalue():
                            st.session_state["cam_snapshots_list"].append(cam_snap)
                            st.rerun()
                with c_col2:
                    st.metric("Captured Snapshots", f"{c_cnt} / 10")
                    if st.session_state["cam_snapshots_list"]:
                        if st.button("🗑️ Clear Snapshots", use_container_width=True, key="app_clear_snaps"):
                            st.session_state["cam_snapshots_list"] = []
                            st.rerun()
                uploaded_files = st.session_state["cam_snapshots_list"]

            if uploaded_files:
                if st.button(f"🚀 Process Attendance Across {len(uploaded_files)} Photo(s)", use_container_width=True, type="primary", key="app_btn_proc"):
                    registered_students = database.get_all_students()
                    if len(registered_students) == 0:
                        st.warning("⚠️ No students registered in database!")
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
                            
                            now_str = datetime.datetime.now()
                            st.session_state["last_annotated_imgs"] = annotated_images
                            st.session_state["last_attendance_records"] = combined_records
                            st.session_state["last_metrics"] = {
                                "total_registered": len(registered_students),
                                "total_detected": total_faces_detected_count,
                                "present": p_count,
                                "absent": a_count
                            }
                            st.session_state["last_slot_name"] = slot_name.strip()
                            st.session_state["last_process_date"] = now_str.strftime("%Y-%m-%d")
                            st.session_state["last_process_time"] = now_str.strftime("%H:%M:%S")

            if "last_attendance_records" in st.session_state:
                st.markdown("---")
                metrics = st.session_state["last_metrics"]
                records = st.session_state["last_attendance_records"]
                slot_title = st.session_state.get("last_slot_name", "Slot A")
                proc_date = st.session_state.get("last_process_date", datetime.datetime.now().strftime("%Y-%m-%d"))
                proc_time = st.session_state.get("last_process_time", datetime.datetime.now().strftime("%H:%M:%S"))
                
                df_records = pd.DataFrame(records)
                present_df = df_records[df_records["Status"] == "Present"].reset_index(drop=True)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Registered Students", metrics["total_registered"])
                m2.metric("Detected Faces", metrics["total_detected"])
                m3.metric("Present Count", len(present_df), delta=f"{(len(present_df)/max(metrics['total_registered'],1))*100:.1f}%")
                m4.metric("Absent Count", len(df_records) - len(present_df))

                if "last_annotated_imgs" in st.session_state and st.session_state["last_annotated_imgs"]:
                    st.markdown("### 🖼️ Visual AI Recognition Results")
                    imgs = st.session_state["last_annotated_imgs"]
                    if len(imgs) == 1:
                        st.image(imgs[0], caption="Green Bounding Box = Matched Student | Red = Unknown", use_container_width=True)
                    else:
                        cols = st.columns(min(len(imgs), 3))
                        for idx, img in enumerate(imgs):
                            with cols[idx % len(cols)]:
                                st.image(img, caption=f"Photo #{idx+1}", use_container_width=True)

                st.markdown(f"### 📋 Present Students Roster ({len(present_df)} Present)")
                if not present_df.empty:
                    display_present = present_df.copy()
                    display_present.insert(0, "S.No", range(1, len(display_present) + 1))
                    st.dataframe(display_present, use_container_width=True, hide_index=True)
                else:
                    st.info("No present students to display.")

                st.markdown("---")
                st.subheader("📥 Export Official Attendance Reports")
                
                exp_p_count = len(df_records[df_records["Status"] == "Present"])
                exp_a_count = len(df_records) - exp_p_count
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

        with att_tab2:
            st.subheader("✏️ Manage & Override Session Attendance")
            st.markdown("Slide the toggle switch next to any student to change their status between **Present** and **Absent**.")
            
            all_db_students = database.get_all_students()
            if not all_db_students:
                st.warning("📂 No students registered in database.")
            else:
                if "last_attendance_records" not in st.session_state:
                    st.session_state["last_attendance_records"] = [
                        {"Register No": s["register_no"], "Name": s["name"], "Department": s["department"], "Status": "Absent"}
                        for s in all_db_students
                    ]
                    st.session_state["last_metrics"] = {"total_registered": len(all_db_students), "total_detected": 0, "present": 0, "absent": len(all_db_students)}

                current_records = st.session_state["last_attendance_records"]
                record_dict = {r["Register No"]: r for r in current_records}
                
                updated_records = []
                for s in all_db_students:
                    reg = s["register_no"]
                    if reg in record_dict:
                        updated_records.append(record_dict[reg])
                    else:
                        updated_records.append({"Register No": reg, "Name": s["name"], "Department": s["department"], "Status": "Absent"})
                st.session_state["last_attendance_records"] = updated_records

                present_count = sum(1 for r in updated_records if r["Status"] == "Present")
                absent_count = len(updated_records) - present_count
                
                sm1, sm2, sm3 = st.columns(3)
                sm1.metric("Total Class Strength", len(updated_records))
                sm2.metric("Currently Present (Slide ON)", present_count)
                sm3.metric("Currently Absent (Slide OFF)", absent_count)

                st.markdown("---")
                b_col1, b_col2, b_col3 = st.columns([1, 1, 2])
                with b_col1:
                    if st.button("✅ Mark All as Present", use_container_width=True, key="app_tab2_all_p"):
                        for r in updated_records:
                            r["Status"] = "Present"
                            st.session_state[f"slide_toggle_{r['Register No']}"] = True
                        st.session_state["last_attendance_records"] = updated_records
                        st.rerun()
                with b_col2:
                    if st.button("❌ Mark All as Absent", use_container_width=True, key="app_tab2_all_a"):
                        for r in updated_records:
                            r["Status"] = "Absent"
                            st.session_state[f"slide_toggle_{r['Register No']}"] = False
                        st.session_state["last_attendance_records"] = updated_records
                        st.rerun()
                with b_col3:
                    search_term = st.text_input("🔍 Search Student:", placeholder="Filter by Name or Reg No...", key="app_tab2_search")

                st.markdown("---")
                st.markdown("#### 📜 Registered Students Attendance Sliding Toggles")
                
                hdr_col1, hdr_col2, hdr_col3, hdr_col4, hdr_col5 = st.columns([0.6, 1.5, 2, 2, 1.5])
                with hdr_col1: st.markdown("**S.No**")
                with hdr_col2: st.markdown("**Register No**")
                with hdr_col3: st.markdown("**Student Name**")
                with hdr_col4: st.markdown("**Department**")
                with hdr_col5: st.markdown("**Attendance (Slide Toggle)**")
                st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #334155;'/>", unsafe_allow_html=True)

                status_changed = False
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
                    with c1: st.write(f"**{idx + 1}**")
                    with c2: st.code(reg, language="text")
                    with c3: st.write(f"**{name}**")
                    with c4: st.caption(dept)
                    with c5:
                        toggle_val = st.toggle("✅ Present" if is_present else "❌ Absent", value=is_present, key=f"slide_toggle_{reg}")
                        if toggle_val != is_present:
                            r["Status"] = "Present" if toggle_val else "Absent"
                            status_changed = True

                if status_changed:
                    st.session_state["last_attendance_records"] = updated_records
                    st.rerun()

        with att_tab3:
            st.subheader("👥 Registered Students Directory & Management")
            all_students = database.get_all_students()
            if not all_students:
                st.warning("📂 No students registered yet.")
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
                st.subheader("🛠️ Faculty Student Profile Management: Edit or Delete")
                student_map = {f"{s['register_no']} - {s['name']} ({s['department']})": s for s in all_students}
                selected_option = st.selectbox("Select a Student to Edit or Delete:", list(student_map.keys()), key="app_tab3_select")
                
                if selected_option:
                    target_student = student_map[selected_option]
                    target_reg = target_student["register_no"]
                    edit_col, delete_col = st.columns([1.2, 1], gap="large")
                    
                    with edit_col:
                        st.markdown("#### ✏️ Modify Student Details")
                        with st.form(f"app_edit_form_{target_reg}"):
                            new_name = st.text_input("Full Name", value=target_student["name"])
                            all_dept_options = ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
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
                                        st.success(f"✅ Updated '{new_name}' ({target_reg}).")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to update student record.")
                                        
                    with delete_col:
                        st.markdown("#### 🗑️ Delete Student Profile")
                        confirm_del = st.checkbox(f"Confirm deletion for {target_reg}", key=f"app_del_confirm_{target_reg}")
                        if st.button(f"🗑️ Permanently Delete {target_reg}", use_container_width=True, type="secondary", key=f"app_btn_del_{target_reg}"):
                            if not confirm_del:
                                st.error("⚠️ Please check confirmation box first.")
                            else:
                                success = database.delete_student(target_reg)
                                if success:
                                    st.success(f"🗑️ Deleted {target_reg}.")
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete student.")
