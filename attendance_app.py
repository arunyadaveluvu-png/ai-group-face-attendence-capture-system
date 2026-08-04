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

# Default face recognition confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

# Page Configuration
st.set_page_config(
    page_title="Faculty Attendance Management System",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, .stApp {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    .stSidebar {
        background-color: #1e293b !important;
        border-right: 1px solid #334155 !important;
    }

    .main-header {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38bdf8 !important;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #94a3b8 !important;
        margin-bottom: 1.5rem;
    }

    /* Executive Metric Card */
    [data-testid="stMetric"], .card-box {
        background: #1e293b !important;
        border: 1px solid #334155 !important;
        border-top: 3px solid #0284c7 !important;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        color: #38bdf8 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem !important;
        color: #cbd5e1 !important;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* Clean Professional Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1e293b !important;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #334155 !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent !important;
        border-radius: 6px;
        color: #94a3b8 !important;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0 18px;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span {
        color: #ffffff !important;
    }

    /* Professional Executive Buttons */
    .stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #0284c7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.3rem !important;
        box-shadow: 0 2px 8px rgba(2, 132, 199, 0.3) !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton button:hover, .stDownloadButton button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #0369a1 !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.4) !important;
    }

    /* Inputs Styling */
    input, select, textarea, [data-baseweb="select"] {
        border-radius: 6px !important;
        border: 1px solid #334155 !important;
        background-color: #1e293b !important;
        color: #f8fafc !important;
    }

    /* Mobile Responsive Adjustments */
    @media (max-width: 768px) {
        .main-header {font-size: 1.6rem;}
        .sub-header {font-size: 0.85rem;}
        .stTabs [data-baseweb="tab"] {font-size: 0.75rem; padding: 0 12px; height: 36px;}
        .stButton button, .stDownloadButton button, div[data-testid="stFormSubmitButton"] > button {padding: 0.45rem 1rem; font-size: 0.85rem;}
        .stMetricValue {font-size: 1.4rem !important;}
        .stMetricLabel {font-size: 0.7rem !important;}
        /* Calendar button size reduction for mobile */
        .stButton button {min-width: 30px; font-size: 0.85rem;}
    }
    .stButton button[data-key^="cal_btn_"] {
        background-color: #15803d !important; /* green */
        color: #ffffff !important;
    }
    .stButton button[data-key^="cal_btn_dis_"] {
        background-color: #7f1d1d !important; /* red */
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Database Connection
database.init_db()

# Session State Initializations
if "faculty_logged_in" not in st.session_state:
    st.session_state["faculty_logged_in"] = False
if "faculty_user" not in st.session_state:
    st.session_state["faculty_user"] = None
if "cam_snapshots_list" not in st.session_state:
    st.session_state["cam_snapshots_list"] = []

# Sidebar Navigation Panel
st.sidebar.title("Faculty Portal")
st.sidebar.markdown("Automated Attendance System")
st.sidebar.markdown("---")
st.sidebar.info(
    "Connected Database: SQLite (attendance_system.db)\n\n"
    "Shared Central Database for Student Profiles and Attendance Logs."
)

def load_image_as_bgr(uploaded_file) -> np.ndarray:
    bytes_data = uploaded_file.getvalue()
    file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr

# ==============================================================================
# FACULTY AUTHENTICATION SCREEN
# ==============================================================================
if not st.session_state["faculty_logged_in"]:
    st.markdown('<div class="main-header">Faculty Attendance Management Portal</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Faculty authentication required to access attendance features.</div>', unsafe_allow_html=True)
    
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        st.markdown("### Faculty Login")
        with st.form("faculty_login_form"):
            username = st.text_input("Username", value="faculty")
            password = st.text_input("Password", type="password", value="password123")
            login_btn = st.form_submit_button("Login to Attendance Portal", use_container_width=True)
            
            if login_btn:
                faculty = database.verify_faculty(username, password)
                if faculty:
                    st.session_state["faculty_logged_in"] = True
                    st.session_state["faculty_user"] = faculty
                    st.success(f"Authentication successful. Welcome, {faculty['name']}.")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password. (Default: faculty / password123).")

# ==============================================================================
# AUTHENTICATED FACULTY DASHBOARD
# ==============================================================================
else:
    faculty_name = st.session_state["faculty_user"]["name"] if st.session_state["faculty_user"] else "Faculty Admin"
    
    header_col, logout_col = st.columns([4, 1])
    with header_col:
        st.markdown(f"### Welcome, **{faculty_name}**")
    with logout_col:
        if st.button("Logout", use_container_width=True):
            st.session_state["faculty_logged_in"] = False
            st.session_state["faculty_user"] = None
            st.rerun()

    with st.sidebar:
        st.markdown("### Central Database Sync")
        if database.is_cloud_mode():
            st.success("☁️ **Cloud DB Active** (Supabase)")
        else:
            st.warning("⚠️ **Local Mode** (Cloud Secrets missing in App Settings -> Secrets)")
            
        with st.expander("🚀 Push Local Data to Supabase"):
            url_cfg, hdrs_cfg = database.get_supabase_config()
            default_url = url_cfg if url_cfg else ""
            default_key = hdrs_cfg["apikey"] if hdrs_cfg else ""
            
            p_url_a = st.text_input("Supabase URL", value=default_url, placeholder="https://xyz.supabase.co", key="att_mig_url")
            p_key_a = st.text_input("Supabase Key", value=default_key, type="password", placeholder="eyJhbG...", key="att_mig_key")
            if st.button("Push Local Data Now", use_container_width=True, key="att_mig_push_btn"):
                if not p_url_a.strip() or not p_key_a.strip():
                    st.error("Enter URL & Key")
                else:
                    import migrate_to_supabase
                    with st.spinner("Uploading to Supabase..."):
                        migrate_to_supabase.migrate(p_url_a.strip(), p_key_a.strip())
                        st.toast("Data pushed to Supabase!", icon="🚀")
                        st.rerun()
            
        registered_cnt = len(database.get_all_students())
        st.metric("Total Registered Students", registered_cnt)
        if st.button("🔄 Sync Registered Students", use_container_width=True, key="att_sync_db_btn"):
            st.toast("Database re-synced!", icon="✅")
            st.rerun()

    tab_att_capture, tab_att_edit, tab_att_history, tab_students_mgmt = st.tabs([
        "Capture Attendance",
        "Edit Attendance",
        "All-Time Attendance History",
        "Manage Registered Students"
    ])

    # ==========================================================================
    # TAB 1: CAPTURE ATTENDANCE ONLY (FRONT PAGE CAPTURE & PROCESS)
    # ==========================================================================
    with tab_att_capture:
        st.subheader("Classroom Attendance Photo Capture & Automated Processing")
        st.markdown("Upload classroom group photos or take camera snapshots to process attendance.")
        
        slot_name = st.text_input("Class / Session / Slot Name *", value="Slot A - Morning Class", key="att_slot_name")
        confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD

        input_source = st.radio(
            "Select Photo Input Method:",
            ["📷 Mobile Real Camera / Upload Group Photo(s) (1 to 10)", "📹 Live Camera Snapshot"],
            horizontal=True,
            key="att_input_src"
        )
        
        uploaded_files = []
        if input_source == "📷 Mobile Real Camera / Upload Group Photo(s) (1 to 10)":
            st.caption("💡 **Mobile Tip:** Tapping below on your mobile device opens your phone's real camera app directly to snap classroom photos.")
            raw_files = st.file_uploader("Snap with Mobile Camera or Choose Image(s) (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="att_up_files")
            if raw_files:
                uploaded_files = raw_files[:10]
        else:
            c_cnt = len(st.session_state["cam_snapshots_list"])
            c_col1, c_col2 = st.columns([3, 1])
            with c_col1:
                cam_snap = st.camera_input(f"Take Snapshot #{c_cnt + 1}" if c_cnt < 10 else "Max 10 Snapshots Captured", key="att_cam_input")
                if cam_snap is not None and c_cnt < 10:
                    if not st.session_state["cam_snapshots_list"] or st.session_state["cam_snapshots_list"][-1].getvalue() != cam_snap.getvalue():
                        st.session_state["cam_snapshots_list"].append(cam_snap)
                        st.rerun()
            with c_col2:
                st.metric("Captured Snapshots", f"{c_cnt} / 10")
                if st.session_state["cam_snapshots_list"]:
                    if st.button("Clear Snapshots", use_container_width=True, key="att_clear_snaps"):
                        st.session_state["cam_snapshots_list"] = []
                        st.rerun()
            uploaded_files = st.session_state["cam_snapshots_list"]

        if uploaded_files:
            if st.button(f"Process Attendance Across {len(uploaded_files)} Photo(s)", use_container_width=True, type="primary", key="att_btn_proc"):
                registered_students = database.get_all_students()
                if len(registered_students) == 0:
                    st.warning("No students registered in database. Please register students first.")
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

                        session_id = database.save_attendance_session(
                            slot_name=slot_name.strip(),
                            faculty_name=faculty_name,
                            date_str=date_str,
                            time_str=time_str,
                            total_students=len(registered_students),
                            present_count=p_count,
                            absent_count=a_count,
                            records=combined_records
                        )
                        
                        st.session_state["last_session_id"] = session_id
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

        if "last_attendance_records" in st.session_state:
            st.markdown("---")
            st.subheader("Attendance Processing Results Summary")
            metrics = st.session_state["last_metrics"]
            records = st.session_state["last_attendance_records"]
            present_df = pd.DataFrame([r for r in records if r["Status"] == "Present"])

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Registered Students", metrics["total_registered"])
            m2.metric("Detected Faces", metrics["total_detected"])
            m3.metric("Present Count", metrics["present"], delta=f"{(metrics['present']/max(metrics['total_registered'],1))*100:.1f}%")
            m4.metric("Absent Count", metrics["absent"])

            if "last_annotated_imgs" in st.session_state and st.session_state["last_annotated_imgs"]:
                st.markdown("### Facial Identification Visual Recognition")
                imgs = st.session_state["last_annotated_imgs"]
                if len(imgs) == 1:
                    st.image(imgs[0], caption="Green Bounding Box = Matched Student | Red = Unmatched / Unknown", use_container_width=True)
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

    # ==========================================================================
    # TAB 2: EDIT ATTENDANCE (SLIDING TOGGLES & PDF/EXCEL EXPORTS)
    # ==========================================================================
    with tab_att_edit:
        st.subheader("Edit Attendance Roster & Manual Toggle Overrides")
        st.markdown("Captured student face detections default to **Present**. Slide the toggle switch next to any student to adjust their status between Present and Absent.")

        all_db_students = database.get_all_students()
        if not all_db_students:
            st.warning("No registered students found in database.")
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
            tb_col1, tb_col2, tb_col3 = st.columns([1, 1, 2])
            with tb_col1:
                if st.button("Mark All as Present", use_container_width=True, key="btn_mark_all_p"):
                    for r in records:
                        r["Status"] = "Present"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = True
                    st.session_state["last_attendance_records"] = records
                    st.rerun()
            with tb_col2:
                if st.button("Mark All as Absent", use_container_width=True, key="btn_mark_all_a"):
                    for r in records:
                        r["Status"] = "Absent"
                        st.session_state[f"slide_toggle_{r['Register No']}"] = False
                    st.session_state["last_attendance_records"] = records
                    st.rerun()
            with tb_col3:
                search_q = st.text_input("Search Student:", placeholder="Filter by Name or Reg No...", key="att_edit_search_input")

            st.markdown("<hr style='margin: 10px 0; border-color: #334155;'/>", unsafe_allow_html=True)

            hdr_c1, hdr_c2, hdr_c3, hdr_c4, hdr_c5 = st.columns([0.6, 1.5, 2, 2, 1.5])
            with hdr_c1: st.markdown("**S.No**")
            with hdr_c2: st.markdown("**Register No**")
            with hdr_c3: st.markdown("**Student Name**")
            with hdr_c4: st.markdown("**Department / Branch**")
            with hdr_c5: st.markdown("**Attendance Status (Slide Toggle)**")
            st.markdown("<hr style='margin: 4px 0 12px 0; border-color: #334155;'/>", unsafe_allow_html=True)

            filtered_recs = records
            if search_q.strip():
                st_term = search_q.strip().lower()
                filtered_recs = [r for r in records if (st_term in r["Register No"].lower() or st_term in r["Name"].lower())]

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
            st.markdown("Official PDF report contains **ONLY Present Students** along with Header metadata (Faculty Name, Slot Name, Total Class Strength, Present Count, Absent Count, Date & Time).")

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
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="Download Official Attendance PDF (.pdf)",
                    data=pdf_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_down_pdf_edit"
                )
            with exp_col2:
                st.download_button(
                    label="Download Attendance Excel Sheet (.xlsx)",
                    data=excel_bytes,
                    file_name=f"Attendance_Report_{date_file_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="btn_down_excel_edit"
                )

    # ==========================================================================
    # TAB 3: ALL-TIME ATTENDANCE HISTORY (INTERACTIVE CALENDAR VIEW BY YEAR & MONTH)
    # ==========================================================================
    with tab_att_history:
        st.subheader("All-Time Attendance History & Interactive Calendar View")
        st.markdown("Select Year and Month, then click any highlighted date on the calendar below to view and export attendance records.")

        saved_sessions = database.get_all_attendance_sessions()

        if not saved_sessions:
            st.info("No attendance sessions recorded in database yet.")
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
                sel_year = st.selectbox("Select Year:", db_years, index=0, key="cal_sel_year")
            with col_m:
                default_m_idx = current_mo - 1
                sel_month_name = st.selectbox("Select Month:", month_names, index=default_m_idx, key="cal_sel_month")
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
                                st.markdown(f"<div style='background:#15803d; border:2px solid #22c55e; border-radius:8px; text-align:center; padding:8px 4px; color:#fff; font-weight:700; font-size:0.95rem; margin-bottom:2px;'>{day_num}<br><span style='font-size:0.65rem; opacity:0.9;'>{sess_cnt} rec</span></div>", unsafe_allow_html=True)
                                if st.button("View", key=f"cal_btn_{date_fmt}", use_container_width=True):
                                    st.session_state["selected_cal_date"] = date_fmt
                                    st.rerun()
                            else:
                                st.markdown(f"<div style='background:#7f1d1d; border:2px solid #ef4444; border-radius:8px; text-align:center; padding:8px 4px; color:#fca5a5; font-weight:600; font-size:0.95rem; margin-bottom:2px; opacity:0.7;'>{day_num}</div>", unsafe_allow_html=True)

            st.markdown("---")
            active_date = st.session_state.get("selected_cal_date", None)

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
                                        key=f"cal_pdf_{s_id}"
                                    )
                                with h_col2:
                                    st.download_button(
                                        label=f"Download Excel Sheet ({info['date_str']})",
                                        data=h_excel_bytes,
                                        file_name=f"Attendance_{info['date_str']}_{s_id}.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        use_container_width=True,
                                        key=f"cal_excel_{s_id}"
                                    )
                                with h_col3:
                                    if st.button("Delete Session", key=f"cal_del_{s_id}", use_container_width=True):
                                        database.delete_attendance_session(s_id)
                                        st.session_state["selected_cal_date"] = None
                                        st.success("Session record deleted successfully.")
                                        st.rerun()
            else:
                st.info("Click on any active date in the calendar above to view detailed attendance logs and export PDF/Excel reports.")

    # ==========================================================================
    # TAB 4: MANAGE REGISTERED STUDENTS DIRECTORY
    # ==========================================================================
    with tab_students_mgmt:
        st.subheader("Registered Students Directory & Profile Management")
        all_students = database.get_all_students()
        
        if not all_students:
            st.warning("No students currently registered in database.")
        else:
            sm1, sm2 = st.columns(2)
            sm1.metric("Total Registered Students", len(all_students))
            sm2.metric("Central Database File", "attendance_system.db")
            st.markdown("---")
            
            filtered_students = [{
                "Register No": s["register_no"],
                "Full Name": s["name"],
                "Department / Branch": s["department"]
            } for s in all_students]
            
            st.dataframe(pd.DataFrame(filtered_students), use_container_width=True, hide_index=True)
            
            st.markdown("---")
            st.subheader("Modify or Delete Student Profile")
            student_map = {f"{s['register_no']} - {s['name']} ({s['department']})": s for s in all_students}
            selected_option = st.selectbox("Select a Student to Manage:", list(student_map.keys()), key="mgmt_select_student")
            
            if selected_option:
                target_student = student_map[selected_option]
                target_reg = target_student["register_no"]
                edit_col, delete_col = st.columns([1.2, 1], gap="large")
                
                with edit_col:
                    st.markdown("#### Modify Student Details")
                    with st.form(f"mgmt_edit_form_{target_reg}"):
                        new_name = st.text_input("Full Name", value=target_student["name"])
                        all_dept_options = ["Computer Science & Engineering", "Information Technology", "Electrical & Electronics", "Mechanical Engineering", "Civil Engineering", "Biotechnology", "Other"]
                        curr_dept = target_student["department"]
                        default_idx = all_dept_options.index(curr_dept) if curr_dept in all_dept_options else len(all_dept_options) - 1
                        new_dept = st.selectbox("Department", all_dept_options, index=default_idx)
                        save_btn = st.form_submit_button("Save Updated Details", use_container_width=True)
                        
                        if save_btn:
                            if not new_name.strip():
                                st.error("Full name cannot be empty.")
                            else:
                                success = database.update_student(target_reg, new_name, new_dept)
                                if success:
                                    st.success(f"Details updated for student {target_reg}.")
                                    st.rerun()
                                else:
                                    st.error("Failed to update student details.")

                with delete_col:
                    st.markdown("#### Delete Student Profile")
                    st.warning(f"Are you sure you want to delete {target_student['name']} ({target_reg})?")
                    if st.button(f"Delete Student {target_reg}", use_container_width=True, key=f"mgmt_del_btn_{target_reg}"):
                        success = database.delete_student(target_reg)
                        if success:
                            st.success(f"Student {target_reg} removed from database.")
                            st.rerun()
                        else:
                            st.error("Failed to delete student.")
