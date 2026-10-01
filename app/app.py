"""
HỆ THỐNG GIÁM SÁT & CHẨN ĐOÁN TÒA NHÀ THÔNG MINH (IOT BUILDING DIAGNOSTIC)
Thành viên 6: Xây dựng Web Dashboard tương tác bằng Streamlit & Phòng thí nghiệm Benchmark Đối đầu.
Phong cách thiết kế: UI/UX Pro Max - 3D Modern Tactile Light Theme (Tông Sáng Hiện Đại & Bố Cục Nổi Khối 3D).
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Xác định thư mục gốc của dự án và thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from model_service import diagnostic_service
    from sample_data import SAMPLE_PRESETS, BUILDING_STATS, ALL_ZONES
except ImportError:
    from app.model_service import diagnostic_service
    from app.sample_data import SAMPLE_PRESETS, BUILDING_STATS, ALL_ZONES

# 1. Cấu hình Trang Streamlit
st.set_page_config(
    page_title="IoT Building Intelligence & Big Data Benchmark",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_html(html_str: str):
    """
    Hàm chuẩn hóa và render HTML an toàn tuyệt đối trên Streamlit.
    Loại bỏ toàn bộ thụt đầu dòng (indentation) và xuống dòng thừa để CommonMark
    không bao giờ có thể hiểu nhầm HTML thành khối code (<pre><code>).
    """
    clean_html = " ".join(
        line.strip() for line in html_str.splitlines() if line.strip()
    )
    st.markdown(clean_html, unsafe_allow_html=True)


# 2. Áp dụng CSS Styling 3D Tông Sáng Cao Cấp (3D Modern Tactile Light Theme)
st.markdown(
    """
<style>
    /* Nhúng phông chữ công nghệ cao cấp Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Nền tổng thể: Tông sáng thanh lịch với hiệu ứng ánh sáng nổi 3D */
    .stApp {
        background-color: #F8FAFC !important;
        background-image: 
            radial-gradient(circle at 10% 5%, rgba(219, 234, 254, 0.5) 0%, transparent 35%),
            radial-gradient(circle at 90% 10%, rgba(238, 242, 255, 0.6) 0%, transparent 40%),
            linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%) !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    /* Tiêu đề & Heading */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #0F172A !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }

    /* Phông chữ Telemetry số liệu */
    .mono-num {
        font-family: 'JetBrains Mono', monospace !important;
        font-feature-settings: 'tnum' on, 'zero' on;
    }

    /* Tùy biến toàn bộ Streamlit Container viền thành thẻ 3D Nổi Khối */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-bottom: 3.5px solid #CBD5E1 !important;
        border-radius: 16px !important;
        box-shadow: 
            0 10px 25px -4px rgba(15, 23, 42, 0.05),
            0 4px 6px -2px rgba(15, 23, 42, 0.02) !important;
        padding: 20px !important;
        margin-bottom: 20px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        transform: translateY(-2px) !important;
        border-bottom-color: #94A3B8 !important;
        box-shadow: 0 16px 30px -4px rgba(15, 23, 42, 0.08) !important;
    }

    /* 3D KPI Bento Metric Pod */
    .kpi-pod-3d {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-bottom: 3.5px solid #CBD5E1;
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 
            0 8px 18px -3px rgba(15, 23, 42, 0.04),
            0 3px 6px -2px rgba(15, 23, 42, 0.02),
            inset 0 1px 0 #FFFFFF;
        transition: all 0.2s ease;
        margin-bottom: 12px;
    }

    .kpi-pod-3d:hover {
        transform: translateY(-3px);
        border-bottom-color: #2563EB;
        box-shadow: 0 14px 26px -3px rgba(37, 99, 235, 0.12);
    }

    .kpi-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748B;
    }

    .kpi-icon-3d {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #BFDBFE;
        border-bottom: 2px solid #93C5FD;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.08);
    }

    .kpi-value-3d {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.2;
    }

    .kpi-footer-3d {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Huy Hiệu Nổi 3D (3D Badges & Chips) */
    .chip-3d {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        letter-spacing: 0.03em;
    }
    .chip-3d-green {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #BBF7D0;
        border-bottom: 2px solid #86EFAC;
    }
    .chip-3d-red {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FECACA;
        border-bottom: 2px solid #FCA5A5;
    }
    .chip-3d-amber {
        background: #FEF3C7;
        color: #B45309;
        border: 1px solid #FDE68A;
        border-bottom: 2px solid #FCD34D;
    }
    .chip-3d-blue {
        background: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #DBEAFE;
        border-bottom: 2px solid #BFDBFE;
    }

    /* Khối Phán Quyết AI 3D (3D Verdict Pod) */
    .verdict-pod-3d {
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 24px;
        transition: all 0.25s ease;
    }

    .verdict-3d-normal {
        background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%);
        border: 1px solid #BBF7D0;
        border-bottom: 5px solid #22C55E;
        box-shadow: 0 14px 28px -4px rgba(34, 197, 94, 0.12), inset 0 1px 0 #FFFFFF;
    }

    .verdict-3d-ventilation {
        background: linear-gradient(135deg, #FFFFFF 0%, #FEF2F2 100%);
        border: 1px solid #FECACA;
        border-bottom: 5px solid #EF4444;
        box-shadow: 0 14px 28px -4px rgba(239, 68, 68, 0.12), inset 0 1px 0 #FFFFFF;
    }

    .verdict-3d-thermal {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFFBEB 100%);
        border: 1px solid #FDE68A;
        border-bottom: 5px solid #F59E0B;
        box-shadow: 0 14px 28px -4px rgba(245, 158, 11, 0.12), inset 0 1px 0 #FFFFFF;
    }

    /* Nút Bấm Khối 3D Phồng Nhấn (Tactile 3D Button) */
    div.stButton > button {
        background: linear-gradient(180deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 800 !important;
        font-size: 0.96rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        border: 1px solid #1D4ED8 !important;
        border-bottom: 4.5px solid #1E40AF !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.28) !important;
        transition: all 0.15s ease !important;
        width: 100% !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(180deg, #60A5FA 0%, #2563EB 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(37, 99, 235, 0.35) !important;
        border-bottom: 5px solid #1E40AF !important;
    }

    div.stButton > button:active {
        transform: translateY(2px) !important;
        border-bottom: 2px solid #1E40AF !important;
        box-shadow: 0 3px 8px rgba(37, 99, 235, 0.2) !important;
    }

    /* Thanh Điều Hướng Tabs 3D Hiện Đại */
    .stTabs [data-baseweb="tab-list"] {
        background: #E2E8F0 !important;
        border-radius: 14px !important;
        padding: 6px !important;
        gap: 6px !important;
        border: 1px solid #CBD5E1 !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 22px !important;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        color: #475569 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.15s ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background: rgba(255, 255, 255, 0.5) !important;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #2563EB !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.08), 0 2px 0 #CBD5E1 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Sidebar Tông Sáng */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
        box-shadow: 4px 0 20px rgba(15, 23, 42, 0.03) !important;
    }

    /* Pulsing Green Live Dot */
    @keyframes live-pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(34, 197, 94, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(34, 197, 94, 0); }
    }

    .live-dot-green {
        width: 10px;
        height: 10px;
        background-color: #22C55E;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        animation: live-pulse 2s infinite;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_probability_chart(probabilities: dict):
    """Vẽ biểu đồ phân bổ xác suất dạng thanh ngang Altair phong cách 3D Light Theme."""
    items = []
    color_map = {
        "Bình thường": "#10B981",
        "Sự cố Thông khí": "#EF4444",
        "Sự cố Điều nhiệt": "#F59E0B",
    }
    for k, v in probabilities.items():
        items.append(
            {"Trạng Thái": k, "Xác Suất (%)": v, "Màu Sắc": color_map.get(k, "#2563EB")}
        )
    df_chart = pd.DataFrame(items)

    chart = (
        alt.Chart(df_chart)
        .mark_bar(cornerRadius=8, height=28)
        .encode(
            x=alt.X(
                "Xác Suất (%):Q",
                title="XÁC SUẤT SUY LUẬN TỪ MÔ HÌNH (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(
                    grid=True,
                    gridColor="rgba(226, 232, 240, 0.8)",
                    labelColor="#64748B",
                    titleColor="#475569",
                    titleFontSize=11,
                    labelFontSize=11,
                ),
            ),
            y=alt.Y(
                "Trạng Thái:N",
                title=None,
                sort=None,
                axis=alt.Axis(
                    labelColor="#0F172A", labelFontSize=13, labelFontWeight="bold"
                ),
            ),
            color=alt.Color("Màu Sắc:N", scale=None),
            tooltip=["Trạng Thái", "Xác Suất (%)"],
        )
        .properties(height=160, background="transparent")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, width="stretch")


def main():
    # ==========================================
    # HEADER CHÍNH: 3D COMMAND BANNER (TÔNG SÁNG)
    # ==========================================
    render_html("""
    <div style='background: #FFFFFF; border: 1px solid #E2E8F0; border-bottom: 4px solid #CBD5E1; border-radius: 18px; padding: 22px 28px; margin-bottom: 24px; box-shadow: 0 10px 25px -4px rgba(15, 23, 42, 0.05);'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 10px;'>
            <div style='display: flex; align-items: center;'>
                <span class='live-dot-green'></span>
                <span class='mono-num' style='font-size: 0.8rem; letter-spacing: 0.08em; color: #15803D; font-weight: 700; text-transform: uppercase;'>
                    HỆ THỐNG TRỰC TUYẾN
                </span>
            </div>
            <div style='display: flex; gap: 8px; flex-wrap: wrap;'>
                <span class='chip-3d chip-3d-blue'>⚡ ĐỘ TRỄ: 1.4ms</span>
                <span class='chip-3d chip-3d-green'>🎯 ĐỘ CHÍNH XÁC: 94.65%</span>
                <span class='chip-3d chip-3d-amber'>🏢 6 TÒA NHÀ / 24 KHU VỰC</span>
            </div>
        </div>
        <div style='display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 15px;'>
            <div>
                <h1 style='margin: 0; font-size: 2.15rem; color: #0F172A;'>
                    🏢 HỆ THỐNG CHẨN ĐOÁN TÒA NHÀ THÔNG MINH (IOT)
                </h1>
                <p style='margin: 6px 0 0 0; color: #64748B; font-size: 1.02rem; font-weight: 500;'>
                    Giám Sát Chất Lượng Không Khí & Hệ Thống Thông Gió | <b>2.100.000 Bản Ghi</b> Big Data (Apache Spark MLlib vs Scikit-Learn)
                </p>
            </div>
            <div>
                <span style='background: #EFF6FF; border: 1px solid #BFDBFE; border-bottom: 2px solid #93C5FD; padding: 8px 16px; border-radius: 10px; font-size: 0.86rem; color: #1D4ED8; font-weight: 700;'>
                    🔬 THỰC NGHIỆM
                </span>
            </div>
        </div>
    </div>
    """)

    # ==========================================
    # SIDEBAR: BẢNG ĐIỀU KHIỂN CẢM BIẾN (3D CONSOLE)
    # ==========================================
    with st.sidebar:
        render_html("""
        <div style='margin-bottom: 18px; padding-bottom: 12px; border-bottom: 2px solid #F1F5F9;'>
            <span class='mono-num' style='font-size: 0.76rem; color: #2563EB; letter-spacing: 0.08em; font-weight: 700;'>
                // IOT SENSOR CONSOLE
            </span>
            <h3 style='margin: 4px 0 0 0; font-size: 1.35rem; color: #0F172A;'>🎛️ BẢNG ĐIỀU KHIỂN</h3>
        </div>
        """)

        # Chọn kịch bản mẫu
        preset_options = ["Tùy chỉnh thông số tự do (Manual Input)"] + list(
            SAMPLE_PRESETS.keys()
        )
        selected_preset = st.selectbox(
            "📋 Kịch Bản Mẫu Thử Nghiệm (Test Presets):",
            options=preset_options,
            index=1,
        )

        if selected_preset != "Tùy chỉnh thông số tự do (Manual Input)":
            preset_data = SAMPLE_PRESETS[selected_preset]
            p = preset_data["params"]
            exp_badge = ""
            if preset_data["expected_label"] == "normal":
                exp_badge = "<span class='chip-3d chip-3d-green'>KỲ VỌNG: NORMAL</span>"
            elif preset_data["expected_label"] == "ventilation_issue":
                exp_badge = (
                    "<span class='chip-3d chip-3d-red'>KỲ VỌNG: VENTILATION</span>"
                )
            else:
                exp_badge = (
                    "<span class='chip-3d chip-3d-amber'>KỲ VỌNG: THERMAL</span>"
                )

            render_html(f"""
            <div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-bottom: 2.5px solid #CBD5E1; border-radius: 12px; padding: 12px 14px; margin-bottom: 15px;'>
                <div style='margin-bottom: 6px;'>{exp_badge}</div>
                <div style='font-size: 0.85rem; color: #475569; line-height: 1.45; font-weight: 500;'>{preset_data['description']}</div>
            </div>
            """)
        else:
            p = SAMPLE_PRESETS[
                "Kịch bản 1: Phòng làm việc tối ưu (Bình thường - Normal)"
            ]["params"]

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 15px 0;'>",
            unsafe_allow_html=True,
        )

        # Nhóm 1: Vị Trí & Hiện Diện
        st.markdown("#### 📍 1. Vị Trí & Mật Độ")
        default_zone_idx = (
            ALL_ZONES.index(p["building_zone"])
            if p["building_zone"] in ALL_ZONES
            else 0
        )
        building_zone = st.selectbox(
            "Mã Khu Vực (Building Zone)", ALL_ZONES, index=default_zone_idx
        )

        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            floor_number = st.number_input(
                "Tầng lầu", min_value=1, max_value=4, value=int(p["floor_number"])
            )
        with col_sb2:
            occupancy_count = st.number_input(
                "Số người hiện diện",
                min_value=0,
                max_value=300,
                value=int(p["occupancy_count"]),
            )

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 15px 0;'>",
            unsafe_allow_html=True,
        )

        # Nhóm 2: Môi Trường & Không Khí (IAQ)
        st.markdown("#### 🌡️ 2. Môi Trường Không Khí (IAQ)")
        indoor_temp = st.slider(
            "Nhiệt độ phòng (°C)", 14.0, 42.0, float(p["indoor_temp_c"]), step=0.1
        )
        outdoor_temp = st.slider(
            "Nhiệt độ ngoài trời (°C)", -5.0, 45.0, float(p["outdoor_temp_c"]), step=0.1
        )
        co2_ppm = st.slider(
            "Nồng độ khí CO2 (ppm)", 300, 2500, int(p["co2_ppm"]), step=10
        )
        pm25 = st.slider(
            "Bụi mịn PM2.5 (µg/m³)", 0.0, 200.0, float(p["pm25_ug_m3"]), step=1.0
        )
        indoor_humidity = st.slider(
            "Độ ẩm trong nhà (%)", 20.0, 95.0, float(p["indoor_humidity_pct"]), step=1.0
        )

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 15px 0;'>",
            unsafe_allow_html=True,
        )

        # Nhóm 3: Hệ Thống Thông Gió & HVAC
        st.markdown("#### ⚙️ 3. Hệ Thống Cơ Khí & HVAC")
        air_flow = st.slider(
            "Lưu lượng cấp gió (m³/h)",
            100.0,
            4000.0,
            float(p["air_flow_m3_h"]),
            step=20.0,
        )
        filter_pressure = st.slider(
            "Áp suất màng lọc (Pa)",
            40.0,
            300.0,
            float(p["filter_pressure_pa"]),
            step=2.0,
        )
        hvac_power = st.slider(
            "Công suất tiêu thụ HVAC (kW)",
            0.5,
            25.0,
            float(p["hvac_power_kw"]),
            step=0.1,
        )
        fan_speed = st.slider(
            "Tốc độ quạt (RPM)", 400.0, 2500.0, float(p["fan_speed_rpm"]), step=50.0
        )
        vibration = st.slider(
            "Độ rung động cơ (mm/s)", 0.2, 10.0, float(p["vibration_mm_s"]), step=0.1
        )
        maintenance_days = st.number_input(
            "Số ngày chưa bảo trì",
            min_value=0,
            max_value=500,
            value=int(p["maintenance_days"]),
        )

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 15px 0;'>",
            unsafe_allow_html=True,
        )

        st.markdown("#### 🧠 4. Động Cơ Phân Tích (AI Engine)")
        engine_choice = st.radio(
            "Chọn mô hình:",
            options=["Scikit-Learn", "Apache Spark MLlib"],
            index=0
        )
        engine = "spark" if "Spark" in engine_choice else "sklearn"

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #E2E8F0; margin: 15px 0;'>",
            unsafe_allow_html=True,
        )
        predict_btn = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN", type="primary")

    # Thu thập toàn bộ tham số đầu vào
    current_params = {
        "building_zone": building_zone,
        "floor_number": floor_number,
        "occupancy_count": occupancy_count,
        "indoor_temp_c": indoor_temp,
        "outdoor_temp_c": outdoor_temp,
        "outdoor_humidity_pct": p.get("outdoor_humidity_pct", 65.0),
        "indoor_humidity_pct": indoor_humidity,
        "outdoor_pm25_ug_m3": p.get("outdoor_pm25_ug_m3", 15.0),
        "co2_ppm": co2_ppm,
        "pm25_ug_m3": pm25,
        "tvoc_ppb": p.get("tvoc_ppb", 120.0),
        "air_flow_m3_h": air_flow,
        "fan_speed_rpm": fan_speed,
        "hvac_power_kw": hvac_power,
        "filter_pressure_pa": filter_pressure,
        "vibration_mm_s": vibration,
        "maintenance_days": maintenance_days,
        "window_open_pct": p.get("window_open_pct", 0.0),
        "equipment_age_years": p.get("equipment_age_years", 3.0),
    }

    # Thực hiện dự đoán với Session State (chỉ cập nhật kết quả khi bấm nút)
    if 'prediction_result' not in st.session_state:
        st.session_state['prediction_result'] = diagnostic_service.predict(current_params, engine=engine)
        
    if predict_btn:
        with st.spinner(f"Đang chạy phân tích bằng {engine_choice}..."):
            st.session_state['prediction_result'] = diagnostic_service.predict(current_params, engine=engine)

    result = st.session_state['prediction_result']

    # ==========================================
    # GIAO DIỆN CHÍNH - 4 TABS 3D HIỆN ĐẠI
    # ==========================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 CHẨN ĐOÁN THỜI GIAN THỰC",
            "🏢 GIÁM SÁT HẠ TẦNG 6 TÒA NHÀ",
            "⚡ PHÒNG THÍ NGHIỆM BENCHMARK BIG DATA",
            "📐 KIẾN TRÚC HỆ THỐNG & TỪ ĐIỂN DỮ LIỆU",
        ]
    )

    # =========================================================================
    # TAB 1: CHẨN ĐOÁN THỜI GIAN THỰC (REAL-TIME 3D AI DIAGNOSTIC)
    # =========================================================================
    with tab1:
        render_html(f"""
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; flex-wrap: wrap;'>
            <div>
                <span class='mono-num' style='color: #2563EB; font-size: 0.8rem; font-weight: 700;'>// LIVE SENSOR READOUT</span>
                <h3 style='margin: 2px 0 0 0; color: #0F172A;'>Trạng Thái Giám Sát Cảm Biến: <span style='color: #2563EB;'>{building_zone}</span> (Tầng {floor_number})</h3>
            </div>
            <div style='color: #64748B; font-size: 0.86rem; font-family: "JetBrains Mono", monospace;'>
                MÃ THIẾT BỊ: <b style='color: #0F172A;'>{p.get('sensor_id', 'S0001')}</b> | CHU KỲ ĐO: <b style='color: #0F172A;'>30 PHÚT</b>
            </div>
        </div>
        """)

        # 1. BENTO KPI GRID 3D: Dùng 5 cột Streamlit độc lập để 100% không bao giờ vỡ layout
        delta_t = indoor_temp - outdoor_temp
        co2_status_chip = (
            "chip-3d-green"
            if co2_ppm < 1000
            else ("chip-3d-amber" if co2_ppm < 1400 else "chip-3d-red")
        )
        co2_label = (
            "AN TOÀN"
            if co2_ppm < 1000
            else ("CẢNH BÁO" if co2_ppm < 1400 else "NGUY CƠ")
        )

        pm25_status_chip = (
            "chip-3d-green"
            if pm25 < 35
            else ("chip-3d-amber" if pm25 < 75 else "chip-3d-red")
        )
        pm25_label = (
            "TỐI ƯU" if pm25 < 35 else ("TRUNG BÌNH" if pm25 < 75 else "Ô NHIỄM")
        )

        fp_status_chip = "chip-3d-green" if filter_pressure < 125 else "chip-3d-red"
        fp_label = "SẠCH" if filter_pressure < 125 else "NGHẸT LỌC"

        col_k1, col_k2, col_k3, col_k4, col_k5 = st.columns(5)
        with col_k1:
            render_html(f"""
            <div class='kpi-pod-3d'>
                <div class='kpi-top'>
                    <span class='kpi-label'>NHIỆT ĐỘ PHÒNG</span>
                    <div class='kpi-icon-3d'>🌡️</div>
                </div>
                <div class='kpi-value-3d'>{indoor_temp:.1f}<span style='font-size: 1rem; color: #64748B;'> °C</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d chip-3d-blue'>ΔT: {delta_t:+.1f}°C</span>
                    <span>vs ngoài trời</span>
                </div>
            </div>
            """)

        with col_k2:
            render_html(f"""
            <div class='kpi-pod-3d'>
                <div class='kpi-top'>
                    <span class='kpi-label'>NỒNG ĐỘ CO2</span>
                    <div class='kpi-icon-3d'>☁️</div>
                </div>
                <div class='kpi-value-3d'>{co2_ppm}<span style='font-size: 1rem; color: #64748B;'> ppm</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d {co2_status_chip}'>{co2_label}</span>
                    <span>chuẩn &lt; 1000</span>
                </div>
            </div>
            """)

        with col_k3:
            render_html(f"""
            <div class='kpi-pod-3d'>
                <div class='kpi-top'>
                    <span class='kpi-label'>BỤI MỊN PM2.5</span>
                    <div class='kpi-icon-3d'>🌫️</div>
                </div>
                <div class='kpi-value-3d'>{pm25:.1f}<span style='font-size: 1rem; color: #64748B;'> µg/m³</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d {pm25_status_chip}'>{pm25_label}</span>
                    <span>ngưỡng 35 µg</span>
                </div>
            </div>
            """)

        with col_k4:
            render_html(f"""
            <div class='kpi-pod-3d'>
                <div class='kpi-top'>
                    <span class='kpi-label'>ÁP SUẤT LỌC KHÍ</span>
                    <div class='kpi-icon-3d'>🛑</div>
                </div>
                <div class='kpi-value-3d'>{filter_pressure:.1f}<span style='font-size: 1rem; color: #64748B;'> Pa</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d {fp_status_chip}'>{fp_label}</span>
                    <span>chuẩn 80-120</span>
                </div>
            </div>
            """)

        with col_k5:
            render_html(f"""
            <div class='kpi-pod-3d'>
                <div class='kpi-top'>
                    <span class='kpi-label'>LƯU LƯỢNG GIÓ</span>
                    <div class='kpi-icon-3d'>💨</div>
                </div>
                <div class='kpi-value-3d'>{air_flow:.0f}<span style='font-size: 1rem; color: #64748B;'> m³/h</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d chip-3d-blue'>{hvac_power:.1f} kW</span>
                    <span>tải HVAC</span>
                </div>
            </div>
            """)

        # 2. KHỐI PHÁN QUYẾT AI 3D (3D VERDICT POD)
        pred_class = result["prediction_class"]
        confidence = result["confidence_pct"]

        if pred_class == 0:
            pod_class = "verdict-3d-normal"
            badge_chip = "<span class='chip-3d chip-3d-green'>● HOẠT ĐỘNG HOÀN TOÀN BÌNH THƯỜNG</span>"
            gauge_color = "#10B981"
        elif pred_class == 1:
            pod_class = "verdict-3d-ventilation"
            badge_chip = "<span class='chip-3d chip-3d-red'>● PHÁT HIỆN SỰ CỐ THÔNG KHÍ (VENTILATION)</span>"
            gauge_color = "#EF4444"
        else:
            pod_class = "verdict-3d-thermal"
            badge_chip = "<span class='chip-3d chip-3d-amber'>● PHÁT HIỆN SỰ CỐ ĐIỀU NHIỆT (THERMAL)</span>"
            gauge_color = "#F59E0B"

        render_html(f"""
        <div class='verdict-pod-3d {pod_class}'>
            <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;'>
                <div>
                    <div style='margin-bottom: 8px;'>{badge_chip}</div>
                    <h2 style='margin: 0; font-size: 1.85rem; color: #0F172A;'>
                        {result["icon"]} {result["label_title"]}
                    </h2>
                    <p style='margin: 8px 0 0 0; color: #475569; font-size: 1.05rem; font-weight: 500;'>
                        Mô hình Trí Tuệ Nhân Tạo đạt mức độ tin cậy suy luận: 
                        <b style='color: {gauge_color}; font-size: 1.3rem; font-family: "JetBrains Mono", monospace;'>{confidence}%</b>
                    </p>
                </div>
                <div style='background: #FFFFFF; padding: 12px 18px; border-radius: 12px; border: 1px solid #E2E8F0; border-bottom: 2px solid #CBD5E1; box-shadow: 0 4px 8px rgba(0,0,0,0.03);'>
                    <div class='mono-num' style='font-size: 0.82rem; color: #64748B;'>SỐ NGÀY CHƯA BẢO TRÌ: <b style='color: #0F172A;'>{maintenance_days} ngày</b></div>
                    <div class='mono-num' style='font-size: 0.82rem; color: #64748B; margin-top: 4px;'>ĐỘ RUNG QUẠT: <b style='color: #0F172A;'>{vibration:.1f} mm/s</b></div>
                    <div class='mono-num' style='font-size: 0.82rem; color: #64748B; margin-top: 4px;'>MẬT ĐỘ NGƯỜI: <b style='color: #0F172A;'>{occupancy_count} người</b></div>
                </div>
            </div>
            
            <div style='margin-top: 18px;'>
                <div style='display: flex; justify-content: space-between; font-size: 0.78rem; font-family: "JetBrains Mono", monospace; color: #64748B; margin-bottom: 6px; font-weight: 600;'>
                    <span>ĐỘ TIN CẬY MÔ HÌNH (CONFIDENCE METRIC)</span>
                    <span style='color: {gauge_color}; font-weight: bold;'>{confidence}%</span>
                </div>
                <div style='background: #E2E8F0; height: 10px; border-radius: 5px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.08);'>
                    <div style='width: {confidence}%; height: 100%; background: {gauge_color}; border-radius: 5px;'></div>
                </div>
            </div>
        </div>
        """)

        # 3. PHÂN TÍCH 2 CỘT 3D: BIỂU ĐỒ XÁC SUẤT & ĐẶC TRƯNG VẬT LÝ IOT
        col_chart, col_contrib = st.columns([1, 1])

        with col_chart:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>📈 Phân Bổ Xác Suất 3 Trạng Thái</h4>
                    <span class='chip-3d chip-3d-blue'>SOFTMAX</span>
                </div>
                <p style='color: #64748B; font-size: 0.88rem; margin-bottom: 16px;'>
                    Tỷ lệ phân phối xác suất dự báo từ tập cây quyết định Random Forest:
                </p>
                """)
                render_probability_chart(result["probabilities"])

        with col_contrib:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>🔬 Đặc Trưng Trọng Yếu (Feature Attribution)</h4>
                    <span class='chip-3d chip-3d-blue'>SHAP / WEIGHTS</span>
                </div>
                <p style='color: #64748B; font-size: 0.88rem; margin-bottom: 14px;'>
                    Các biến cảm biến có tỷ trọng đóng góp cao nhất vào chẩn đoán:
                </p>
                """)
                for feat in result["contributions"]:
                    render_html(f"""
                    <div style='display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.9rem;'>
                        <div style='display: flex; align-items: center; gap: 8px;'>
                            <span style='color: #2563EB;'>🔹</span>
                            <span style='color: #334155; font-weight: 500;'>{feat['display_name']}</span>
                        </div>
                        <div style='text-align: right;'>
                            <span class='mono-num' style='color: #0F172A; font-weight: 700; margin-right: 10px;'>{feat['value']}</span>
                            <span class='chip-3d chip-3d-blue' style='font-size: 0.72rem;'>{feat['importance']:.1f}%</span>
                        </div>
                    </div>
                    """)

        # 4. KHUYẾN NGHỊ KỸ THUẬT 3D
        with st.container(border=True):
            render_html("""
            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 16px;'>
                <div style='width: 38px; height: 38px; background: #EFF6FF; border: 1px solid #BFDBFE; border-bottom: 2px solid #93C5FD; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem;'>
                    🛠️
                </div>
                <div>
                    <h4 style='margin: 0; font-size: 1.18rem; color: #0F172A;'>Khuyến Nghị Kỹ Thuật & Quy Trình Khắc Phục</h4>
                    <span style='color: #64748B; font-size: 0.85rem;'>Được sinh tự động từ luật chuyên gia ASHRAE kết hợp suy luận mô hình AI</span>
                </div>
            </div>
            """)
            for rec in result["recommendations"]:
                render_html(f"""
                <div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB; border-radius: 0 10px 10px 0; padding: 13px 18px; margin-bottom: 10px; color: #334155; font-size: 0.93rem; line-height: 1.5; font-weight: 500;'>
                    {rec}
                </div>
                """)

    # =========================================================================
    # TAB 2: GIÁM SÁT HẠ TẦNG 6 TÒA NHÀ (3D FLEET COMMAND)
    # =========================================================================
    with tab2:
        render_html("""
        <div style='margin-bottom: 22px;'>
            <span class='mono-num' style='color: #2563EB; font-size: 0.8rem; font-weight: 700;'>// FLEET COMMAND DASHBOARD</span>
            <h3 style='margin: 2px 0 0 0; color: #0F172A;'>Giám Sát Toàn Bộ 6 Tòa Nhà & 24 Khu Vực Cảm Biến</h3>
            <p style='color: #64748B; font-size: 0.98rem; margin-top: 4px; font-weight: 500;'>
                Bản đồ tổng hợp sức khỏe vận hành dựa trên <b>2.100.000 bản ghi</b> cảm biến liên tục trong năm 2025:
            </p>
        </div>
        """)

        # 6 Mini 3D Building Fleet Cards qua 6 cột Streamlit
        col_b_cards = st.columns(6)
        for idx, bldg in enumerate(BUILDING_STATS):
            with col_b_cards[idx]:
                risk_chip = (
                    "chip-3d-green"
                    if bldg["risk"] == "Thấp"
                    else (
                        "chip-3d-amber"
                        if bldg["risk"] == "Trung bình"
                        else "chip-3d-red"
                    )
                )
                render_html(f"""
                <div class='kpi-pod-3d' style='padding: 14px; text-align: center;'>
                    <div style='font-weight: 800; color: #0F172A; font-size: 1rem;'>{bldg["building"]}</div>
                    <div class='mono-num' style='font-size: 1.5rem; color: #2563EB; font-weight: 800; margin: 6px 0;'>{bldg["normal_pct"]}%</div>
                    <div style='font-size: 0.72rem; color: #64748B; margin-bottom: 8px; font-weight: 600;'>TỶ LỆ KHÔNG LỖI</div>
                    <span class='chip-3d {risk_chip}'>{bldg["risk"].upper()}</span>
                </div>
                """)

        # Bảng Tổng Hợp Chi Tiết (Sleek 3D Table Container)
        df_bldg = pd.DataFrame(BUILDING_STATS)
        with st.container(border=True):
            render_html("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;'>
                <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>📊 Bảng Chỉ Số Vận Hành Từng Tòa Nhà</h4>
                <span class='chip-3d chip-3d-blue'>120 IOT SENSORS ACTIVE</span>
            </div>
            """)
            st.dataframe(
                df_bldg.rename(
                    columns={
                        "building": "Tòa Nhà",
                        "zones": "Số Khu Vực",
                        "sensors": "Số Cảm Biến",
                        "normal_pct": "Tỷ Lệ Bình Thường (%)",
                        "ventilation_pct": "Lỗi Thông Khí (%)",
                        "thermal_pct": "Lỗi Điều Nhiệt (%)",
                        "risk": "Mức Độ Rủi Ro",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        # 2 Biểu đồ phân tích khu vực & tầng lầu
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            with st.container(border=True):
                render_html("""
                <h4 style='margin-bottom: 12px; font-size: 1.1rem; color: #0F172A;'>📉 Tỷ Lệ Sự Cố Bình Quân Theo Tòa Nhà (%)</h4>
                """)
                chart_err = df_bldg[
                    ["building", "ventilation_pct", "thermal_pct"]
                ].set_index("building")
                chart_err.columns = ["Lỗi Thông Khí (%)", "Lỗi Điều Nhiệt (%)"]
                st.bar_chart(chart_err, color=["#EF4444", "#F59E0B"])

        with col_g2:
            with st.container(border=True):
                render_html("""
                <h4 style='margin-bottom: 12px; font-size: 1.1rem; color: #0F172A;'>🗺️ Rủi Ro Sự Cố Nhiệt Theo Tầng Lầu (Floor 1 - 4)</h4>
                """)
                floor_dist = pd.DataFrame(
                    {
                        "Tầng Lầu": [
                            "Tầng 1 (Trệt)",
                            "Tầng 2",
                            "Tầng 3",
                            "Tầng 4 (Mái)",
                        ],
                        "Rủi ro sự cố nhiệt (%)": [5.8, 6.2, 7.1, 7.9],
                    }
                ).set_index("Tầng Lầu")
                st.line_chart(floor_dist["Rủi ro sự cố nhiệt (%)"], color="#2563EB")
                st.caption(
                    "ℹ️ *Ghi chú kỹ thuật: Tầng 4 chịu bức xạ nhiệt mái mặt trời làm gia tăng 36% tỷ lệ sự cố điều nhiệt so với Tầng 1.*"
                )

    # =========================================================================
    # TAB 3: PHÒNG THÍ NGHIỆM BENCHMARK BIG DATA (3D BENCHMARK LAB)
    # =========================================================================
    with tab3:
        render_html("""
        <div style='margin-bottom: 24px;'>
            <span class='mono-num' style='color: #2563EB; font-size: 0.8rem; font-weight: 700;'>// BENCHMARK LABORATORY</span>
            <h3 style='margin: 2px 0 0 0; color: #0F172A;'>Kết Quả Thực Nghiệm Benchmark Đối Đầu Đa Chiều</h3>
            <p style='color: #64748B; font-size: 0.98rem; margin-top: 4px; font-weight: 500;'>
                Đối đầu trực diện giữa thư viện máy tính đơn (**Scikit-Learn**) và cụm phân tán Big Data (**Apache Spark MLlib**) trên <b>2.100.000 bản ghi</b>:
            </p>
        </div>
        """)

        # 3 EXECUTIVE 3D TROPHY CARDS: Sử dụng 3 cột Streamlit riêng biệt để loại bỏ 100% code artifacts
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            render_html("""
            <div class='kpi-pod-3d' style='border-bottom: 4px solid #2563EB; box-shadow: 0 10px 24px rgba(37, 99, 235, 0.1);'>
                <div class='kpi-top'>
                    <span class='kpi-label'>TỐC ĐỘ HUẤN LUYỆN (2.1M DÒNG)</span>
                    <span class='chip-3d chip-3d-blue'>NHANH HƠN 2.5X</span>
                </div>
                <div class='kpi-value-3d' style='color: #2563EB;'>178.6s <span style='font-size: 1.05rem; color: #64748B; font-weight: 500;'>vs 459.7s</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d chip-3d-green'>🏆 SPARK THẮNG</span>
                    <span>Tiết kiệm 4.7 phút đào tạo</span>
                </div>
            </div>
            """)

        with col_t2:
            render_html("""
            <div class='kpi-pod-3d' style='border-bottom: 4px solid #10B981; box-shadow: 0 10px 24px rgba(16, 185, 129, 0.1);'>
                <div class='kpi-top'>
                    <span class='kpi-label'>TIÊU THỤ RAM ĐỈNH (PEAK RAM)</span>
                    <span class='chip-3d chip-3d-green'>TIẾT KIỆM 71% RAM</span>
                </div>
                <div class='kpi-value-3d' style='color: #10B981;'>4.15 GB <span style='font-size: 1.05rem; color: #64748B; font-weight: 500;'>vs 14.2 GB</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d chip-3d-green'>🛡️ CHỐNG SẬP OOM</span>
                    <span>An toàn trên máy cá nhân</span>
                </div>
            </div>
            """)

        with col_t3:
            render_html("""
            <div class='kpi-pod-3d' style='border-bottom: 4px solid #6366F1; box-shadow: 0 10px 24px rgba(99, 102, 241, 0.1);'>
                <div class='kpi-top'>
                    <span class='kpi-label'>ĐỘ CHÍNH XÁC (ACCURACY & F1)</span>
                    <span class='chip-3d chip-3d-blue'>TỐI ƯU TOÀN DIỆN</span>
                </div>
                <div class='kpi-value-3d' style='color: #6366F1;'>94.65% <span style='font-size: 1.05rem; color: #64748B; font-weight: 500;'>vs 94.25%</span></div>
                <div class='kpi-footer-3d'>
                    <span class='chip-3d chip-3d-blue'>+0.4% CHÍNH XÁC</span>
                    <span>Macro F1 đạt 0.9290</span>
                </div>
            </div>
            """)

        # KHUNG HIỂN THỊ 5 BIỂU ĐỒ 300 DPI
        with st.container(border=True):
            render_html("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='font-size: 1.25rem;'>📊</span>
                    <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>1. Biểu Đồ Thời Gian Huấn Luyện & Điểm Giao Thoa (Cross-over Point)</h4>
                </div>
                <span class='chip-3d chip-3d-blue'>HÌNH 1 // 300 DPI</span>
            </div>
            """)
            chart_p1 = os.path.join(
                BASE_DIR, "docs", "charts", "scalability_training_time.png"
            )
            if os.path.exists(chart_p1):
                st.image(
                    chart_p1,
                    caption="Hình 1: Điểm giao thoa hiệu năng tại ~820.000 dòng. Vượt qua mốc này, Apache Spark thể hiện sự vượt trội áp đảo.",
                )

        # Cặp Biểu đồ 2 & 3
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.05rem; color: #0F172A;'>2. Tiêu Thụ RAM Đỉnh & Giới Hạn OOM</h4>
                    <span class='chip-3d chip-3d-red'>NGUY CƠ TRÀN RAM</span>
                </div>
                """)
                chart_p2 = os.path.join(
                    BASE_DIR, "docs", "charts", "scalability_peak_ram.png"
                )
                if os.path.exists(chart_p2):
                    st.image(
                        chart_p2,
                        caption="Hình 2: Tiêu thụ RAM đỉnh - Scikit-Learn chạm ngưỡng 14.2 GB suýt làm sập hệ điều hành 16GB RAM.",
                    )

        with col_c2:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.05rem; color: #0F172A;'>3. Khả Năng Tăng Tốc Đa Lõi (Speedup)</h4>
                    <span class='chip-3d chip-3d-green'>SPARK MULTI-CORE</span>
                </div>
                """)
                chart_p3 = os.path.join(
                    BASE_DIR, "docs", "charts", "spark_cores_speedup.png"
                )
                if os.path.exists(chart_p3):
                    st.image(
                        chart_p3,
                        caption="Hình 3: Khả năng mở rộng tuyến tính của Apache Spark MLlib qua 2, 4, và 8 Cores.",
                    )

        # Cặp Biểu đồ 4 & 5
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.05rem; color: #0F172A;'>4. Ma Trận Nhầm Lẫn Đối Đầu (Confusion Matrix)</h4>
                    <span class='chip-3d chip-3d-blue'>HEATMAP</span>
                </div>
                """)
                chart_p4 = os.path.join(
                    BASE_DIR, "docs", "charts", "confusion_matrix_comparison.png"
                )
                if os.path.exists(chart_p4):
                    st.image(
                        chart_p4,
                        caption="Hình 4: Ma trận nhầm lẫn đối đầu - Spark MLlib nhận diện chính xác 95.1% sự cố thông khí.",
                    )

        with col_c4:
            with st.container(border=True):
                render_html("""
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                    <h4 style='margin: 0; font-size: 1.05rem; color: #0F172A;'>5. Đánh Giá Toàn Diện 6 Chiều Kỹ Thuật</h4>
                    <span class='chip-3d chip-3d-blue'>RADAR CHART</span>
                </div>
                """)
                chart_p5 = os.path.join(
                    BASE_DIR, "docs", "charts", "radar_metrics_comparison.png"
                )
                if os.path.exists(chart_p5):
                    st.image(
                        chart_p5,
                        caption="Hình 5: Biểu đồ Radar 6 chiều chứng minh Spark áp đảo về Quản trị RAM và Khả năng Scale-out.",
                    )

        # BẢNG TỔNG HỢP SO SÁNH ĐỐI ĐẦU CHÍNH THỨC
        with st.container(border=True):
            render_html("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;'>
                <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>📋 Bảng Tổng Hợp Đối Đầu Kỹ Thuật Toàn Diện</h4>
                <span class='chip-3d chip-3d-green'>DỮ LIỆU ĐÃ XÁC THỰC</span>
            </div>
            """)

            comparison_table = pd.DataFrame(
                [
                    {
                        "Tiêu Chí": "Thời gian Huấn luyện (2.1M dòng)",
                        "Scikit-Learn (Đơn máy)": "459.7s (~7.6 phút)",
                        "Apache Spark MLlib (4 Cores)": "178.6s (~3.0 phút)",
                        "Kết Quả": "🏆 Spark nhanh hơn 2.5x",
                    },
                    {
                        "Tiêu Chí": "Mức tiêu thụ RAM đỉnh",
                        "Scikit-Learn (Đơn máy)": "14.2 GB (Nguy cơ OOM)",
                        "Apache Spark MLlib (4 Cores)": "4.15 GB (Spill-to-Disk)",
                        "Kết Quả": "🏆 Spark tiết kiệm 71% RAM",
                    },
                    {
                        "Tiêu Chí": "Nguy cơ Tràn Bộ Nhớ (OOM Crash)",
                        "Scikit-Learn (Đơn máy)": "Rất cao trên máy tính 16GB",
                        "Apache Spark MLlib (4 Cores)": "Không có (Bộ nhớ RDD phân vùng)",
                        "Kết Quả": "🏆 Spark an toàn tuyệt đối",
                    },
                    {
                        "Tiêu Chí": "Độ chính xác (Accuracy)",
                        "Scikit-Learn (Đơn máy)": "94.25%",
                        "Apache Spark MLlib (4 Cores)": "94.65%",
                        "Kết Quả": "🏆 Spark nhỉnh hơn +0.4%",
                    },
                    {
                        "Tiêu Chí": "Macro F1-Score",
                        "Scikit-Learn (Đơn máy)": "0.9248",
                        "Apache Spark MLlib (4 Cores)": "0.9290",
                        "Kết Quả": "🏆 Spark tối ưu hơn",
                    },
                    {
                        "Tiêu Chí": "Khả năng Scale-out (Mở rộng cụm)",
                        "Scikit-Learn (Đơn máy)": "Giới hạn trong 1 PC",
                        "Apache Spark MLlib (4 Cores)": "Dễ dàng thêm Worker Nodes",
                        "Kết Quả": "🏆 Chuẩn kiến trúc Big Data",
                    },
                ]
            )
            st.dataframe(comparison_table, width="stretch", hide_index=True)

            render_html("""
            <div style='background: #EFF6FF; border: 1px solid #BFDBFE; border-left: 4.5px solid #2563EB; border-radius: 0 12px 12px 0; padding: 16px 20px; margin-top: 16px;'>
                <div style='font-weight: 800; color: #1D4ED8; margin-bottom: 6px; font-size: 1rem;'>
                    💡 BÀI HỌC KỸ THUẬT CỐT LÕI (KEY ARCHITECTURAL TAKEAWAYS):
                </div>
                <ul style='margin: 0; padding-left: 20px; color: #334155; font-size: 0.94rem; line-height: 1.6; font-weight: 500;'>
                    <li><b>Dưới 820.000 bản ghi</b>: Scikit-Learn chiếm ưu thế tốc độ nhờ tối ưu hóa C-binding trực tiếp và không chịu chi phí khởi tạo JVM ban đầu.</li>
                    <li><b>Từ 1.000.000 đến 2.100.000 bản ghi</b>: Apache Spark MLlib là giải pháp sống còn để ngăn ngừa lỗi tràn RAM (OOM Crash), tự động tràn bộ nhớ sang ổ đĩa (Spill-to-disk) và tăng tốc độ xử lý gấp 2.5 lần.</li>
                </ul>
            </div>
            """)

    # =========================================================================
    # TAB 4: KIẾN TRÚC HỆ THỐNG & TỪ ĐIỂN DỮ LIỆU (3D ARCHITECTURE)
    # =========================================================================
    with tab4:
        render_html("""
        <div style='margin-bottom: 24px;'>
            <span class='mono-num' style='color: #2563EB; font-size: 0.8rem; font-weight: 700;'>// PIPELINE & SCHEMAS</span>
            <h3 style='margin: 2px 0 0 0; color: #0F172A;'>Kiến Trúc Luồng Xử Lý Big Data & Từ Điển Dữ Liệu</h3>
            <p style='color: #64748B; font-size: 0.98rem; margin-top: 4px; font-weight: 500;'>
                Thiết kế kiến trúc End-to-End từ nguồn 2.1 triệu dòng dữ liệu đến dịch vụ suy luận chẩn đoán thông minh:
            </p>
        </div>
        """)

        # Sơ đồ Pipeline 3D
        with st.container(border=True):
            render_html("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;'>
                <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>📐 Sơ Đồ Khối Luồng Xử Lý End-to-End (Data Architecture)</h4>
                <span class='chip-3d chip-3d-blue'>DISTRIBUTED PIPELINE</span>
            </div>
            """)
            st.code(
                """
[ NGUỒN CẢM BIẾN IOT: 2.100.000 Bản Ghi (11_iot_building_diagnostic.csv - 313 MB) ]
                                │
                                ▼
[ TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG SPARK: StringIndexer + Imputer + VectorAssembler ]
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[ PIPELINE 1: SCIKIT-LEARN BASELINE ]         [ PIPELINE 2: APACHE SPARK MLLIB ]
• Huấn luyện tiến trình đơn máy               • Huấn luyện phân tán đa lõi (4-8 Cores)
• RAM tăng dốc đứng (14.2 GB)                 • RDD Partitioning & Spill-to-Disk (4.15 GB)
• Thời gian: 459.7 giây                       • Thời gian: 178.6 giây (Nhanh hơn 2.5x)
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
              [ PHÒNG THÍ NGHIỆM BENCHMARK ĐỐI ĐẦU ]
              • Điểm giao thoa hiệu năng: ~820K dòng
              • Bộ 5 biểu đồ chất lượng xuất bản (300 DPI)
                                │
                                ▼
              [ STREAMLIT WEB APP CHẨN ĐOÁN THỜI GIAN THỰC ]
            """,
                language="text",
            )

        # Từ Điển 24 Thuộc Tính Dữ Liệu
        with st.container(border=True):
            render_html("""
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;'>
                <h4 style='margin: 0; font-size: 1.15rem; color: #0F172A;'>📖 Từ Điển 24 Thuộc Tính Dữ Liệu (IoT Data Dictionary)</h4>
                <span class='chip-3d chip-3d-green'>24 FEATURES / 1 TARGET</span>
            </div>
            """)

            dict_data = [
                (
                    "01",
                    "reading_id",
                    "integer",
                    "Mã bản ghi cảm biến duy nhất (loại bỏ khỏi mô hình để tránh overfitting)",
                    "Định Danh",
                ),
                (
                    "02",
                    "recorded_at",
                    "datetime",
                    "Thời điểm ghi nhận tín hiệu cảm biến YYYY-MM-DDTHH:MM (chia tập time-series)",
                    "Thời Gian",
                ),
                (
                    "03",
                    "building_zone",
                    "string",
                    "Mã vị trí B01-Z01 đến B06-Z04 (24 khu vực tại 6 tòa nhà)",
                    "Vị Trí",
                ),
                (
                    "04",
                    "sensor_id",
                    "string",
                    "Mã thiết bị S0001 đến S0120 (120 trạm đo cảm biến vật lý)",
                    "Thiết Bị",
                ),
                (
                    "05",
                    "floor_number",
                    "integer",
                    "Tầng lắp đặt trạm đo (Tầng 1 đến 4)",
                    "Vị Trí",
                ),
                (
                    "06",
                    "occupancy_count",
                    "integer",
                    "Số lượng người hiện diện ước tính trong phòng (có giá trị khuyết)",
                    "Hiện Diện",
                ),
                (
                    "07",
                    "outdoor_temp_c",
                    "float",
                    "Nhiệt độ môi trường bên ngoài tòa nhà (°C)",
                    "Môi Trường",
                ),
                (
                    "08",
                    "outdoor_humidity_pct",
                    "float",
                    "Độ ẩm không khí môi trường bên ngoài (%)",
                    "Môi Trường",
                ),
                (
                    "09",
                    "outdoor_pm25_ug_m3",
                    "float",
                    "Nồng độ bụi mịn PM2.5 môi trường ngoài trời (µg/m³)",
                    "Môi Trường",
                ),
                (
                    "10",
                    "indoor_temp_c",
                    "float",
                    "Nhiệt độ không khí bên trong phòng giám sát (°C)",
                    "IAQ",
                ),
                (
                    "11",
                    "indoor_humidity_pct",
                    "float",
                    "Độ ẩm không khí bên trong phòng (%)",
                    "IAQ",
                ),
                (
                    "12",
                    "co2_ppm",
                    "float",
                    "Nồng độ khí CO2 trong phòng (phần triệu - ppm, chuẩn < 1000)",
                    "IAQ",
                ),
                (
                    "13",
                    "pm25_ug_m3",
                    "float",
                    "Nồng độ bụi mịn PM2.5 trong phòng (µg/m³, chuẩn < 35)",
                    "IAQ",
                ),
                (
                    "14",
                    "tvoc_ppb",
                    "float",
                    "Tổng hợp chất hữu cơ bay hơi TVOC (ppb, có giá trị khuyết)",
                    "IAQ",
                ),
                (
                    "15",
                    "air_flow_m3_h",
                    "float",
                    "Lưu lượng không khí cấp vào phòng qua miệng gió (m³/giờ)",
                    "HVAC",
                ),
                (
                    "16",
                    "fan_speed_rpm",
                    "float",
                    "Tốc độ vòng quay của quạt thông gió (vòng/phút)",
                    "HVAC",
                ),
                (
                    "17",
                    "hvac_power_kw",
                    "float",
                    "Công suất tiêu thụ điện năng tức thời của hệ thống HVAC (kW)",
                    "HVAC",
                ),
                (
                    "18",
                    "filter_pressure_pa",
                    "float",
                    "Độ chênh áp suất trước và sau màng lọc khí (Pa, chuẩn 80-120)",
                    "Cơ Khí",
                ),
                (
                    "19",
                    "vibration_mm_s",
                    "float",
                    "Vận tốc rung của cụm động cơ quạt hút (mm/s, có giá trị khuyết)",
                    "Cơ Khí",
                ),
                (
                    "20",
                    "maintenance_days",
                    "integer",
                    "Số ngày trôi qua kể từ lần bảo dưỡng, thay lọc gần nhất",
                    "Vận Hành",
                ),
                (
                    "21",
                    "window_open_pct",
                    "float",
                    "Mức độ mở cửa sổ lấy gió tự nhiên quy đổi theo % (có giá trị khuyết)",
                    "Vận Hành",
                ),
                (
                    "22",
                    "equipment_age_years",
                    "float",
                    "Tuổi thọ khấu hao của thiết bị cảm biến & quạt (năm)",
                    "Thiết Bị",
                ),
                (
                    "23",
                    "firmware_version",
                    "string",
                    "Phiên bản vi điều khiển cảm biến (thuộc tính metadata gây nhiễu)",
                    "Metadata",
                ),
                (
                    "24",
                    "diagnostic_state",
                    "string",
                    "NHÃN MỤC TIÊU (TARGET): normal / ventilation_issue / thermal_issue",
                    "Nhãn AI",
                ),
            ]

            df_dict = pd.DataFrame(
                dict_data,
                columns=[
                    "STT",
                    "Tên Thuộc Tính",
                    "Kiểu Dữ Liệu",
                    "Ý Nghĩa Kỹ Thuật",
                    "Nhóm Đặc Trưng",
                ],
            )
            st.dataframe(df_dict, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
