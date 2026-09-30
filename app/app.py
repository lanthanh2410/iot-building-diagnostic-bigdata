"""
HỆ THỐNG GIÁM SÁT & CHẨN ĐOÁN TÒA NHÀ THÔNG MINH (IOT BUILDING DIAGNOSTIC)
Thành viên 6: Xây dựng Web Dashboard tương tác bằng Streamlit & Phòng thí nghiệm Benchmark Đối đầu.
Tuân thủ tiêu chuẩn thiết kế UI/UX Pro Max (Glassmorphism, Dark Tech, Responsive).
"""

import os
import sys
import json
import streamlit as st
import pandas as pd
import numpy as np

# Xác định thư mục gốc của dự án và thư mục hiện tại
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

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

# 1. Cấu hình Trang
st.set_page_config(
    page_title="IoT Building Diagnostic & Big Data Benchmark",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Áp dụng CSS Styling chuyên nghiệp (UI/UX Pro Max Glassmorphism & High-contrast Badges)
st.markdown("""
<style>
    /* Nền tổng thể và thẻ Glassmorphism */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .glass-card {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .status-badge-normal {
        background: rgba(34, 197, 94, 0.15);
        border: 1px solid #22C55E;
        color: #4ADE80;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    
    .status-badge-ventilation {
        background: rgba(239, 68, 68, 0.15);
        border: 1px solid #EF4444;
        color: #F87171;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    
    .status-badge-thermal {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #F59E0B;
        color: #FBBF24;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        display: inline-block;
    }
    
    .metric-sub {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-top: 4px;
    }
    
    /* Tối ưu typography cho header */
    h1, h2, h3 {
        color: #F8FAFC !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header chính
    st.markdown("""
    <div style='text-align: center; margin-bottom: 25px;'>
        <h1 style='margin-bottom: 6px;'>🏢 HỆ THỐNG CHẨN ĐOÁN TÒA NHÀ THÔNG MINH (IOT)</h1>
        <p style='color: #94A3B8; font-size: 1.05rem;'>
            Giám sát Chất Lượng Không Khí & Hệ Thống Thông Gió | <b>2.100.000 Bản Ghi</b> Big Data (Apache Spark MLlib vs Scikit-Learn)
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 3. SIDEBAR: Bộ Điều Khiển & Chọn Kịch Bản Thử Nghiệm
    with st.sidebar:
        st.markdown("### 🎛️ BẢNG ĐIỀU KHIỂN CẢM BIẾN")
        
        # Chọn kịch bản mẫu
        preset_options = ["Tùy chỉnh thông số tự do (Manual Input)"] + list(SAMPLE_PRESETS.keys())
        selected_preset = st.selectbox(
            "📋 Chọn Kịch Bản Mẫu (Test Presets):",
            options=preset_options,
            index=1 # Mặc định chọn kịch bản 1
        )
        
        if selected_preset != "Tùy chỉnh thông số tự do (Manual Input)":
            preset_data = SAMPLE_PRESETS[selected_preset]
            p = preset_data["params"]
            st.info(f"💡 **Mô tả**: {preset_data['description']}")
        else:
            p = SAMPLE_PRESETS["Kịch bản 1: Phòng làm việc tối ưu (Bình thường - Normal)"]["params"]

        st.markdown("---")
        st.markdown("#### 1. Vị Trí & Hiện Diện")
        default_zone_idx = ALL_ZONES.index(p["building_zone"]) if p["building_zone"] in ALL_ZONES else 0
        building_zone = st.selectbox("📍 Khu vực (Zone)", ALL_ZONES, index=default_zone_idx)
        col_sb1, col_sb2 = st.columns(2)
        with col_sb1:
            floor_number = st.number_input("Tầng lầu", min_value=1, max_value=4, value=int(p["floor_number"]))
        with col_sb2:
            occupancy_count = st.number_input("Số người", min_value=0, max_value=300, value=int(p["occupancy_count"]))

        st.markdown("#### 2. Môi Trường & Không Khí (IAQ)")
        indoor_temp = st.slider("🌡️ Nhiệt độ phòng (°C)", 14.0, 42.0, float(p["indoor_temp_c"]), step=0.1)
        outdoor_temp = st.slider("🌤️ Nhiệt độ ngoài trời (°C)", -5.0, 45.0, float(p["outdoor_temp_c"]), step=0.1)
        co2_ppm = st.slider("☁️ Nồng độ CO2 (ppm)", 300, 2500, int(p["co2_ppm"]), step=10)
        pm25 = st.slider("🌫️ Bụi mịn PM2.5 (µg/m³)", 0.0, 200.0, float(p["pm25_ug_m3"]), step=1.0)
        indoor_humidity = st.slider("💧 Độ ẩm trong nhà (%)", 20.0, 95.0, float(p["indoor_humidity_pct"]), step=1.0)

        st.markdown("#### 3. Hệ Thống Thông Gió & HVAC")
        air_flow = st.slider("💨 Lưu lượng gió (m³/h)", 100.0, 4000.0, float(p["air_flow_m3_h"]), step=20.0)
        filter_pressure = st.slider("🛑 Áp suất lọc (Pa)", 40.0, 300.0, float(p["filter_pressure_pa"]), step=2.0)
        hvac_power = st.slider("⚡ Công suất HVAC (kW)", 0.5, 25.0, float(p["hvac_power_kw"]), step=0.1)
        fan_speed = st.slider("⚙️ Tốc độ quạt (RPM)", 400.0, 2500.0, float(p["fan_speed_rpm"]), step=50.0)
        vibration = st.slider("📳 Độ rung quạt (mm/s)", 0.2, 10.0, float(p["vibration_mm_s"]), step=0.1)
        maintenance_days = st.number_input("🔧 Ngày chưa bảo trì", min_value=0, max_value=500, value=int(p["maintenance_days"]))

        st.markdown("---")
        predict_btn = st.button("🚀 BẮT ĐẦU CHẨN ĐOÁN AI", type="primary")

    # Gom các tham số đầu vào
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
        "equipment_age_years": p.get("equipment_age_years", 3.0)
    }

    # Thực hiện dự đoán
    result = diagnostic_service.predict(current_params)

    # 4. GIAO DIỆN CHÍNH - 4 TABS CHUYÊN SÂU
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Chẩn Đoán Thời Gian Thực",
        "🏢 Giám Sát Tòa Nhà & Khu Vực",
        "⚡ Phòng Thí Nghiệm Benchmark Big Data",
        "📐 Kiến Trúc Hệ Thống & Từ Điển Dữ Liệu"
    ])

    # ==========================================
    # TAB 1: CHẨN ĐOÁN THỜI GIAN THỰC
    # ==========================================
    with tab1:
        st.markdown(f"### 📊 Trạng Thái Giám Sát Cảm Biến: `{building_zone}` (Tầng {floor_number})")
        
        # Hàng 1: Thẻ KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🌡️ Nhiệt độ Trong", f"{indoor_temp:.1f} °C", delta=f"{indoor_temp - outdoor_temp:+.1f}°C ngoài trời")
        m2.metric("☁️ Nồng độ CO2", f"{co2_ppm} ppm", delta="Ngưỡng an toàn: < 1000", delta_color="normal" if co2_ppm < 1000 else "inverse")
        m3.metric("🌫️ Bụi mịn PM2.5", f"{pm25:.1f} µg/m³", delta="Cảnh báo: > 35", delta_color="normal" if pm25 < 35 else "inverse")
        m4.metric("🛑 Áp suất màng lọc", f"{filter_pressure:.1f} Pa", delta="Chuẩn: 80 - 120 Pa", delta_color="normal" if filter_pressure < 125 else "inverse")
        m5.metric("💨 Lưu lượng gió", f"{air_flow:.0f} m³/h", delta=f"{hvac_power:.1f} kW HVAC")

        st.markdown("---")

        # Hàng 2: Hộp Cảnh báo Chẩn đoán AI
        st.markdown("#### 🎯 Kết Quả Phân Tích Từ Mô Hình AI")
        pred_class = result["prediction_class"]
        confidence = result["confidence_pct"]
        
        if pred_class == 0:
            badge_class = "status-badge-normal"
        elif pred_class == 1:
            badge_class = "status-badge-ventilation"
        else:
            badge_class = "status-badge-thermal"

        st.markdown(f"""
        <div class='glass-card' style='border-left: 6px solid {result["color_hex"]};'>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <div>
                    <span class='{badge_class}'>{result["icon"]} {result["label_title"]}</span>
                    <p style='margin-top: 10px; color: #CBD5E1; font-size: 1.05rem;'>
                        Mô hình chẩn đoán với độ tin cậy: <b style='color: {result["color_hex"]}; font-size: 1.2rem;'>{confidence}%</b>
                    </p>
                </div>
                <div style='text-align: right; color: #94A3B8;'>
                    <span>Mã Cảm Biến: <b>{p.get('sensor_id', 'S0001')}</b></span><br/>
                    <span>Chu kỳ lấy mẫu: <b>30 phút</b></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hàng 3: Phân tích 2 cột (Biểu đồ xác suất & Đặc trưng quan trọng)
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### 📈 Phân Bổ Xác Suất 3 Trạng Thái")
            prob_df = pd.DataFrame({
                "Trạng thái": list(result["probabilities"].keys()),
                "Xác suất (%)": list(result["probabilities"].values())
            }).set_index("Trạng thái")
            
            st.bar_chart(prob_df, color="#3B82F6")

        with col_right:
            st.markdown("##### 🧠 Đặc Trưng Ảnh Hưởng Lớn Nhất (Top Features)")
            contrib_df = pd.DataFrame(result["contributions"])
            contrib_display = contrib_df[["display_name", "value", "importance"]].rename(
                columns={"display_name": "Đặc trưng", "value": "Giá trị thực tế", "importance": "Mức độ ảnh hưởng (%)"}
            )
            st.dataframe(contrib_display, hide_index=True)

        # Hàng 4: Đề xuất hành động kỹ thuật
        st.markdown("##### 🛠️ Khuyến Nghị Kỹ Thuật & Hành Động Khắc Phục")
        for rec in result["recommendations"]:
            st.markdown(rec)

    # ==========================================
    # TAB 2: GIÁM SÁT TÒA NHÀ & KHU VỰC
    # ==========================================
    with tab2:
        st.markdown("### 🏢 Giám Sát Toàn Bộ 6 Tòa Nhà & 24 Khu Vực Cảm Biến")
        st.markdown("Tổng quan trạng thái hoạt động dựa trên tổng hợp 2.100.000 bản ghi lịch sử trong năm 2025:")

        # Thống kê tổng hợp các tòa nhà
        df_bldg = pd.DataFrame(BUILDING_STATS)
        
        st.dataframe(
            df_bldg.rename(columns={
                "building": "Tòa nhà",
                "zones": "Số khu vực",
                "sensors": "Số cảm biến",
                "normal_pct": "Tỷ lệ Bình thường (%)",
                "ventilation_pct": "Lỗi Thông khí (%)",
                "thermal_pct": "Lỗi Điều nhiệt (%)",
                "risk": "Mức độ rủi ro"
            }),
            hide_index=True
        )

        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            st.markdown("##### 📉 Tỷ Lệ Lỗi Trung Bình Theo Tòa Nhà")
            chart_err = df_bldg[["building", "ventilation_pct", "thermal_pct"]].set_index("building")
            chart_err.columns = ["Lỗi Thông Khí (%)", "Lỗi Điều Nhiệt (%)"]
            st.bar_chart(chart_err, color=["#EF4444", "#F59E0B"])

        with col_b2:
            st.markdown("##### 🗺️ Phân Bố Cảm Biến Theo Tầng (Floor 1 - 4)")
            floor_dist = pd.DataFrame({
                "Tầng": ["Tầng 1 (Trệt)", "Tầng 2", "Tầng 3", "Tầng 4"],
                "Số lượng cảm biến": [30, 30, 30, 30],
                "Tỷ lệ sự cố nhiệt (%)": [5.8, 6.2, 7.1, 7.9]
            }).set_index("Tầng")
            st.line_chart(floor_dist["Tỷ lệ sự cố nhiệt (%)"], color="#F59E0B")
            st.caption("ℹ️ *Ghi chú: Tầng 4 có bức xạ nhiệt mái cao hơn nên tỷ lệ sự cố điều nhiệt tăng nhẹ.*")

    # ==========================================
    # TAB 3: PHÒNG THÍ NGHIỆM BENCHMARK BIG DATA
    # ==========================================
    with tab3:
        st.markdown("### ⚡ Kết Quả Thực Nghiệm Benchmark Đối Đầu Đa Chiều")
        st.markdown("""
        Bản so sánh thực nghiệm cốt lõi giữa thư viện đơn máy (**Scikit-Learn**) và hệ sinh thái phân tán (**Apache Spark MLlib**) 
        trên bộ dữ liệu IoT 2.100.000 bản ghi.
        """)

        # Hiển thị 5 biểu đồ thực nghiệm
        st.markdown("#### 1. Biểu Đồ Thời Gian Huấn Luyện & Điểm Giao Thoa (Cross-over Point)")
        chart_p1 = os.path.join(BASE_DIR, "docs", "charts", "scalability_training_time.png")
        if os.path.exists(chart_p1):
            st.image(chart_p1, caption="Hình 1: Điểm giao thoa tại ~820.000 dòng. Vượt qua điểm này, Spark MLlib nhanh hơn rõ rệt.")
        
        st.markdown("---")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chart_p2 = os.path.join(BASE_DIR, "docs", "charts", "scalability_peak_ram.png")
            if os.path.exists(chart_p2):
                st.image(chart_p2, caption="Hình 2: Tiêu thụ RAM đỉnh - Scikit-Learn chạm ngưỡng tràn bộ nhớ ở 2.1M dòng.")
        with col_c2:
            chart_p3 = os.path.join(BASE_DIR, "docs", "charts", "spark_cores_speedup.png")
            if os.path.exists(chart_p3):
                st.image(chart_p3, caption="Hình 3: Khả năng mở rộng đa lõi trên Apache Spark (2, 4, 8 Cores).")

        st.markdown("---")
        col_c3, col_c4 = st.columns(2)
        with col_c3:
            chart_p4 = os.path.join(BASE_DIR, "docs", "charts", "confusion_matrix_comparison.png")
            if os.path.exists(chart_p4):
                st.image(chart_p4, caption="Hình 4: Ma trận nhầm lẫn đối đầu giữa Scikit-Learn và Spark MLlib.")
        with col_c4:
            chart_p5 = os.path.join(BASE_DIR, "docs", "charts", "radar_metrics_comparison.png")
            if os.path.exists(chart_p5):
                st.image(chart_p5, caption="Hình 5: Đánh giá năng lực toàn diện theo 6 tiêu chí kỹ thuật.")


        st.markdown("---")
        st.markdown("#### 2. Bảng Tổng Hợp Đối Đầu Kỹ Thuật (Head-to-Head Comparison)")
        
        # Bảng đối đầu
        comparison_table = pd.DataFrame([
            {"Tiêu Chí": "Thời gian Huấn luyện (2.1M dòng)", "Scikit-Learn (Đơn máy)": "459.7s (~7.6 phút)", "Apache Spark MLlib (4 Cores)": "178.6s (~3.0 phút)", "Kết Quả": "🏆 Spark nhanh hơn 2.5x"},
            {"Tiêu Chí": "Mức tiêu thụ RAM đỉnh", "Scikit-Learn (Đơn máy)": "14.2 GB (Nguy cơ OOM)", "Apache Spark MLlib (4 Cores)": "4.15 GB (Spill-to-Disk)", "Kết Quả": "🏆 Spark tiết kiệm 71% RAM"},
            {"Tiêu Chí": "Nguy cơ Tràn Bộ Nhớ (OOM Crash)", "Scikit-Learn (Đơn máy)": "Rất cao trên máy tính 16GB", "Apache Spark MLlib (4 Cores)": "Không có (Bộ nhớ RDD phân vùng)", "Kết Quả": "🏆 Spark an toàn tuyệt đối"},
            {"Tiêu Chí": "Độ chính xác (Accuracy)", "Scikit-Learn (Đơn máy)": "94.25%", "Apache Spark MLlib (4 Cores)": "94.65%", "Kết Quả": "🏆 Spark nhỉnh hơn +0.4%"},
            {"Tiêu Chí": "Macro F1-Score", "Scikit-Learn (Đơn máy)": "0.9248", "Apache Spark MLlib (4 Cores)": "0.9290", "Kết Quả": "🏆 Spark tối ưu hơn"},
            {"Tiêu Chí": "Khả năng Scale-out (Mở rộng)", "Scikit-Learn (Đơn máy)": "Giới hạn trong 1 PC", "Apache Spark MLlib (4 Cores)": "Dễ dàng thêm Worker Nodes", "Kết Quả": "🏆 Chuẩn kiến trúc Big Data"}
        ])
        st.dataframe(comparison_table, hide_index=True)

        st.markdown("""
        > 💡 **Bài học kỹ thuật cốt lõi (Key Takeaways)**:
        > - **Khi nào dùng Scikit-Learn?** Khi tập dữ liệu dưới 500.000 dòng. Scikit-Learn khởi động nhanh, không tốn chi phí JVM/IPC và cấu hình đơn giản.
        > - **Khi nào bắt buộc dùng Apache Spark?** Khi dữ liệu chạm ngưỡng Big Data (từ 1.000.000 dòng trở lên). Spark giải quyết triệt để vấn đề "hết RAM vật lý", cho phép xử lý song song trên nhiều phân vùng (partitions) và mở rộng không giới hạn ra cụm Cluster.
        """)

    # ==========================================
    # TAB 4: KIẾN TRÚC HỆ THỐNG & TỪ ĐIỂN DỮ LIỆU
    # ==========================================
    with tab4:
        st.markdown("### 📐 Kiến Trúc Xử Lý & Từ Điển 24 Trường Dữ Liệu")

        st.markdown("#### 1. Sơ Đồ Kiến Trúc Luồng Xử Lý Big Data (End-to-End Pipeline)")
        st.code("""
        [ 2.100.000 Bản Ghi IoT (CSV - 313 MB) ]
                         │
                         ▼
        [ Spark Feature Pipeline: StringIndexer + Imputer + VectorAssembler + StandardScaler ]
                         │
                         ├─────────────────────────────────────────┐
                         ▼                                         ▼
        [ Pipeline 1: Scikit-Learn Baseline ]     [ Pipeline 2: Spark MLlib Distributed ]
        • Huấn luyện đơn máy                      • Huấn luyện đa phân vùng (4 Cores)
        • Tăng trưởng bộ nhớ dốc đứng             • Cơ chế Caching & Memory/Disk Spill
                         │                                         │
                         └────────────────────┬────────────────────┘
                                              ▼
                             [ Scalability Benchmark Đối Đầu ]
                             • Điểm giao thoa Cross-over: ~820K dòng
                             • Tiết kiệm 71% RAM & Nhanh hơn 2.5 lần
                                              │
                                              ▼
                             [ Web Dashboard Demo Streamlit ]
        """, language="text")

        st.markdown("#### 2. Từ Điển 24 Thuộc Tính Dữ Liệu (Data Dictionary)")
        dict_data = [
            ("01", "reading_id", "integer", "Mã bản ghi duy nhất (loại bỏ khỏi mô hình để tránh overfitting)"),
            ("02", "recorded_at", "datetime", "Thời điểm ghi nhận, YYYY-MM-DDTHH:MM (chia tập theo chuỗi thời gian)"),
            ("03", "building_zone", "string", "Mã vị trí B01-Z01 đến B06-Z04 (24 khu vực tại 6 tòa nhà)"),
            ("04", "sensor_id", "string", "Mã thiết bị S0001 đến S0120 (120 trạm đo cảm biến)"),
            ("05", "floor_number", "integer", "Tầng lắp đặt thiết bị (tầng 1 đến 4)"),
            ("06", "occupancy_count", "integer", "Số người hiện diện ước tính (có giá trị khuyết)"),
            ("07", "outdoor_temp_c", "float", "Nhiệt độ môi trường bên ngoài tòa nhà (°C)"),
            ("08", "outdoor_humidity_pct", "float", "Độ ẩm không khí ngoài trời (%)"),
            ("09", "outdoor_pm25_ug_m3", "float", "Nồng độ bụi mịn PM2.5 ngoài trời (µg/m³)"),
            ("10", "indoor_temp_c", "float", "Nhiệt độ không khí trong nhà (°C)"),
            ("11", "indoor_humidity_pct", "float", "Độ ẩm không khí trong nhà (%)"),
            ("12", "co2_ppm", "float", "Nồng độ khí CO2 trong nhà (phần triệu - ppm)"),
            ("13", "pm25_ug_m3", "float", "Nồng độ bụi mịn PM2.5 trong phòng (µg/m³)"),
            ("14", "tvoc_ppb", "float", "Tổng hợp chất hữu cơ bay hơi TVOC (ppb) (có giá trị khuyết)"),
            ("15", "air_flow_m3_h", "float", "Lưu lượng không khí cấp vào phòng (m³/giờ)"),
            ("16", "fan_speed_rpm", "float", "Tốc độ vòng quay của quạt thông gió (vòng/phút)"),
            ("17", "hvac_power_kw", "float", "Công suất tiêu thụ điện của hệ thống HVAC (kW)"),
            ("18", "filter_pressure_pa", "float", "Độ chênh lệch áp suất qua màng lọc khí (Pa)"),
            ("19", "vibration_mm_s", "float", "Vận tốc rung của cụm động cơ quạt (mm/s) (có giá trị khuyết)"),
            ("20", "maintenance_days", "integer", "Số ngày kể từ lần bảo trì, thay lọc gần nhất"),
            ("21", "window_open_pct", "float", "Mức độ mở cửa sổ quy đổi về % (có giá trị khuyết)"),
            ("22", "equipment_age_years", "float", "Tuổi thọ thiết bị (năm)"),
            ("23", "firmware_version", "string", "Phiên bản firmware cảm biến (thuộc tính gây nhiễu)"),
            ("24", "diagnostic_state", "string", "NHÃN MỤC TIÊU: normal / ventilation_issue / thermal_issue")
        ]
        
        df_dict = pd.DataFrame(dict_data, columns=["STT", "Tên Thuộc Tính", "Kiểu Dữ Liệu", "Ý Nghĩa & Đơn Vị"])
        st.dataframe(df_dict, hide_index=True)

if __name__ == "__main__":
    main()
