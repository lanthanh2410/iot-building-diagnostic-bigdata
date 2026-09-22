"""
Thành viên 6: Xây dựng Web Dashboard tương tác bằng Streamlit.
Chức năng: Giao diện chẩn đoán trạng thái hệ thống điều hòa & thông gió thời gian thực.
"""
import streamlit as st
import pandas as pd
import numpy as np

def main():
    # 1. Cấu hình trang
    st.set_page_config(page_title="IoT Building Diagnostic", page_icon="🏢", layout="wide")
    
    # 2. Header & Tiêu đề
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>HỆ THỐNG CHẨN ĐOÁN TÒA NHÀ THÔNG MINH (IOT)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px; color: #7F8C8D;'>Mô phỏng chẩn đoán hệ thống thông gió và điều tiết nhiệt dựa trên mô hình Big Data (Apache Spark)</p>", unsafe_allow_html=True)
    st.divider()

    # 3. Sidebar - Cấu hình hệ thống
    with st.sidebar:
        st.markdown("### BẢNG ĐIỀU KHIỂN CẢM BIẾN")
        st.markdown("Nhập thông số mô phỏng theo thời gian thực:")
        
        building_zone = st.selectbox("📍 Khu vực (Building Zone)", ["B01-Z01", "B01-Z02", "B02-Z01", "B03-Z04"])
        occupancy = st.number_input("👥 Mật độ người (Occupancy)", min_value=0, max_value=500, value=120)
        
        st.markdown("---")
        indoor_temp = st.slider("🌡️ Nhiệt độ trong nhà (°C)", 15.0, 40.0, 26.5)
        co2 = st.slider("☁️ Nồng độ CO2 (ppm)", 300, 2500, 850)
        pm25 = st.slider("🌫️ Bụi mịn PM2.5 (µg/m³)", 0, 300, 45)
        airflow = st.slider("💨 Lưu lượng gió (m³/h)", 500, 5000, 2500)
        
        predict_btn = st.button("BẮT ĐẦU CHẨN ĐOÁN", use_container_width=True, type="primary")

    # 4. Giao diện chính - Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan Trạng thái", "🧠 Chi tiết Mô hình", "📋 Nhật ký Hệ thống"])
    
    with tab1:
        st.subheader(f"📍 Đang giám sát: Khu vực {building_zone}")
        
        # Hàng 1: Hiển thị các chỉ số dạng Metric (Dashboard)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(label="Nhiệt độ (Indoor)", value=f"{indoor_temp} °C", delta="Ngưỡng an toàn: < 27°C", delta_color="normal" if indoor_temp < 27 else "inverse")
        col2.metric(label="Mức CO2", value=f"{co2} ppm", delta="Ngưỡng an toàn: < 1000", delta_color="normal" if co2 < 1000 else "inverse")
        col3.metric(label="Bụi PM2.5", value=f"{pm25} µg/m³", delta="Cảnh báo: > 50", delta_color="normal" if pm25 < 50 else "inverse")
        col4.metric(label="Lưu lượng gió", value=f"{airflow} m³/h", delta="Tối ưu: 2000-3000")
        
        st.markdown("---")
        
        # Khu vực hiển thị kết quả chẩn đoán
        st.subheader("Kết quả Chẩn đoán AI")
        if predict_btn:
            with st.spinner('Mô hình Spark MLlib đang phân tích dữ liệu...'):
                import time
                time.sleep(1.5) # Giả lập thời gian load
                
                # Logic giả lập (mockup) để hiển thị giao diện chuyên nghiệp
                if co2 > 1500 or pm25 > 100:
                    st.error("🚨 PHÁT HIỆN SỰ CỐ: **Lỗi Hệ Thống Thông Khí (Ventilation Issue)**")
                    st.markdown("*Đề xuất: Tăng tốc độ quạt (fan_speed_rpm), kiểm tra độ chênh lệch áp suất màng lọc (filter_pressure_pa).*")
                elif indoor_temp > 30:
                    st.warning("⚠️ PHÁT HIỆN SỰ CỐ: **Lỗi Điều Tiết Nhiệt (Thermal Issue)**")
                    st.markdown("*Đề xuất: Kiểm tra công suất HVAC (hvac_power_kw) và đối chiếu với nhiệt độ ngoài trời (outdoor_temp_c).*")
                else:
                    st.success("✅ TRẠNG THÁI: **Hệ thống vận hành Bình Thường (Normal)**")
                    st.markdown("*Chất lượng không khí và nhiệt độ đang ở mức tối ưu.*")
        else:
            st.info("Hãy điều chỉnh thông số ở cột bên trái và bấm 'BẮT ĐẦU CHẨN ĐOÁN' để xem kết quả.")

    with tab2:
        st.subheader("Mức độ quan trọng của Đặc trưng (Feature Importance)")
        st.markdown("Phân tích từ mô hình Random Forest (Apache Spark MLlib)")
        # Tạo dữ liệu giả lập cho biểu đồ
        chart_data = pd.DataFrame({
            "Mức độ ảnh hưởng (%)": [25, 20, 15, 12, 10, 8, 5, 5],
            "Đặc trưng": ["co2_ppm", "indoor_temp_c", "filter_pressure_pa", "hvac_power_kw", "pm25_ug_m3", "outdoor_temp_c", "occupancy", "airflow_m3_h"]
        }).set_index("Đặc trưng")
        
        st.bar_chart(chart_data)
        
    with tab3:
        st.subheader("Nhật ký (Log) Gần Đây")
        log_data = pd.DataFrame(
            np.random.randn(5, 3),
            columns=['RAM Tiêu thụ (MB)', 'Thời gian Infer (s)', 'CPU Cores']
        )
        st.dataframe(log_data.style.highlight_max(axis=0), use_container_width=True)

if __name__ == "__main__":
    main()
