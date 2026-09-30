"""
Bộ dữ liệu mẫu thực tế và thống kê tổng hợp phục vụ Web Dashboard Streamlit.
Trích xuất từ bộ dữ liệu gốc 2.100.000 bản ghi (11_iot_building_diagnostic.csv).
"""

# Danh sách kịch bản mẫu điển hình trích xuất từ tập kiểm thử
SAMPLE_PRESETS = {
    "Kịch bản 1: Phòng làm việc tối ưu (Bình thường - Normal)": {
        "description": "Văn phòng B01-Z01, tầng 1, mật độ vừa phải. Nhiệt độ 22.1°C, CO2 510 ppm, màng lọc sạch 92 Pa. Hệ thống HVAC vận hành trơn tru.",
        "expected_label": "normal",
        "params": {
            "building_zone": "B01-Z01",
            "sensor_id": "S0001",
            "floor_number": 1,
            "occupancy_count": 4,
            "outdoor_temp_c": 10.9,
            "outdoor_humidity_pct": 65.0,
            "outdoor_pm25_ug_m3": 12.0,
            "indoor_temp_c": 22.14,
            "indoor_humidity_pct": 52.0,
            "co2_ppm": 510.0,
            "pm25_ug_m3": 9.69,
            "tvoc_ppb": 110.0,
            "air_flow_m3_h": 478.6,
            "fan_speed_rpm": 1200.0,
            "hvac_power_kw": 4.0,
            "filter_pressure_pa": 92.12,
            "vibration_mm_s": 1.2,
            "maintenance_days": 25,
            "window_open_pct": 10.0,
            "equipment_age_years": 2.5
        }
    },
    "Kịch bản 2: Nghẹt màng lọc & CO2 tăng cao (Lỗi Thông Khí - Ventilation Issue)": {
        "description": "Khu vực B01-Z02, tầng 2, phòng họp đông người. Áp suất lọc tăng vọt lên 145 Pa, lưu lượng gió giảm còn 234 m³/h, bụi PM2.5 tăng gấp 4 lần.",
        "expected_label": "ventilation_issue",
        "params": {
            "building_zone": "B01-Z02",
            "sensor_id": "S0007",
            "floor_number": 2,
            "occupancy_count": 18,
            "outdoor_temp_c": 11.87,
            "outdoor_humidity_pct": 70.0,
            "outdoor_pm25_ug_m3": 15.0,
            "indoor_temp_c": 20.44,
            "indoor_humidity_pct": 58.0,
            "co2_ppm": 1250.0,
            "pm25_ug_m3": 42.31,
            "tvoc_ppb": 340.0,
            "air_flow_m3_h": 234.34,
            "fan_speed_rpm": 950.0,
            "hvac_power_kw": 3.4,
            "filter_pressure_pa": 144.94,
            "vibration_mm_s": 2.8,
            "maintenance_days": 180,
            "window_open_pct": 0.0,
            "equipment_age_years": 5.0
        }
    },
    "Kịch bản 3: Quá tải HVAC & Bất đối xứng nhiệt (Lỗi Điều Nhiệt - Thermal Issue)": {
        "description": "Khu vực B01-Z01, tầng 1. Công suất điều hòa đẩy lên 5.67 kW liên tục nhưng chênh lệch nhiệt độ trong/ngoài bất thường, quạt rung mạnh.",
        "expected_label": "thermal_issue",
        "params": {
            "building_zone": "B01-Z01",
            "sensor_id": "S0003",
            "floor_number": 1,
            "occupancy_count": 4,
            "outdoor_temp_c": 10.41,
            "outdoor_humidity_pct": 60.0,
            "outdoor_pm25_ug_m3": 8.0,
            "indoor_temp_c": 26.8,
            "indoor_humidity_pct": 38.0,
            "co2_ppm": 568.0,
            "pm25_ug_m3": 2.72,
            "tvoc_ppb": 95.0,
            "air_flow_m3_h": 536.57,
            "fan_speed_rpm": 1450.0,
            "hvac_power_kw": 5.67,
            "filter_pressure_pa": 124.64,
            "vibration_mm_s": 3.4,
            "maintenance_days": 90,
            "window_open_pct": 20.0,
            "equipment_age_years": 4.2
        }
    },
    "Kịch bản 4: Giờ cao điểm hội trường lớn (Bình thường có tải - Normal High Load)": {
        "description": "Hội trường B03-Z01, tầng 3, 120 người tham dự. Hệ thống tự động bù lưu lượng gió 2.800 m³/h, CO2 duy trì an toàn 820 ppm.",
        "expected_label": "normal",
        "params": {
            "building_zone": "B03-Z01",
            "sensor_id": "S0045",
            "floor_number": 3,
            "occupancy_count": 120,
            "outdoor_temp_c": 28.5,
            "outdoor_humidity_pct": 75.0,
            "outdoor_pm25_ug_m3": 25.0,
            "indoor_temp_c": 23.5,
            "indoor_humidity_pct": 50.0,
            "co2_ppm": 820.0,
            "pm25_ug_m3": 18.0,
            "tvoc_ppb": 160.0,
            "air_flow_m3_h": 2850.0,
            "fan_speed_rpm": 1600.0,
            "hvac_power_kw": 12.5,
            "filter_pressure_pa": 110.0,
            "vibration_mm_s": 1.5,
            "maintenance_days": 15,
            "window_open_pct": 0.0,
            "equipment_age_years": 1.8
        }
    }
}

# Thống kê tổng quan 24 khu vực tại 6 tòa nhà (B01 đến B06)
BUILDING_STATS = [
    {"building": "Tòa nhà B01", "zones": 4, "sensors": 20, "normal_pct": 76.5, "ventilation_pct": 16.8, "thermal_pct": 6.7, "risk": "Thấp"},
    {"building": "Tòa nhà B02", "zones": 4, "sensors": 20, "normal_pct": 73.2, "ventilation_pct": 19.5, "thermal_pct": 7.3, "risk": "Trung bình"},
    {"building": "Tòa nhà B03", "zones": 4, "sensors": 20, "normal_pct": 78.4, "ventilation_pct": 15.2, "thermal_pct": 6.4, "risk": "Thấp"},
    {"building": "Tòa nhà B04", "zones": 4, "sensors": 20, "normal_pct": 71.8, "ventilation_pct": 21.0, "thermal_pct": 7.2, "risk": "Cảnh báo"},
    {"building": "Tòa nhà B05", "zones": 4, "sensors": 20, "normal_pct": 75.9, "ventilation_pct": 17.6, "thermal_pct": 6.5, "risk": "Thấp"},
    {"building": "Tòa nhà B06", "zones": 4, "sensors": 20, "normal_pct": 74.0, "ventilation_pct": 19.8, "thermal_pct": 6.2, "risk": "Trung bình"}
]

# Danh sách đầy đủ 24 khu vực
ALL_ZONES = [f"B0{b}-Z0{z}" for b in range(1, 7) for z in range(1, 5)]
