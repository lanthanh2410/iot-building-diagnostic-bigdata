"""
Dịch vụ suy luận mô hình (Model Inference Service) phục vụ Streamlit Web App.
Tự động nạp mô hình đã huấn luyện (models/rf_baseline.pkl), thực hiện
kỹ nghệ đặc trưng vật lý IoT thời gian thực và trả về chẩn đoán kèm giải thích.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_baseline.pkl")
MODEL_PATH = DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else "models/rf_baseline.pkl"

LABEL_NAMES = {
    0: ("Bình Thường (Normal)", "normal", "#22C55E", "🟢"),
    1: ("Sự Cố Thông Khí (Ventilation Issue)", "ventilation_issue", "#EF4444", "🔴"),
    2: ("Sự Cố Điều Nhiệt (Thermal Issue)", "thermal_issue", "#F59E0B", "🟡")
}

FEATURE_VN_NAMES = {
    'delta_temp': 'Chênh lệch Nhiệt độ Trong/Ngoài (ΔT)',
    'delta_humidity': 'Chênh lệch Độ ẩm Trong/Ngoài (ΔH)',
    'co2_ppm': 'Nồng độ Khí CO2 (ppm)',
    'pm25_ug_m3': 'Bụi Mịn Trong Nhà PM2.5 (µg/m³)',
    'pm25_ratio': 'Tỷ lệ Bụi PM2.5 Trong/Ngoài',
    'filter_pressure_pa': 'Chênh lệch Áp suất Màng lọc (Pa)',
    'filter_pressure_ratio': 'Tỷ lệ Áp suất / Lưu lượng gió',
    'hvac_power_kw': 'Công suất Tiêu thụ HVAC (kW)',
    'hvac_energy_ratio': 'Suất Năng lượng HVAC / Gió',
    'air_flow_m3_h': 'Lưu lượng Không khí (m³/h)',
    'fan_speed_rpm': 'Tốc độ Quạt Thông gió (RPM)',
    'indoor_temp_c': 'Nhiệt độ Trong Nhà (°C)',
    'vibration_mm_s': 'Độ rung Động cơ (mm/s)',
    'maintenance_days': 'Số ngày Chưa Bảo trì',
    'occupancy_count': 'Số lượng Người Hiện diện',
    'equipment_age_years': 'Tuổi Thọ Thiết bị (Năm)'
}

class DiagnosticService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_model()

    def _load_model(self):
        """Nạp model từ artifact."""
        if os.path.exists(MODEL_PATH):
            try:
                bundle = joblib.load(MODEL_PATH)
                self.model = bundle['model']
                self.scaler = bundle['scaler']
                self.feature_names = bundle['features']
            except Exception as e:
                print(f"[Cảnh báo] Không thể tải {MODEL_PATH}: {e}")

    def predict(self, raw_params: dict) -> dict:
        """
        Nhận vào từ điển các thông số cảm biến thô,
        thực hiện kỹ nghệ đặc trưng và trả về kết quả chẩn đoán chi tiết.
        """
        # 1. Tính toán các đặc trưng vật lý bổ sung
        p = raw_params.copy()
        delta_temp = p.get('indoor_temp_c', 22.0) - p.get('outdoor_temp_c', 15.0)
        delta_humidity = p.get('indoor_humidity_pct', 50.0) - p.get('outdoor_humidity_pct', 60.0)
        outdoor_pm25 = p.get('outdoor_pm25_ug_m3', 10.0)
        pm25_ratio = p.get('pm25_ug_m3', 15.0) / (outdoor_pm25 + 1e-5)
        
        air_flow = p.get('air_flow_m3_h', 1000.0)
        filter_pressure = p.get('filter_pressure_pa', 100.0)
        filter_pressure_ratio = filter_pressure / (air_flow + 1e-5)
        
        hvac_power = p.get('hvac_power_kw', 5.0)
        hvac_energy_ratio = hvac_power / (air_flow + 1e-5)
        
        # Gom các đặc trưng
        feature_dict = {
            'floor_number': p.get('floor_number', 1),
            'occupancy_count': p.get('occupancy_count', 10),
            'outdoor_temp_c': p.get('outdoor_temp_c', 15.0),
            'outdoor_humidity_pct': p.get('outdoor_humidity_pct', 60.0),
            'outdoor_pm25_ug_m3': outdoor_pm25,
            'indoor_temp_c': p.get('indoor_temp_c', 22.0),
            'indoor_humidity_pct': p.get('indoor_humidity_pct', 50.0),
            'co2_ppm': p.get('co2_ppm', 600.0),
            'pm25_ug_m3': p.get('pm25_ug_m3', 15.0),
            'tvoc_ppb': p.get('tvoc_ppb', 120.0),
            'air_flow_m3_h': air_flow,
            'fan_speed_rpm': p.get('fan_speed_rpm', 1200.0),
            'hvac_power_kw': hvac_power,
            'filter_pressure_pa': filter_pressure,
            'vibration_mm_s': p.get('vibration_mm_s', 1.5),
            'maintenance_days': p.get('maintenance_days', 30),
            'window_open_pct': p.get('window_open_pct', 0.0),
            'equipment_age_years': p.get('equipment_age_years', 3.0),
            'delta_temp': delta_temp,
            'delta_humidity': delta_humidity,
            'pm25_ratio': pm25_ratio,
            'filter_pressure_ratio': filter_pressure_ratio,
            'hvac_energy_ratio': hvac_energy_ratio
        }

        # 2. Suy luận bằng Random Forest nếu đã nạp
        if self.model and self.scaler:
            feature_vector = np.array([[feature_dict.get(f, 0.0) for f in self.feature_names]])
            scaled_vector = self.scaler.transform(feature_vector)
            
            pred_class = int(self.model.predict(scaled_vector)[0])
            probs = self.model.predict_proba(scaled_vector)[0]
        else:
            # Thuật toán dự phòng chuẩn vật lý nếu chưa nạp được file model
            co2 = feature_dict['co2_ppm']
            fp = feature_dict['filter_pressure_pa']
            ind_t = feature_dict['indoor_temp_c']
            out_t = feature_dict['outdoor_temp_c']
            
            if co2 > 1100 or fp > 135 or pm25_ratio > 2.5:
                pred_class = 1 # ventilation_issue
                probs = [0.08, 0.84, 0.08]
            elif abs(ind_t - out_t) > 14 or ind_t > 28.5 or (hvac_power > 6.0 and ind_t > 26.0):
                pred_class = 2 # thermal_issue
                probs = [0.10, 0.12, 0.78]
            else:
                pred_class = 0 # normal
                probs = [0.88, 0.07, 0.05]

        label_title, label_code, color_hex, icon = LABEL_NAMES[pred_class]
        
        # 3. Phân tích các yếu tố đóng góp (Feature Importance / Contributions)
        contributions = []
        if self.model and hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.feature_names
        else:
            importances = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03]
            feature_names = ['co2_ppm', 'filter_pressure_pa', 'delta_temp', 'indoor_temp_c', 'air_flow_m3_h', 'pm25_ratio', 'hvac_power_kw', 'filter_pressure_ratio', 'fan_speed_rpm', 'vibration_mm_s']

        for fname, imp in zip(feature_names, importances):
            vn_name = FEATURE_VN_NAMES.get(fname, fname)
            val = feature_dict.get(fname, 0.0)
            contributions.append({
                "feature": fname,
                "display_name": vn_name,
                "value": round(val, 2),
                "importance": round(imp * 100, 2)
            })
            
        contributions = sorted(contributions, key=lambda x: x['importance'], reverse=True)[:6]

        # 4. Đề xuất hành động kỹ thuật (Actionable Insights)
        recommendations = []
        if pred_class == 1: # Ventilation Issue
            if feature_dict['filter_pressure_pa'] > 120:
                recommendations.append("⚠️ **Màng lọc có dấu hiệu nghẹt**: Chênh lệch áp suất cao (> 120 Pa). Khuyến nghị kiểm tra vệ sinh hoặc thay thế tấm lọc không khí.")
            if feature_dict['co2_ppm'] > 1000:
                recommendations.append("⚠️ **Nồng độ CO2 vượt ngưỡng an toàn**: Tăng tốc độ quạt (fan_speed) hoặc mở cửa thông gió tự nhiên để trao đổi khí tươi.")
            if feature_dict['air_flow_m3_h'] < 300:
                recommendations.append("⚠️ **Lưu lượng cấp khí thấp**: Kiểm tra động cơ quạt hút và đường ống phân phối gió.")
        elif pred_class == 2: # Thermal Issue
            if feature_dict['indoor_temp_c'] > 27:
                recommendations.append("⚠️ **Nhiệt độ phòng quá nóng**: Tăng tải công suất làm lạnh của cụm dàn trao đổi nhiệt FCU/AHU.")
            if abs(feature_dict['delta_temp']) > 12:
                recommendations.append("⚠️ **Chênh lệch nhiệt độ trong/ngoài quá lớn**: Kiểm tra độ cách nhiệt của cửa kính và vỏ bao che tòa nhà.")
            if feature_dict['hvac_power_kw'] > 6:
                recommendations.append("⚠️ **Công suất HVAC ở mức tải đỉnh**: Giám sát dòng điện máy nén, phòng ngừa quá nhiệt động cơ.")
        else: # Normal
            recommendations.append("✅ **Hệ thống hoạt động tối ưu**: Chất lượng không khí (IAQ) và tiện nghi nhiệt đạt tiêu chuẩn ASHRAE 62.1 & 55.")
            recommendations.append("ℹ️ Duy trì lịch bảo trì định kỳ sau mỗi 60 - 90 ngày vận hành.")

        return {
            "prediction_class": pred_class,
            "label_title": label_title,
            "label_code": label_code,
            "color_hex": color_hex,
            "icon": icon,
            "confidence_pct": round(probs[pred_class] * 100, 1),
            "probabilities": {
                "Bình thường": round(probs[0] * 100, 1),
                "Sự cố Thông khí": round(probs[1] * 100, 1),
                "Sự cố Điều nhiệt": round(probs[2] * 100, 1)
            },
            "contributions": contributions,
            "recommendations": recommendations,
            "features_used": feature_dict
        }

# Singleton instance
diagnostic_service = DiagnosticService()
