"""
Thành viên 2: Đặc trưng Vật lý (Feature Engineering) & Xử lý Missing Values.
Nhiệm vụ:
- Điền khuyết (Imputation) bằng median theo building_zone.
- Kỹ nghệ đặc trưng nhiệt độ & độ ẩm, bụi, hệ thống HVAC.
"""
import os
os.makedirs("data/processed", exist_ok=True)
if __name__ == "__main__":
    print("Chạy Module Feature Engineering...")
