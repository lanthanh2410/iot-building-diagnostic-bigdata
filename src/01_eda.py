"""
Thành viên 1: Khám phá Dữ liệu Lớn (EDA) & Tiền xử lý.
Nhiệm vụ: 
- Load tập dữ liệu lớn.
- Chia tập Train/Test theo chuỗi thời gian năm 2025.
- Loại bỏ các cột không cần thiết (reading_id, v.v.).
"""
import os
os.makedirs("data/processed", exist_ok=True)
if __name__ == "__main__":
    print("Chạy Module EDA...")
