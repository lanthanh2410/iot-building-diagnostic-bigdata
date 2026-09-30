# KỊCH BẢN THUYẾT TRÌNH DEMO BÁO CÁO (2 - 3 PHÚT)
## Thành Viên 6: Thực Nghiệm Benchmark Đối Đầu & Web App Streamlit

> **Thời lượng**: 2 phút 30 giây đến 3 phút  
> **Công cụ trình chiếu**: Ứng dụng Streamlit Web App (`http://localhost:8501`)  
> **Mục tiêu**: Làm nổi bật giá trị Big Data cốt lõi của đồ án, chứng minh ưu thế của Apache Spark so với Scikit-Learn trên quy mô 2.100.000 bản ghi, và trình diễn khả năng ứng dụng thực tế vào tòa nhà thông minh.

---

### PHẦN 1: MỞ ĐẦU & TỔNG QUAN HỆ THỐNG (0:00 - 0:30)

* **Thao tác màn hình**:
  - Mở sẵn trình duyệt tại trang chủ Web App: `http://localhost:8501`.
  - Giữ màn hình ở **Tab 1: Chẩn Đoán Thời Gian Thực**.
* **Lời thoại thuyết trình**:
  > *"Kính thưa thầy/cô và hội đồng, em là Thành viên 6, chịu trách nhiệm thực hiện bộ thực nghiệm Benchmark đối đầu giữa Scikit-Learn và Apache Spark MLlib, đồng thời lập trình ứng dụng Web Dashboard thời gian thực trên nền tảng Streamlit.*
  > 
  > *Giao diện trước mắt thầy/cô đang kết nối với hệ thống giám sát gồm 120 cảm biến tại 24 khu vực thuộc 6 tòa nhà thông minh, quản lý hơn 2.1 triệu bản ghi IoT. Hệ thống tự động phân loại 3 trạng thái vận hành: Bình thường, Sự cố thông khí và Sự cố điều nhiệt."*

---

### PHẦN 2: DEMO TÍNH NĂNG CHẨN ĐOÁN THỜI GIAN THỰC (0:30 - 1:15)

* **Thao tác màn hình**:
  1. Nhìn vào thanh Sidebar bên trái: Click chọn dropdown **Kịch bản 1: Phòng làm việc tối ưu**.
  2. Bấm nút **BẮT ĐẦU CHẨN ĐOÁN AI** -> Chỉ vào thẻ trạng thái màu xanh lá: **🟢 Bình thường (Normal) - Độ tin cậy 98.9%**.
  3. Tiếp tục chọn **Kịch bản 2: Nghẹt màng lọc & CO2 tăng cao**.
  4. Bấm **BẮT ĐẦU CHẨN ĐOÁN AI** -> Thẻ cảnh báo chuyển ngay sang màu đỏ rực: **🔴 Sự Cố Thông Khí (Ventilation Issue) - 100%**.
  5. Cuộn nhẹ xuống dưới, chỉ vào 2 biểu đồ cột: **Phân bổ xác suất** và **Đặc trưng ảnh hưởng lớn nhất (Top Features)**.
* **Lời thoại thuyết trình**:
  > *"Tại Tab 1, hệ thống cung cấp 2 chế độ: chọn nhanh các kịch bản kiểm thử thực tế hoặc tinh chỉnh thanh trượt cảm biến tùy ý.*
  > 
  > *Ví dụ, khi em chọn Kịch bản 2 - phòng họp đông người bị nghẹt màng lọc: Áp suất lọc tăng vọt lên 145 Pa và CO2 vượt 1.250 ppm. Mô hình AI lập tức phát hiện sự cố thông khí với độ tin cậy 100%. Bên dưới, hệ thống giải thích rõ ràng các đặc trưng dẫn đến quyết định (Explainable AI) như chênh lệch áp suất lọc và lưu lượng gió, đồng thời đưa ra khuyến nghị kỹ thuật ngay lập tức cho đội bảo trì."*

---

### PHẦN 3: ĐIỂM SÁNG BENCHMARK BIG DATA & ĐIỂM GIAO THOA (1:15 - 2:15)
*(Đây là phần quan trọng nhất giúp ghi điểm tuyệt đối về chuyên môn Big Data)*

* **Thao tác màn hình**:
  - Click chuyển sang **Tab 3: Phòng Thí Nghiệm Benchmark Big Data**.
  - Cuộn đến **Hình 1: Biểu đồ Thời gian huấn luyện & Điểm giao thoa (Cross-over Point)**.
  - Sau đó chỉ vào **Hình 2 (Peak RAM)** và **Bảng đối đầu kỹ thuật** phía dưới.
* **Lời thoại thuyết trình**:
  > *"Tiếp theo là phần cốt lõi của môn học Big Data: Thực nghiệm đối đầu trực tiếp giữa Scikit-Learn và Apache Spark MLlib trên 4 mốc dữ liệu: 200 nghìn, 500 nghìn, 1 triệu và trọn vẹn 2.1 triệu dòng.*
  > 
  > *Như thầy/cô quan sát trên Hình 1, nhóm em đã phát hiện và chứng minh được **Điểm giao thoa (Cross-over Point) tại ngưỡng khoảng 820.000 dòng**:*
  > - *Ở quy mô nhỏ dưới 820K dòng, Scikit-Learn chạy nhanh hơn vì không phải chịu chi phí khởi tạo JVM, RPC và Shuffle của Spark.*
  > - *Tuy nhiên, khi chạm ngưỡng Big Data từ 1 triệu đến 2.1 triệu dòng, Spark MLlib phân tán vượt trội hoàn toàn: Thời gian huấn luyện chỉ mất 178 giây (so với gần 460 giây của Scikit-Learn, nhanh hơn 2.5 lần).*
  > 
  > *Đặc biệt ở Hình 2 về tiêu thụ RAM: Scikit-Learn ngốn tới 14.2 GB RAM và có nguy cơ sập Out-Of-Memory trên máy đơn lẻ. Trong khi đó, Apache Spark nhờ cơ chế phân vùng RDD và bộ nhớ đệm tự động tràn sang đĩa (Spill-to-disk) chỉ tiêu tốn 4.15 GB RAM ổn định, tiết kiệm hơn 71% bộ nhớ."*

---

### PHẦN 4: KHẢ NĂNG MỞ RỘNG ĐA LÕI & KẾT LUẬN (2:15 - 2:45)

* **Thao tác màn hình**:
  - Chỉ vào **Hình 3: Tốc độ tăng tốc theo số lõi CPU (2 cores, 4 cores, 8 cores)**.
  - Chuyển nhanh qua **Tab 2 (Giám sát tòa nhà)** để thầy cô thấy bản đồ tổng thể.
* **Lời thoại thuyết trình**:
  > *"Trên quy mô 2.1 triệu dòng, nhóm cũng đo đạc tính mở rộng đa lõi của Spark: Khi nâng từ 2 cores lên 4 cores và 8 cores, thời gian giảm từ 328 giây xuống còn 104 giây, đạt hệ số tăng tốc 3.15 lần với hiệu suất song song xấp xỉ 80%.*
  > 
  > *Toàn bộ mã nguồn Web App, bộ 5 biểu đồ độ phân giải cao và số liệu log đã được nhóm em đóng gói hoàn thiện và tích hợp vào đồ án. Em xin kết thúc phần demo và sẵn sàng lắng nghe câu hỏi từ thầy/cô!"*

---

## BỘ CÂU HỎI PHẢN BIỆN DỰ KIẾN TỪ HỘI ĐỒNG & CÁCH TRẢ LỜI

### Câu hỏi 1: *"Tại sao ở 200.000 dòng Scikit-Learn lại nhanh hơn Apache Spark?"*
- **Trả lời**: *"Thưa thầy/cô, Apache Spark hoạt động trên kiến trúc Master-Worker phức tạp. Mỗi khi khởi động job, Spark phải chịu chi phí cố định (overhead) gồm: khởi tạo JVM, phân chia partition, giao tiếp RPC giữa Driver và Executors, và lập lịch DAG. Ở tập dữ liệu nhỏ (200K dòng), thời gian overhead này chiếm phần lớn. Trong khi đó Scikit-Learn là thư viện C/Cython chạy trực tiếp trên bộ nhớ RAM tiến trình đơn lẻ nên không có chi phí này. Chỉ khi dữ liệu đủ lớn (vượt ngưỡng giao thoa 820K dòng), sức mạnh tính toán song song của Spark mới bù đắp và vượt xa overhead ban đầu."*

### Câu hỏi 2: *"Cơ chế nào giúp Spark không bị lỗi tràn bộ nhớ (Out-Of-Memory) như Scikit-Learn khi xử lý 2.1 triệu dòng?"*
- **Trả lời**: *"Dạ thưa thầy/cô, Scikit-Learn bắt buộc phải nạp toàn bộ ma trận dữ liệu vào một khối RAM liên tục (in-memory contiguous array). Khi huấn luyện Random Forest, các luồng threads nhân bản thêm bản sao dữ liệu dẫn đến RAM vọt lên 14 - 16 GB. Ngược lại, Spark sử dụng kiến trúc bộ nhớ Unified Memory Manager chia dữ liệu thành các Partitions nhỏ (ví dụ 8 - 16 partitions). Nếu bộ nhớ RAM đạt ngưỡng cấu hình, Spark sẽ tự động kích hoạt cơ chế Spill-to-Disk (ghi tạm các phân vùng xuống ổ cứng SSD) và giải phóng bộ nhớ đệm, đảm bảo job không bao giờ bị crash OOM."*

### Câu hỏi 3: *"Web App Streamlit của nhóm gọi mô hình như thế nào để đảm bảo tốc độ phản hồi nhanh cho người dùng?"*
- **Trả lời**: *"Dạ, nhóm em áp dụng kiến trúc Hybrid Inference Engine: Để phục vụ người dùng cuối với độ trễ thấp (< 50ms), mô hình được export ra định dạng tối ưu và chạy suy luận cục bộ, tự động tính toán các đặc trưng vật lý IoT (chênh lệch nhiệt độ, áp suất lọc, tỷ lệ năng lượng HVAC) ngay khi người dùng kéo thanh trượt. Đối với bài toán huấn luyện lại hoặc phân tích dữ liệu lớn định kỳ hàng triệu bản ghi, hệ thống sẽ gửi lệnh batch-job về cho cụm Spark xử lý."*
