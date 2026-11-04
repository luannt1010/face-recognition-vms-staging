# Báo cáo LaTeX module FACE

Mở thư mục này trong trình soạn LaTeX của Codex để xem PDF preview, hoặc compile
`main.tex` bằng XeLaTeX. Sử dụng font Times New Roman, Arial và Courier New.
Trên Overleaf chọn compiler XeLaTeX; nếu thiếu font, đổi ba lệnh setfont đầu file
sang các font có hỗ trợ tiếng Việt trong môi trường đó.

Ba trường để trống ở đầu main.tex: `TenBaoCao`, `HoTen`, `Github`.
Điền nội dung trong cặp ngoặc nhọn của từng newcommand. Với GitHub có ký tự
đặc biệt, dùng `\url{https://github.com/...}` làm nội dung.

Ảnh và nội dung cần bổ sung trước khi nộp:

- `images/vms_bbox.png`: bbox FACE trên trang giám sát trực tiếp.
- `images/vms_event.png`: lịch sử FACE và snapshot đã hiển thị trên Web VMS.
- Link video demo ở phụ lục.
- Số liệu kiểm thử danh tính và số lượt IN/OUT có nhãn, nếu đã thu thập.

Nếu chưa có ảnh, LaTeX vẫn compile và hiện khung để điền. Sau khi thêm ảnh thật,
cập nhật caption và các câu ghi “cần bổ sung” cho đúng kết quả đã xác nhận.
Báo cáo dùng lối viết trung tính, lựa chọn TensorRT FP16 theo cấu hình đã chốt.
