# Thêm nhân viên từ CSV

Chạy trong thư mục root bằng cùng môi trường Python của main/demo:

```powershell
python -m scripts.enroll_employees --image-root D:/private/task2_atin --check
python -m scripts.enroll_employees --image-root D:/private/task2_atin
```

Ảnh của CSV hiện có đang nằm ở `D:/private/task2_atin/pics`.
Nếu chuyển thư mục pics vào project này thì bỏ `--image-root`.
Mặc định đọc `configs/employees.csv`; đổi bằng `--csv`.
Thêm `--device cpu` nếu muốn detector chạy CPU.

Luồng: đọc CSV -> detect đúng một mặt -> crop -> extract vector -> lưu database.
Dùng cùng model, cách crop và tiền xử lý với main. Ảnh không có mặt, có nhiều
mặt hoặc mặt quá nhỏ bị bỏ qua, có log lý do. `--check` chỉ kiểm tra CSV/đường
dẫn, không kiểm tra nội dung ảnh, không chạy model hay ghi DB.

Database lấy từ `.env` và phải có bảng theo `sql/init.sql`.
Chạy lại cùng user_code sẽ cập nhật tên và embedding, giữ các thông tin khác
như phòng ban/số điện thoại. Mỗi nhân viên lưu thành công được commit riêng.
Nếu có dòng bị bỏ qua thì script trả exit code 1, các dòng thành công vẫn được lưu.
