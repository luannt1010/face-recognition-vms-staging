# Database nhận diện khuôn mặt

Compose dùng PostgreSQL 16 với [image pgvector](https://github.com/pgvector/pgvector#docker).
Đọc POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD và POSTGRES_HOST_PORT từ `.env`.
File khởi tạo là `sql/init.sql` do bạn cung cấp: bảng employees, embedding 512 chiều,
mỗi ma_nv duy nhất và các thông tin nhân viên.

Cấu hình giữ tên project `task2_atin`, container `face_recognition_pgvector` và
volume `face_recognition_pgdata` của database đang chạy. Compose dùng lại volume đó.

Chạy từ root project:

```powershell
docker compose config --quiet
docker compose up -d db
docker compose ps
docker compose logs --tail 50 db
```

Nếu container bị tạo lại khi chạy up, các kết nối database sẽ ngắt ngắn hạn;
nên dừng main/demo trước khi thực hiện rồi chạy lại sau khi db healthy.

Python chạy trực tiếp trên Windows kết nối bằng DATABASE_URL trong `.env`,
host localhost/127.0.0.1 và port POSTGRES_HOST_PORT (hiện là 55432).
Khi Python chạy trong cùng mạng Compose thì dùng PGHOST=db, PGPORT=5432 và
các biến POSTGRES_*. Không đặt PGHOST=db khi chạy Python trực tiếp trên Windows.

Mở psql để kiểm tra:

```powershell
docker compose exec db sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Sau đó nhập:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
\d employees
SELECT count(*) FROM employees;
```

`docker compose down` vẫn giữ volume dữ liệu. SQL khởi tạo chỉ tự chạy với
volume rỗng theo [cơ chế PostgreSQL image](https://hub.docker.com/_/postgres).
Với volume hiện có, chỉnh sql/init.sql không tự thay đổi schema đã tạo trước đó;
đổi POSTGRES_* cũng không tự đổi tài khoản/database đã có.

Nếu khởi tạo trên máy mới, bảng employees ban đầu rỗng. Cần enroll ảnh mẫu
bằng cùng model extractor để có embedding; employees.csv không tự nhập vào bảng.
