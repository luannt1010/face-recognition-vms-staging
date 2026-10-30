# Nhận diện khuôn mặt và tích hợp VMS

Project xử lý video RTSP để phát hiện, theo dõi và nhận diện khuôn mặt, xác định
lượt IN/OUT theo vùng hoặc vạch, sau đó gửi kết quả tới VMS qua MQTT.

Luồng xử lý:

```text
VMS gửi camera và vùng/vạch qua MQTT
                 ↓
RTSP → Detect → Track → Extract/Search → Event IN/OUT
                                              ↓
                         Upload snapshot → Publish FACE và bbox
                                              ↓
                                      Backend / Web VMS
```

## Thành phần chính

| Thành phần | Chức năng |
|---|---|
| YOLO11n, TensorRT FP16 | Phát hiện khuôn mặt |
| ByteTrack | Theo dõi và gán ID qua các frame |
| Extractor ONNX | Tạo embedding 512 chiều từ ảnh mặt |
| PostgreSQL + pgvector | Lưu nhân viên, tìm ứng viên bằng cosine similarity |
| Event processor | Kiểm tra tâm mặt đi qua polygon hoặc line |
| Identity cache | Giảm số lần extract/search theo track |
| VMS client | Nhận cấu hình, upload snapshot, gửi sự kiện và bbox |

Database nhân viên của module AI phục vụ so khớp. Backend VMS tiếp nhận sự kiện;
danh sách nhân viên giữa hai hệ thống chưa được đồng bộ tự động.

## Cấu trúc thư mục

```text
main.py                         Vòng xử lý camera và publish kết quả
configs/                        Cấu hình model, runtime, CSV nhân viên, vùng demo
src/core_ai/                    Reader, detector, tracker, extractor, event, cache
src/database/                   Truy vấn vector và xác minh danh tính
src/vms_service/                MQTT client và upload snapshot
scripts/enroll_employees.py     Đăng ký nhân viên từ CSV
sql/init.sql                    Schema PostgreSQL và extension vector
docker-compose.yml              Database chạy Docker
experiments/detector_tensorrt/   Notebook export và benchmark detector
tests/                          Kiểm thử offline
bao_cao_face/                   Báo cáo LaTeX
```

## Môi trường

Sử dụng môi trường Python đã có PyTorch CUDA, OpenCV, Ultralytics, ONNX,
ONNX Runtime, NumPy, supervision, trackers, psycopg, pgvector, PyYAML,
python-dotenv, paho-mqtt 2.x và requests.

Cấu hình hiện tại dùng TensorRT FP16 trên GPU NVIDIA. Kernel/runtime chạy
engine cần tương thích với môi trường build. Engine nên được export trên GPU
đích bằng notebook trong `experiments/detector_tensorrt`.

Nếu dùng detector `.pt`, đổi đường dẫn model trong config sang checkpoint phù
hợp. File engine và model embedding cần được chuẩn bị trước khi chạy.

## Cấu hình

Tạo `.env` tại thư mục gốc bằng thông tin của môi trường triển khai:

```dotenv
POSTGRES_DB=face_recognition
POSTGRES_USER=face_app
POSTGRES_PASSWORD=CHANGE_ME
POSTGRES_HOST_PORT=55432
DATABASE_URL=postgresql://face_app:CHANGE_ME@127.0.0.1:55432/face_recognition

MQTT_HOST=your-broker-domain
MQTT_PORT=your-port
MQTT_USERNAME=your-user
MQTT_PASSWORD=your-password
MQTT_CLIENT_ID=your-id
VMS_ZONE_ID=your-zone-id
VMS_ORGANIZATION_ID=your-organiztion-id
VMS_STORAGE_UPLOAD_URL=your-url
```

Các giá trị trên là ví dụ, cần thay bằng cấu hình thực tế. Mật khẩu trong
`DATABASE_URL` cần URL-encode nếu có ký tự đặc biệt. Tên database, user,
password và port của URL phải khớp với các biến `POSTGRES_*`.

`MQTT_HOST` chỉ chứa hostname, không thêm `wss://` hoặc `/mqtt`. Không đưa
`.env` có thông tin thật lên repository.

Model và tham số xử lý được cấu hình trong `configs/config.yml`:

```yaml
model_detect:
  weight: 'experiments/detector_tensorrt/outputs/img640/fp16/detector.engine'

model_extract:
  weight: 'path/to/face_embedding_model.onnx'

runtime:
  processing_fps: 10
  recognition_threshold: 0.4
  recognition_interval: 0.5
  event_cooldown: 1.0
  track_ttl: 2.0
```

Đoạn YAML trên chỉ minh họa các mục chính, không thay thế toàn bộ config.
Đường dẫn tương đối của model tính từ root project. Config extractor hiện có
đang trỏ tới project cũ, cần sửa nếu chuyển máy. Hai biến tùy chọn
`FACE_DETECTOR_WEIGHTS` và `FACE_EXTRACTOR_WEIGHTS` trong môi trường được ưu tiên
hơn đường dẫn YAML.

## Khởi động database

Chạy các lệnh từ thư mục gốc:

```powershell
docker compose config --quiet
docker compose up -d db
docker compose ps
```

Compose dùng container `face_recognition_pgvector`, project `task2_atin` và
volume `face_recognition_pgdata` để giữ dữ liệu của database hiện có.
`sql/init.sql` chỉ tự chạy khi volume rỗng; thay file SQL không tự migrate
schema cũ. Database mới chưa có embedding nhân viên.

Python chạy trực tiếp trên Windows dùng `DATABASE_URL` qua localhost và port
đã publish. `PGHOST=db` chỉ dùng khi chương trình chạy trong cùng mạng Compose;
nếu có `PGHOST`, code ưu tiên cấu hình này thay cho `DATABASE_URL`.

## Đăng ký nhân viên

File `configs/employees.csv` có các cột:

```csv
user_code,user_name,image_path
EMP001,Nguyen Van A,pics/employee_01.jpg
```

Mỗi ảnh cần có đúng một khuôn mặt đủ lớn. Script dùng cùng detector/extractor
với main, tạo embedding rồi lưu theo mã nhân viên. Chạy lại cùng mã sẽ cập nhật
tên và embedding, giữ các thông tin nhân viên khác.

Nếu thư mục `pics` nằm trong project hiện tại:

```powershell
python -m scripts.enroll_employees --check
python -m scripts.enroll_employees
```

Các ảnh của CSV hiện tại nằm trong project cũ; có thể chỉ định thư mục gốc ảnh:

```powershell
python -m scripts.enroll_employees --image-root D:/private/task2_atin --check
python -m scripts.enroll_employees --image-root D:/private/task2_atin
```

`--check` chỉ kiểm tra CSV và file ảnh, không chạy model hoặc ghi database.
Các dòng lỗi được bỏ qua và ghi log; các dòng thành công được lưu riêng.

## Chạy demo local

```powershell
python -m src.core_ai.camera_reader --rtsp "rtsp://IP:PORT/stream" --debug
```

Demo hiển thị khung mặt, tên, điểm so khớp và IN/OUT. Mặc định sử dụng polygon
trong `configs/demo_zones.json`; có thể đổi bằng `--zones path/to/zones.json`.
Terminal in `[IN]` hoặc `[OUT]` khi có lượt qua biên. Nhấn Q/Esc để dừng.

Demo không kết nối hay publish lên VMS. Demo vẫn cần database để so khớp danh tính.

## Chạy pipeline VMS

```powershell
python main.py
```

Main tự nhận camera bật FACE, lấy RTSP và cập nhật vùng/vạch từ VMS. Camera
không có RTSP được bỏ qua. Main không mở cửa sổ OpenCV; kết quả được xem trên
Web VMS và terminal. Nhấn Ctrl+C để dừng.

Camera cần thuộc phạm vi cấu hình, bật module FACE và có vùng/vạch FACE hoạt
động để tạo IN/OUT. Trên trang giám sát trực tiếp, chọn chức năng FACE và bật
Phát hiện AI để xem bbox. Xem thêm log từng lần publish bbox bằng:

```powershell
python main.py --debug
```

| Chức năng | Topic | QoS |
|---|---|---|
| Xin camera | `smart_vms/cameras/getdevice` | 1 |
| Nhận camera | `smart_vms/cameras/zone/{zone_id}` | 1 |
| Nhận polygon và line FACE | `smart_vms/cameras/+/ai_zones/FACE` | 1 |
| Gửi sự kiện | `smart_vms/ai_events/FACE` | 1 |
| Gửi bbox | `smart_vms/ai/bbox/{camera_code}` | 0 |

Theo hợp đồng VMS, `ai_zones/FACE` đã có cả polygon và line crossing; client
không subscribe riêng `/FACE/lines`. Khi có sự kiện, ảnh được upload qua
Storage API; `object_key` trả về được gửi ở `snapshot_url`. Upload lỗi thì có
thể gửi event không ảnh. Người lạ không được gửi `user_code`.

Publish FACE thành công chỉ xác nhận broker ACK, chưa xác nhận backend lưu
sự kiện. Cần kiểm tra lịch sử và snapshot trên Web VMS để xác nhận toàn luồng.

## Quy tắc sự kiện và cache

- Điểm đại diện là tâm bbox khuôn mặt.
- Polygon: từ ngoài vào là IN, từ trong ra là OUT.
- Line crossing: đi sang phía `direction_point` là IN, chiều ngược lại là OUT.
  Chỉ cắt đoạn vạch hữu hạn mới được tính.
- Track mới xuất hiện chỉ ghi nhớ vị trí; chưa tạo sự kiện ngay.
- Chỉ detection thật được dùng; bỏ bbox dự đoán.
- Cooldown áp dụng theo camera, vùng và track. Lượt xảy ra trong cooldown bị
  bỏ qua, không gửi trễ sau đó.
- Mỗi camera có cache identity riêng. Track mới được nhận diện ngay, sau đó
  làm mới mỗi `recognition_interval` giây. Frame có IN/OUT dùng identity mới;
  nếu đã nhận diện tại frame đó thì không extract lại.
- Track hết hạn hoặc tracker reset sẽ xóa cache. Đặt `recognition_interval: 0`
  để nhận diện mọi frame. Extract vẫn chạy từng mặt, không batch.

## Kết quả benchmark detector

Kết quả trên RTX 4050 Laptop, input 640, batch 1, 300 lần đo/model:

| Biến thể | Predict trung bình (ms) | mAP50 (%) | mAP50–95 (%) |
|---|---:|---:|---:|
| PyTorch FP32 | 14.54 | 95.32 | 67.27 |
| TensorRT FP32 | 7.45 | 95.32 | 67.28 |
| TensorRT FP16 | 7.24 | 95.33 | 67.35 |
| TensorRT INT8 | 6.93 | 94.25 | 66.84 |

TensorRT FP16 là cấu hình được lựa chọn. Timing gồm tiền xử lý, forward,
hậu xử lý và lấy kết quả về CPU; FPS detector không phải FPS toàn pipeline.
Độ chính xác được đánh giá trên 164 ảnh test, INT8 calibration bằng 500 ảnh
train. Các số mAP phản ánh detector, không phản ánh nhận diện danh tính.

Notebook, dữ liệu đo và hướng dẫn export nằm trong
`experiments/detector_tensorrt`. Thay GPU hoặc môi trường runtime có thể cần
build lại engine.

## Kiểm thử và hạn chế

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử offline dùng dữ liệu giả để kiểm tra logic; không thay thế đánh giá
camera thực tế. Các test demo được bỏ qua nếu thiếu OpenCV/python-dotenv.

Model extractor còn ở giai đoạn thử nghiệm và cần tiếp tục huấn luyện/tinh
chỉnh cho camera thực tế. Danh sách hiện có một ảnh mẫu/người, chưa căn chỉnh
mặt theo landmarks; người đã đăng ký đôi lúc vẫn bị báo là người lạ.
Chưa có số liệu đánh giá danh tính độc lập hoặc đo mức cải thiện độ trễ do cache.

Các hạn chế khác gồm khả năng lặp sự kiện khi track đổi ID, bỏ sót khi đối
tượng đi xuyên polygon giữa hai frame và upload/publish event đang chặn vòng
xử lý. Hệ thống chưa có hàng đợi sự kiện bền vững để gửi bù khi offline.

Hướng phát triển gồm bổ sung ảnh mẫu đa góc, cải thiện extractor, đánh giá
ngưỡng cosine trên dữ liệu độc lập, đo thời gian từng bước và tách upload/
publish event sang worker riêng.


