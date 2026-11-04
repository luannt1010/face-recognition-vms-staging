# Nhận diện khuôn mặt và tích hợp VMS

Pipeline Python đọc video RTSP, theo dõi người, nhận diện khuôn mặt và phát
sự kiện IN/OUT theo vùng hoặc vạch. Kết quả được gửi tới VMS qua MQTT;
ảnh sự kiện được upload qua Storage API.

```text
RTSP → YOLO (person) → Deep SORT → Person track → IN/OUT theo vùng/vạch
                         ↑             ↓
               SCRFD → InsightFace → Danh tính theo track
                                       ↓
                         MQTT bbox / FACE event + snapshot
```

## Thành phần chính

| Thành phần | Vai trò |
|---|---|
| YOLO11n / `PersonDetector` | Phát hiện người, lọc COCO class 0 |
| `PersonTrackerDeepSort` | Theo dõi người; mặc định dùng embedder MobileNet |
| SCRFD | Phát hiện khuôn mặt và 5 landmarks |
| `InsightFaceExtractor` | Căn chỉnh ảnh mặt, tạo và chuẩn hóa embedding |
| PostgreSQL + pgvector | Lưu nhân viên và tìm kiếm theo cosine similarity |
| `IdentityCache` | Giảm số lần extract và truy vấn theo từng track |
| `IdentityManager` | Giữ danh tính đã xác nhận trong vòng đời person track |
| `EventProcessor` | Xác định IN/OUT bằng tâm bbox người |
| `VMSClient` | Nhận camera/vùng, gửi bbox, sự kiện và upload snapshot |

Mỗi camera có tracker, cache và identity manager riêng. Các model và verifier
được dùng chung, các camera được xử lý lần lượt. Danh tính giữa các track ID
hoặc camera khác nhau chưa được hợp nhất. Database nhân viên AI chưa tự đồng
bộ với danh sách nhân viên của backend VMS.

## Cấu trúc project

```text
main.py                         Pipeline VMS, không mở cửa sổ video
configs/config.yml              Model, inference, runtime
configs/demo_zones.json         Polygon/vạch cho demo local
src/core_ai/                    Reader, detector, tracker, identity, event
src/database/                   Truy vấn vector và xác minh danh tính
src/vms_service/                MQTT client và upload snapshot
scripts/enroll_employees.py     Đăng ký nhân viên từ CSV
scripts/doc_enroll.md           Hướng dẫn enrollment
sql/init.sql                    Schema PostgreSQL và extension vector
docker-compose.yml              PostgreSQL + pgvector
docker/db/README.md              Hướng dẫn database
experiments/detector_tensorrt/   Ghi chú và dependencies thử nghiệm TensorRT
tests/                          Kiểm thử offline
bao_cao_face/                   Mã nguồn báo cáo LaTeX
VMS_INTEGRATION.md               Chi tiết tích hợp MQTT và sự kiện
```

## Chuẩn bị môi trường và model

Chạy từ thư mục gốc trong môi trường Python đã cài các thư viện của project:
PyTorch phù hợp CUDA, OpenCV, Ultralytics, TensorRT, ONNX, ONNX Runtime,
InsightFace, NumPy, supervision, trackers, deep-sort-realtime, psycopg,
pgvector, PyYAML, python-dotenv, paho-mqtt 2.x và requests.

Project hiện chưa có file requirements tổng cho toàn pipeline. File
`experiments/detector_tensorrt/requirements.txt` chỉ phục vụ thí nghiệm.

Chuẩn bị các model riêng; weight, engine, ảnh nhân viên và dữ liệu database
không nằm trong repository. Engine TensorRT cần tương thích với GPU, CUDA và
runtime đang dùng; nên build trên môi trường triển khai.

| Chương trình | Detector mặt | Tracker người |
|---|---|---|
| `main.py` | SCRFD từ model pack InsightFace, mặc định `buffalo_s` | Deep SORT |
| Demo `camera_reader` | SCRFD TensorRT từ engine local | Deep SORT |
| Enrollment | SCRFD từ model pack InsightFace | Không sử dụng |

Model pack InsightFace cần có trong `~/.insightface/models` hoặc được tải ở
lần khởi tạo đầu. Main và enrollment chọn pack bằng `model_detect.model_name`;
main hỗ trợ ghi đè bằng `--detector-model`.

## Cấu hình

Tự tạo `.env` tại root bằng thông tin của môi trường triển khai. Không commit
file này. Các biến được sử dụng gồm:

- Database: `DATABASE_URL`, hoặc `PGHOST`, `PGPORT`, `POSTGRES_DB`,
  `POSTGRES_USER`, `POSTGRES_PASSWORD`.
- Docker Compose: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
  `POSTGRES_HOST_PORT`.
- MQTT: `MQTT_HOST`, `MQTT_PORT`, `MQTT_USERNAME`, `MQTT_PASSWORD`,
  `MQTT_CLIENT_ID`.
- Phạm vi VMS: `VMS_ZONE_ID`, `VMS_ORGANIZATION_ID`.
- Snapshot: `VMS_STORAGE_UPLOAD_URL`.

`MQTT_HOST` chỉ chứa hostname. Thông tin kết nối database phải khớp với
container. Python chạy trực tiếp trên Windows dùng localhost và port được
publish; `PGHOST=db` dành cho chương trình trong cùng mạng Compose.

Điều chỉnh `configs/config.yml` theo máy đang chạy. Ví dụ các mục chính:

```yaml
model_detect:
  model_name: buffalo_s

model_extract:
  weight: models/glintr100.onnx

inference:
  conf: 0.5
  imgsz: 640
  device: 0

person_tracking:
  weight: yolo11n.engine
  conf: 0.4
  imgsz: 640

runtime:
  processing_fps: 15
  recognition_threshold: 0.4
  recognition_interval: 0.5
  event_cooldown: 1.0
  track_ttl: 2.0
  min_face_size: 32
  boundary_margin: 0.003
```

Đây là ví dụ, không thay thế toàn bộ config. Config đang lưu một số đường dẫn
model tuyệt đối trên Windows; cần sửa khi chuyển máy. Đường dẫn tương đối
được tính từ root project. Code hiện đọc model embedding trực tiếp từ
`model_extract.weight`, không hỗ trợ ghi đè qua `FACE_EXTRACTOR_WEIGHTS`.

Các CLI hiện nhận `--device` dạng số nguyên, mặc định `0`; `--device cpu`
không được parser chấp nhận. Các mục `enable_cmc` và `lost_track_buffer` trong
config chưa được truyền vào tracker Deep SORT hiện tại. Tracker khởi tạo với
`max_age=15`, `nn_budget=50`, `max_cosine_distance=0.4`, embedder `mobilenet`.

## Khởi động database

```powershell
docker compose config --quiet
docker compose up -d db
docker compose ps
```

Compose dùng project `task2_atin`, container `face_recognition_pgvector` và
volume `face_recognition_pgdata`. `sql/init.sql` chỉ tự chạy khi volume rỗng;
chỉnh SQL không tự migrate database đã có. Database mới chưa có embedding.
Xem thêm [hướng dẫn database](docker/db/README.md).

## Đăng ký nhân viên

Tạo `configs/employees.csv` với các cột:

```csv
user_code,user_name,image_path
EMP001,Nguyen Van A,pics/employee_01.jpg
```

Đường dẫn ảnh tương đối được tính từ `--image-root`, mặc định là root project.

```powershell
python -m scripts.enroll_employees --check
python -m scripts.enroll_employees --device 0
```

Nếu ảnh ở ngoài project:

```powershell
python -m scripts.enroll_employees --image-root D:/employee-data --check
python -m scripts.enroll_employees --image-root D:/employee-data --device 0
```

`--check` chỉ kiểm tra CSV và file ảnh, không chạy model hay ghi database.
Enrollment yêu cầu đúng một khuôn mặt đủ lớn, căn chỉnh bằng 5 landmarks và
lưu embedding 512 chiều. Nếu không phát hiện mặt, script thử đệm viền 25%
rồi đưa tọa độ về ảnh gốc. Ảnh vẫn không hợp lệ sẽ bị bỏ qua và ghi log.

Chạy lại cùng mã nhân viên sẽ cập nhật tên và embedding, giữ các trường khác.
Các dòng thành công được commit riêng; script trả exit code 1 nếu có dòng lỗi.
Embedding trong database cần được tạo bằng cùng model và cách căn chỉnh với
pipeline. Xem [hướng dẫn enrollment](scripts/doc_enroll.md).

## Chạy demo local

Demo hiện dùng engine SCRFD được chỉ định trực tiếp tại
`D:\vms-combine-ai\det_500m_fp16.engine` trong `run_demo()`. Cần chuẩn bị engine
ở đường dẫn đó hoặc điều chỉnh đường dẫn trước khi chạy trên máy khác.
`--detector-model`/`--weights` vẫn có trong CLI nhưng chưa thay đổi engine
TensorRT đang được khởi tạo trong demo.

```powershell
python -m src.core_ai.camera_reader --rtsp "rtsp://your-camera/stream" --device 0 --debug
```

Demo vẽ bbox người, track ID, khuôn mặt, danh tính và hướng IN/OUT. Mặc định
đọc `configs/demo_zones.json`; đổi bằng `--zones path/to/zones.json`.
Nhấn Q/Esc để dừng. Demo cần database để so khớp, không kết nối/publish VMS.

## Chạy pipeline VMS

```powershell
python main.py --config configs/config.yml --device 0
```

Thêm `--debug` để xem log publish bbox. Main nhận camera bật FACE trong phạm
vi cấu hình, ưu tiên restream FACE rồi tới RTSP gốc. Camera không có URL sẽ
được bỏ qua; camera bị tắt/xóa/đổi URL sẽ được dừng. Reconnect reset tracking,
identity và lịch sử sự kiện, giữ nguyên model. Dừng bằng Ctrl+C.

| Chức năng | Topic | QoS |
|---|---|---|
| Xin camera | `smart_vms/cameras/getdevice` | 1 |
| Nhận camera | `smart_vms/cameras/zone/{zone_id}` | 1 |
| Nhận polygon/vạch FACE | `smart_vms/cameras/+/ai_zones/FACE` | 1 |
| Gửi sự kiện | `smart_vms/ai_events/FACE` | 1 |
| Gửi bbox | `smart_vms/ai/bbox/{camera_code}` | 0 |

BBox có `class: person`, topic/module vẫn là FACE. Snapshot upload thành công
trả về `object_key` để gửi trong `snapshot_url`; upload lỗi có thể gửi event
không ảnh. Người chưa nhận diện không được gán `user_code`.

Broker ACK chưa xác nhận backend đã lưu sự kiện. Cần kiểm tra bbox, lịch sử
và snapshot trên Web VMS để xác nhận toàn luồng.

## Quy tắc nhận diện và IN/OUT

- Ghép mặt với người theo vị trí, ưu tiên vùng trên của bbox; bỏ trường hợp
  nhập nhằng hoặc nhiều mặt thuộc cùng một người.
- Giữ danh tính đã xác nhận khi person track còn tồn tại dù không thấy mặt.
  Track ID mới cần xác nhận danh tính lại.
- Dùng tâm bbox người: ngoài → trong polygon là IN, trong → ngoài là OUT.
  Với vạch hữu hạn, đi sang phía `direction_point` là IN, ngược lại là OUT.
- Track mới chỉ ghi nhớ vị trí. Chỉ các detection hiện tại được dùng.
- Cooldown theo camera, vùng và track; lượt trong cooldown bị bỏ qua.
- Nhận diện mặt theo `recognition_interval`; frame có sự kiện sẽ xác minh lại
  nếu thấy mặt. Không thấy mặt thì dùng danh tính đã giữ trên person track.

## Kiểm thử và trạng thái hiện tại

```powershell
python -m unittest discover -s tests -v
```

Lần kiểm tra ngày 04/11/2026 trong môi trường `free_env`: 45 test, 33 đạt,
11 lỗi và 1 thất bại. Bộ test chưa được cập nhật đầy đủ cho API detector,
tracker và `process_frame` mới; một test landmarks cũng đang lỗi.
Chưa thể xem bộ test hiện tại là xác nhận pipeline hoạt động đầy đủ.

Main còn cần thống nhất khóa dữ liệu timing giữa `process_frame()`
(`timing_ms`) và vòng thu thập thống kê (`timings_ms`) trước khi chạy toàn
luồng. Demo còn phụ thuộc đường dẫn engine local như mô tả ở trên.

Kiểm thử offline không thay thế đánh giá camera thực. Chưa có số liệu độc lập
về độ chính xác nhận diện hoặc tốc độ toàn pipeline YOLO + Deep SORT + SCRFD.
Các benchmark detector TensorRT cũ chỉ phản ánh detector của thí nghiệm đó.
Notebook và dữ liệu benchmark local được Git bỏ qua, không kèm trong repo.

Các hạn chế khác: có thể lặp sự kiện khi track đổi ID, bỏ sót khi người đi
xuyên vùng giữa hai frame; upload/publish đang chặn vòng xử lý và chưa có
hàng đợi bền vững để gửi bù khi offline.
