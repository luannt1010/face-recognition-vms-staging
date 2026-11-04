# Nhận diện mặt: event processor và MQTT

Đọc `src/core_ai/event_processor.py` trước, sau đó `src/vms_service/vms_client.py`.
Hai class độc lập: processor xác định hướng, client nhận cấu hình và gửi dữ liệu.
Phần kết nối dùng callback API v2 của
[Paho MQTT](https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html).

## Chạy pipeline main

Để thử local có cửa sổ hiển thị, chạy từ root project:

```powershell
python -m src.core_ai.camera_reader --rtsp "rtsp://your-camera/stream" --debug
```

Demo không kết nối MQTT. Mặc định vẽ polygon trong `configs/demo_zones.json`:
tâm bbox người đi vào là IN, đi ra là OUT. Terminal in `[IN]`/`[OUT]` kèm track ID,
tên và điểm so khớp mỗi lần qua biên. Chấm nhỏ trong bbox là tâm đang xét.
Nhấn Q/Esc hoặc đóng cửa sổ để thoát. Thêm `--device cpu`
nếu muốn chạy detector bằng CPU (extractor tự chọn provider có sẵn).
Thêm `--zones configs/demo_zones.json` để đọc file JSON chứa `ai_zones` theo
§6.2, hoặc mảng zone trực tiếp. Đây là file cấu hình local, không tự cập nhật từ VMS.

Hàm `run_demo(line=..., rtsp_url=...)` vẫn nhận hai điểm vạch chuẩn hoá như demo
cũ. Với vạch tự truyền, hướng IN nằm bên trái vector từ điểm đầu tới điểm cuối,
được vẽ bằng mũi tên. Có thể truyền `zones=[...]` với `direction_point` cụ thể.

Để chạy đầy đủ với MQTT:

Màu mỗi detection được gửi qua `detections[].color`: xanh `#22c55e` cho người
đã khớp, đỏ `#ef4444` cho người lạ. Frontend VMS cần đọc field này để áp màu.
Log `VMS config` cho biết số vùng FACE nhận được; log `pipeline` mỗi 2 giây
cho biết số detection, track, mặt hợp lệ, mặt khớp, vùng hoạt động và event.
Publish bbox thành công chuyển sang mức DEBUG để dễ đọc log sự kiện FACE.
Dùng `python main.py --debug` nếu muốn xem từng lần publish bbox.

Nếu main chờ mãi và log `cameras_FACE=[]`, xem `VMS camera scope` để biết đúng
topic, `zone_id` và `organization_id` đang dùng. `received=0` nghĩa là VMS trả
danh sách camera rỗng cho scope này; `received>0` nhưng `cameras_FACE=[]` nghĩa
là các camera trả về chưa có module `FACE`. Log `camera_modules` cho biết mã
camera và module hiện tại, không in URL RTSP.
Subscription `smart_vms/cameras/+/ai_zones/FACE` có thể nhận retained vùng của
camera ngoài zone đang chọn. Vì vậy `zones_FACE=1` không có nghĩa main được
cấu hình chạy camera đó; `camera_selected=False` cho biết camera chưa nằm
trong danh sách FACE đã chọn (cũng có thể vì message danh sách chưa đến).
Đặt `VMS_ZONE_ID` và `VMS_ORGANIZATION_ID` đúng phạm vi triển khai trước khi
khởi động lại; camera cần thuộc zone đó và được bật module FACE trên VMS.

Lịch sử FACE chỉ xuất hiện khi một track đã có vị trí trước đó rồi đi qua biên
polygon/vạch. Không có vùng FACE, xuất hiện sẵn trong vùng, hoặc chỉ đứng trong
vùng đều chưa tạo sự kiện. `success=True` của bbox QoS 0 chỉ là Paho nhận lệnh
gửi, không xác nhận frontend đã nhận/vẽ và không tạo lịch sử FACE.

```powershell
python main.py --config configs/config.yml
```

Chạy bằng môi trường Python đã cài các thư viện AI của project (OpenCV,
Ultralytics, ONNX/ONNX Runtime, NumPy, supervision, trackers, psycopg, pgvector),
cùng PyYAML và các thư viện MQTT/HTTP bên dưới. Dừng bằng Ctrl+C.

Luồng chính nằm trong `main()`, còn `process_frame()` trình bày rõ từng bước:
YOLO + BoT-SORT → SCRFD → ghép mặt/người → xác minh mặt và giữ tên → publish.
Event xét tâm bbox người; có sự kiện thì xác minh lại mặt nếu thấy mặt.
Reader lấy frame mới nhất.
Mỗi camera dùng tracker, cache và identity manager riêng; SCRFD/YOLO/extractor/verifier dùng chung và xử lý
lần lượt. `processing_fps` là giới hạn trên, tốc độ thực còn phụ thuộc model,
database và thời gian upload/publish.

Main tự nhận camera bật FACE từ VMS, bỏ camera ảo không có RTSP, dừng camera
khi bị tắt/xóa/đổi URL. Camera reconnect sẽ reset tracker, cache, identity manager
và event processor, không nạp lại model. `min_face_size` chỉ giới hạn nhận diện mặt;
người không thấy mặt hoặc mặt quá nhỏ vẫn có bbox và có thể sinh IN/OUT.

Identity được cache riêng theo từng camera và track_id trong
`src/core_ai/identity_cache.py`. Track mới nhận diện ngay; track đang thấy được
làm mới mỗi `runtime.recognition_interval` (mặc định 0.5 giây). Trong khoảng đó
không extract/search DB lại. Track mất detection quá `track_ttl` bị xóa khỏi
cache; reconnect/reset tracker cũng xóa cache. Cả người quen và người lạ đều
được làm mới định kỳ, nên người lạ không bị giữ mãi trong trạng thái đó.

Nếu frame có IN/OUT và có mặt ghép được, identity được xác minh lại trên frame đó.
Không thấy mặt thì dùng tên đã giữ trên cùng person track; track mới chưa có mặt
được gửi như người chưa xác định. Extractor vẫn xử lý từng mặt, không batch. Cache không lưu ảnh/embedding
và không sửa cách so khớp, ngưỡng hay dữ liệu PostgreSQL.

Có thể đặt `recognition_interval: 0` để quay về nhận diện mọi frame. Kết quả
overlay có thể giữ tên cũ tối đa một chu kỳ khi tracker gán nhầm ID; cache không
giải quyết lỗi nhận diện hay lỗi tracker.

Mặc định: SCRFD (`model_detect.model_name`, `buffalo_s`) + InsightFaceExtractor +
PersonTracker (YOLO COCO class 0 + BoT-SORT). `person_tracking.weight` chọn model
YOLO; mỗi camera có BoT-SORT riêng, chưa dùng ReID/OSNet. BBox gửi với `class: person`
trên topic FACE hiện có.
Model embedding đọc từ config; có thể ghi đè bằng `FACE_EXTRACTOR_WEIGHTS`.
`--device cpu` chọn CPU; `--detector-model` đổi model pack SCRFD.
Không gửi `person_id` local lên VMS vì hai database có thể dùng ID khác nhau.

Khai báo `VMS_STORAGE_UPLOAD_URL` để sự kiện có snapshot. Upload lỗi thì log và
gửi event không ảnh. MQTT mất kết nối thì dừng xử lý camera ở vòng tiếp theo;
chương trình không lưu sự kiện offline để gửi bù.

## Quy tắc IN/OUT

- Dùng **tâm bbox khuôn mặt**, không dùng chân người vì tracker hiện theo dõi mặt.
- Polygon: ngoài → trong là IN, trong → ngoài là OUT. Đây là quy ước đã thống nhất,
  tài liệu VMS không quy định hướng cho polygon.
- Line: cắt đoạn vạch sang phía `direction_point` là IN, ngược lại là OUT.
  Tên/code line không quyết định hướng. Processor cũng đọc được field `point` ở §6.3.
- Track mới xuất hiện chỉ ghi nhớ vị trí, chưa phát sự kiện.
- Chỉ nhận `is_detection=True` từ FaceTracker, bỏ bbox dự đoán.
- Dải biên mặc định 0.003 trên toạ độ chuẩn hoá giúp giảm rung; chưa phải bộ lọc
  hoàn toàn cho mọi trường hợp rung mạnh.
- Cooldown theo từng camera + zone + track, mặc định 1 giây. Chuyển hướng trong
  cooldown bị bỏ, không gửi trễ sau đó. Giảm về 0 nếu muốn giữ mọi lần qua vạch.
- Sau `track_ttl` không thấy detection (mặc định 2 giây), quên track đó.
- Sửa vùng tự xóa lịch sử; khi reconnect RTSP/tạo lại tracker, gọi `reset(code)`.
- Polygon chỉ so sánh trong/ngoài giữa các frame: hai frame đều ở ngoài nhưng
  nhảy qua toàn bộ vùng sẽ không sinh sự kiện. Tăng processing FPS nếu cần.

## Topic

| Chức năng | Topic | QoS |
|---|---|---|
| Xin camera | `smart_vms/cameras/getdevice` | 1 |
| Nhận camera | `smart_vms/cameras/zone/51` | 1 |
| Nhận vùng/vạch FACE | `smart_vms/cameras/+/ai_zones/FACE` | 1 |
| Gửi sự kiện | `smart_vms/ai_events/FACE` | 1 |
| Gửi bbox | `smart_vms/ai/bbox/{camera_code}` | 0 |

Theo §6.2, `ai_zones/FACE` đã chứa cả polygon và line crossing nên client chỉ
subscribe topic đó, không nhận thêm bản sao từ `/FACE/lines`.
Payload rỗng xóa vùng. Danh sách camera mới thay thế danh sách cũ; camera không
bật FACE không được publish. `get_cameras()` trả danh sách camera đang bật FACE.
Khi reconnect MQTT, client đăng ký lại và xin camera lại, cache cũ bị xóa.

## Cấu hình

Cài thêm thư viện nếu môi trường chưa có:

```powershell
python -m pip install 'paho-mqtt>=2,<3' python-dotenv requests
```

Thêm các biến sau vào `.env` hiện có, điền tài khoản từ VMS:

```dotenv
MQTT_HOST=your-broker-domain
MQTT_PORT=443
MQTT_USERNAME=your-user
MQTT_PASSWORD=your-password
MQTT_CLIENT_ID=tts_luan_01
VMS_ZONE_ID=51
VMS_ORGANIZATION_ID=103
VMS_STORAGE_UPLOAD_URL=https://your-domain/storage/upload/file
```

Host chỉ là domain, không có `wss://` hay `/mqtt`. Không đưa `.env` lên Git.

## Ghép vào vòng xử lý hiện có

Đoạn dưới minh họa bên trong vòng xử lý một frame, sau khi đã chạy detector,
tracker và verifier. Nó không phải chương trình main hoàn chỉnh.

```python
from src.core_ai.event_processor import EventProcessor
from src.vms_service.vms_client import VMSClient

processor = EventProcessor(cooldown=1, track_ttl=2)
vms = VMSClient.from_env()
vms.start()

try:
    # Với mỗi frame của camera đang có trong vms.get_cameras():
    # tracks = tracker.update(detector.predict([frame])[0])
    # identities = {track_id: kết quả verifier của mặt trên frame này}
    zones = vms.get_zones(camera_code)
    events = processor.process(camera_code, tracks, frame.shape, zones)
    vms.publish_bbox(camera_code, tracks, frame.shape)
    for event in events:
        identity = identities[event["track_id"]]
        vms.publish_face(camera_code, event, identity, event_time=sample["captured_at"])
finally:
    vms.stop()
```

Để đính kèm ảnh: encode JPEG bằng `cv2.imencode`, gọi
`vms.upload_snapshot(camera_code, jpeg_bytes, upload_url)`, rồi truyền object_key
nhận được vào `snapshot_url` của `publish_face`. Không gửi ảnh base64 qua MQTT.
Nếu upload lỗi, bắt lỗi ở vòng xử lý và có thể gửi event không ảnh theo §4.

Vòng chạy camera phải đọc lại `get_cameras()` thường xuyên, dừng reader và gọi
`processor.reset(code)` khi camera bị xóa/tắt FACE/mất kết nối. Khi camera reader
đổi `sample['generation']`, reset processor và tạo lại tracker. Mỗi camera dùng
một tracker riêng. Nhận diện trên ảnh gốc trước khi vẽ bbox; không ghép identity
bằng thứ tự trong list, dùng track_id.

`publish_face()` trả True khi broker ACK; VMS không trả ACK lưu database. Khi
timeout, trạng thái nhận có thể chưa rõ. Client không bổ sung hàng đợi bền vững
hay retry cấp ứng dụng. Backend cũng dedup FACE theo camera/danh tính trong 1s,
nên hai lượt quá gần nhau có thể bị backend gộp (§5.2).

Chạy kiểm thử offline (không kết nối staging):

```powershell
python -m unittest discover -s tests -v
```
