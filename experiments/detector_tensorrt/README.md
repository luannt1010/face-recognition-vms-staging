# Thử nghiệm detector YOLO11n với TensorRT

Mở `detector_quantization_benchmark.ipynb`, chọn kernel có PyTorch CUDA và chạy
các cell từ trên xuống. Notebook mặc định dùng RTX 4050 hiện tại, checkpoint
train-4 của project, input 640, batch 1.

Cài các thư viện còn thiếu trong đúng môi trường Python:

```powershell
python -m pip install -r experiments/detector_tensorrt/requirements.txt
python -m pip install "tensorrt>=10,<11"
python -m jupyter lab experiments/detector_tensorrt/detector_quantization_benchmark.ipynb
```

Giữ bản PyTorch CUDA đang chạy được, không cài lại torch CPU. Môi trường
`free_env` đã được kiểm tra truy cập GPU; có thể dùng môi trường YOLO khác.
Chưa có TensorRT thì notebook chỉ đo PyTorch và báo bỏ qua các engine.

Notebook so sánh PyTorch FP32, TensorRT FP32 mặc định, FP16 và INT8. ONNX là
định dạng trung gian khi export. File gốc và cấu hình pipeline không bị sửa.
Mỗi engine dùng checkpoint copy và thư mục riêng, không ghi đè lẫn nhau.

Calibration INT8: lấy ngẫu nhiên cố định 500 ảnh từ train. Timing: 24 ảnh test
nạp sẵn, 20 warmup, 3 vòng x 100 lần đo. Đồng bộ CUDA và lấy kết quả về CPU.
Validation Precision/Recall/mAP dùng toàn bộ test có nhãn, tách khỏi calibration.
Đường dẫn dataset hiện trỏ tới `D:/private/task2_atin/data`; sửa ở cell cấu hình
nếu chuyển dữ liệu.

Kết quả nằm trong `outputs/img640/`: CSV thời gian thô và tổng hợp, metadata,
đồ thị, ảnh so sánh, engine và manifest. `comparison.csv` chỉ có khi chạy
validation. Số FPS là detector, chưa bao gồm camera/extractor/database/MQTT.

Không chọn INT8 chỉ dựa vào tốc độ. Cần kiểm tra Recall/mAP và mặt nhỏ trên ảnh
test; FP16 có thể phù hợp hơn. Engine phải build/đo trên GPU đích; file `.engine`
không có tính di động như checkpoint `.pt`.

API export được đối chiếu với
[Ultralytics TensorRT](https://docs.ultralytics.com/integrations/tensorrt/) và
[NVIDIA support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html).
Notebook hỗ trợ cả `quantize` mới và `half/int8` cũ. TensorRT 10.x được chọn
để dùng đường calibration cũ; TensorRT 11 cần Ultralytics hỗ trợ ModelOpt.

## Baseline đã chạy ngày 30/10/2026

Đã chạy các cell chuẩn bị dữ liệu và benchmark PyTorch trên RTX 4050 Laptop,
PyTorch 2.13.0+cu126, Ultralytics 8.4.123, 640x640, batch 1, 24 ảnh test,
20 warmup và 300 lần đo. Predict trung bình 14.076 ms/ảnh, p95 14.924 ms/ảnh,
tương đương 71.04 FPS detector; forward do Ultralytics báo trung bình 10.146 ms.

Chưa chạy TensorRT/INT8 vì môi trường thử chưa có TensorRT; validation mAP cũng chưa chạy.
Các CSV/đồ thị hiện có chỉ là baseline, không chứng minh quantization nhanh hơn.
Notebook chưa lưu execution output; có thể mở và chạy lại từ đầu sau khi cài runtime.
