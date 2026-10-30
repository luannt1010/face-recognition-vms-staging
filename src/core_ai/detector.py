import numpy as np
from ultralytics import YOLO

class SCRFDDetector:
    def __init__(self):
        from insightface.app import FaceAnalysis

        self.model = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], 
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.det_size = (640, 640)
        self.model.prepare(ctx_id=0, det_thresh=0.5, det_size=self.det_size)

    def warmup(self):
        dummy = np.zeros((self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
        self.model.get(dummy)

    def detect_many_faces(self, frame):
        faces = self.model.get(frame)
        if not faces:
            return {"bbox": np.empty((0, 4), dtype=np.float32),
                    "det_score": np.empty((0,), dtype=np.float32)}
        bboxes, scores = [], []
        for face in faces:
            bboxes.append(face["bbox"])
            scores.append(face["det_score"])
        return {"bbox": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
                "det_score": np.asarray(scores, dtype=np.float32)}


    def detect_one_face(self, frame):
        face = self.model.get(frame, max_num=1)
        if not face:
            return {"bbox": np.empty((0, 4), dtype=np.float32),
                    "det_score": np.empty((0,), dtype=np.float32)}
        return {"bbox": np.asarray([item["bbox"] for item in face], dtype=np.float32).reshape(-1, 4),
                "det_score": np.asarray([item["det_score"] for item in face], dtype=np.float32)}

    def align(self, frame, landmarks):
        from insightface.utils import face_align

        return face_align.norm_crop(frame, landmarks, image_size=112)


# class InferenceWorkerOnnx:
#     def __init__(self, wp, conf, imgsz, classes, device=0):
#         self.wp = wp
#         self.conf = conf
#         self.imgsz = imgsz
#         self.device = device
#         self.classes = classes
#
#         self.session = None
#         self.input_name = None
#         self.output_name = None
#
#     def _load_model(self):
#         if self.session is not None:
#             return
#         try:
#             self.session = ort.InferenceSession(self.wp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#             self.input_name = self.session.get_inputs()[0].name
#             self.output_name = self.session.get_outputs()[0].name
#         except Exception as e:
#             raise RuntimeError(f"Load ONNX model error: {e}")
#
#     def _preprocess(self, frame):
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image = cv2.resize(image, (640, 640))
#         image = image.astype(np.float32) / 255.0
#         image = np.transpose(image, (2, 0, 1))
#         image = np.expand_dims(image, axis=0)
#         return image
#
#     def predict(self, frames: list):
#         self._load_model()
#         outputs = []
#         if not frames:
#             return outputs
#         for frame in frames:
#             frame = self._preprocess(frame)
#             output = self.session.run([self.output_name], {self.input_name: frame})
#             boxes, classes, scores, points = yolov5_facepostprocess(output, self.conf, 0.45)
#             outputs.append({
#                 "bbox": boxes,
#                 "det_score": scores
#             })
#         return outputs

class InferenceWorker:
    def __init__(self, wp, conf, imgsz, classes, device=0):
        self.model = YOLO(wp)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.classes = classes

    def predict(self, frames):
        """Hàm predict theo batch"""
        output = []
        if not frames:
            return output
        results = self.model.predict(
            source=frames,
            conf=self.conf,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )

        if len(results) != len(frames):
            raise RuntimeError("Inference result count does not match input batch")

        for result in results:
            bbox_lst = result.boxes.xyxy.cpu().numpy().astype(np.float32)
            score_lst = result.boxes.conf.cpu().numpy()

            output.append({
                "bbox": bbox_lst,
                "det_score": score_lst
            })
        return output


if __name__ == "__main__":
    import cv2
    from PIL import Image
    img_path = r"C:\Users\luann\Downloads\images.jpg"
    img_arr = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    detector = SCRFDDetector()
    detector.warmup()
    results = detector.detect_one_face(img_arr)
    # landmarks = results[0]["kps"]
    # res = detector.align(img_arr, landmarks)
    # img = Image.fromarray(res).convert("RGB")
    # img.show()
    print(results)

# if __name__ == "__main__":
#     import cv2
#     import time
#     from pathlib import Path
#
#     # Cấu hình
#     worker = InferenceWorker(
#         wp=r"D:\private\task2_atin\src\detector\runs\detect\train-4\weights\best.pt",
#         conf=0.3,
#         imgsz=640,
#         classes=None,
#         device=0
#     )
#
#     # Đọc ảnh test
#     image_paths = list(Path(r"D:\private\task2_atin\pics").glob("*"))
#     frames = []
#     valid_paths = []
#
#     for path in image_paths:
#         frame = cv2.imread(str(path))
#         if frame is not None:
#             frames.append(frame)
#             valid_paths.append(path)
#
#     if not frames:
#         print("Không tìm thấy ảnh trong thư mục test_images!")
#         exit()
#
#     # Chạy inference theo batch
#     start = time.perf_counter()
#     outputs = worker.predict(frames)
#     print(outputs)
#     elapsed = time.perf_counter() - start
#
#     print(f"Batch size: {len(frames)}")
#     print(f"Inference time: {elapsed:.4f}s")
#     print(f"Throughput: {len(frames) / elapsed:.2f} images/s")
#
#     # Hiển thị kết quả
#     for path, output in zip(valid_paths, outputs):
#         frame = cv2.imread(str(path))
#
#         bboxes = output["bbox"]
#         scores = output["det_score"]
#         if bboxes is None:
#             print(bboxes)
#             break
#         for bbox, score in zip(bboxes, scores):
#             x1, y1, x2, y2 = map(int, bbox)
#
#             cv2.rectangle(
#                 frame, (x1, y1), (x2, y2), (0, 255, 0), 2
#             )
#
#             label = f"{score:.2f}"
#             cv2.putText(
#                 frame, label, (x1, max(y1 - 10, 0)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                 (0, 255, 0), 2
#             )
#
#         print(f"{path.name}: {len(bboxes)} detections")
#
#         cv2.imshow(path.name, frame)
#
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
