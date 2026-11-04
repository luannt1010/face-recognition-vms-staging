import numpy as np
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import tensorrt as trt
from insightface.model_zoo.scrfd import SCRFD

class TensorRTSession:
    # Thứ tự output của det_500m.onnx, đã đối chiếu với engine.
    OUTPUT_NAMES = [
        "443", "468", "493",   # Scores: stride 8, 16, 32
        "446", "471", "496",   # Bbox
        "449", "474", "499",   # Landmarks
    ]

    def __init__(self, engine_path, device=0):
        self.device = torch.device(f"cuda:{int(device)}")
        self.logger = trt.Logger(trt.Logger.ERROR)

        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(
                Path(engine_path).read_bytes()
            )
            if self.engine is None:
                raise RuntimeError("Không load được TensorRT engine")

            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("Không tạo được execution context")

            self.stream = torch.cuda.Stream(device=self.device)

            self.metadata = {}
            self.buffers = {}
            self.dtypes = {}
            input_names = []

            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                shape = tuple(self.engine.get_tensor_shape(name))

                if any(dim <= 0 for dim in shape):
                    raise ValueError("Adapter này cần engine có shape cố định")

                dtype = np.dtype(
                    trt.nptype(self.engine.get_tensor_dtype(name))
                )
                torch_dtype = torch.from_numpy(
                    np.empty((), dtype=dtype)
                ).dtype

                self.metadata[name] = SimpleNamespace(
                    name=name,
                    shape=list(shape),
                )
                self.dtypes[name] = dtype
                self.buffers[name] = torch.empty(
                    shape,
                    dtype=torch_dtype,
                    device=self.device,
                )

                success = self.context.set_tensor_address(
                    name, self.buffers[name].data_ptr()
                )
                if not success:
                    raise RuntimeError(f"Không bind được tensor {name}")

                if (
                    self.engine.get_tensor_mode(name)
                    == trt.TensorIOMode.INPUT
                ):
                    input_names.append(name)

            if len(input_names) != 1:
                raise ValueError("Engine phải có đúng một input")

            self.input_name = input_names[0]

            if tuple(self.metadata[self.input_name].shape) != (
                1, 3, 640, 640
            ):
                raise ValueError("Engine phải nhận input 1x3x640x640")

            output_names = {
                name
                for name in self.metadata
                if self.engine.get_tensor_mode(name)
                == trt.TensorIOMode.OUTPUT
            }
            if output_names != set(self.OUTPUT_NAMES):
                raise ValueError("Tên output không khớp engine det_500m")

    def get_inputs(self):
        return [self.metadata[self.input_name]]

    def get_outputs(self):
        return [
            self.metadata[name]
            for name in self.OUTPUT_NAMES
        ]

    def run(self, output_names, input_feed):
        array = np.ascontiguousarray(
            input_feed[self.input_name],
            dtype=self.dtypes[self.input_name],
        )

        expected_shape = tuple(
            self.metadata[self.input_name].shape
        )
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"Input shape {array.shape}, cần {expected_shape}"
            )

        with torch.cuda.device(self.device):
            with torch.cuda.stream(self.stream):
                self.buffers[self.input_name].copy_(
                    torch.from_numpy(array)
                )

                success = self.context.execute_async_v3(
                    stream_handle=self.stream.cuda_stream
                )
                if not success:
                    raise RuntimeError("TensorRT inference thất bại")

            self.stream.synchronize()

            return [
                self.buffers[name].cpu().numpy()
                for name in output_names
            ]

class SCRFDTensorRTDetector:
    def __init__(
        self,
        engine_path,
        det_size=(640, 640),
        det_thresh=0.5,
        device=0,
    ):
        if tuple(det_size) != (640, 640):
            raise ValueError("Engine hiện tại yêu cầu det_size=(640, 640)")

        self.session = TensorRTSession(
            engine_path,
            device=device,
        )

        self.model = SCRFD(session=self.session)
        self.model.prepare(
            ctx_id=int(device),
            input_size=det_size,
            det_thresh=det_thresh,
        )

    def detect_many_faces(self, frame, max_num=0):
        boxes, landmarks = self.model.detect(
            frame,
            max_num=max_num,
        )

        if landmarks is None:
            raise RuntimeError("Model không cung cấp landmarks")

        return {
            "bbox": np.asarray(
                boxes[:, :4], dtype=np.float32
            ).reshape(-1, 4),
            "det_score": np.asarray(
                boxes[:, 4], dtype=np.float32
            ),
            "landmarks": np.asarray(
                landmarks, dtype=np.float32
            ).reshape(-1, 5, 2),
        }

    def detect_one_face(self, frame):
        return self.detect_many_faces(frame, max_num=1)

    def warmup(self):
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect_many_faces(frame)

    def align(self, frame, landmarks):
        from insightface.utils import face_align

        return face_align.norm_crop(
            frame,
            landmark=landmarks,
            image_size=112,
        )

class SCRFDDetector:
    def __init__(self, model_name:str="buffalo_s", det_size:tuple=(640, 640), det_thresh:float=0.5, device=0):
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis(name=model_name, allowed_modules=["detection"],
                                  providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.det_size = det_size
        self.model.prepare(ctx_id=device, det_thresh=det_thresh, det_size=self.det_size)

    def warmup(self):
        dummy = np.zeros((self.det_size[1], self.det_size[0], 3), dtype=np.uint8)
        self.model.get(dummy)

    def detect_many_faces(self, frame, max_num=0):
        faces = self.model.get(frame, max_num=max_num)
        if not faces:
            return {
                "bbox": np.empty((0, 4), dtype=np.float32),
                "det_score": np.empty((0,), dtype=np.float32),
                "landmarks": np.empty((0, 5, 2), dtype=np.float32)
            }

        bboxes, scores, landmarks = [], [], []
        for face in faces:
            bboxes.append(face.bbox)
            scores.append(face.det_score)
            landmarks.append(face.kps)
        return {"bbox": np.asarray(bboxes, dtype=np.float32).reshape(-1, 4),
                "det_score": np.asarray(scores, dtype=np.float32),
                "landmarks": np.asarray(landmarks, dtype=np.float32).reshape(-1, 5, 2)}


    def detect_one_face(self, frame):
        return self.detect_many_faces(frame, max_num=1)

    def align(self, frame, landmarks):
        from insightface.utils import face_align

        return face_align.norm_crop(frame, landmarks, image_size=112)


class InferenceWorker:
    def __init__(self, wp, conf, imgsz, classes, device=0):
        from ultralytics import YOLO

        self.model = YOLO(wp)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.classes = classes

    def warmup(self):
        self.predict([np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)])

    def predict(self, frames):
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

class PersonDetector:
    def __init__(self, model_path, conf, imgsz, device):
        self.model = InferenceWorker(
            wp=model_path,
            conf=conf,
            imgsz=imgsz,
            classes=[0],
            device=device
        )
    def warmup(self):
        self.model.warmup()
    def detect(self, frame):
        return self.model.predict([frame])[0] #Lấy frame mới nhất