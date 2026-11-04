import onnx
import cv2
import numpy as np
import onnxruntime as ort
from src.utils import l2_normalize
from insightface.model_zoo import get_model
from insightface.utils import face_align

class InsightFaceExtractor:
    def __init__(self, model_path, device=0):
        ctx_id = -1 if str(device).lower() == 'cpu' else int(device)
        providers = ['CPUExecutionProvider'] if ctx_id < 0 else [
            ('CUDAExecutionProvider', {'device_id': ctx_id}), 'CPUExecutionProvider']
        self.model = get_model(model_path, providers=providers)
        self.model.prepare(ctx_id=ctx_id)

    def warmup(self):
        self.extract(np.zeros((112, 112, 3), dtype=np.uint8))

    def extract(self, frame, kps=None, align=False):
        if align:
            if kps is None:
                raise ValueError("lack kps args")
            frame = face_align.norm_crop(frame, landmark=kps, image_size=112)
        embedding = self.model.get_feat(frame).flatten()
        return l2_normalize(embedding)

class Extractor:
    def __init__(self, wp):
        self.wp = wp
        self.session = None
        self.input_name = None
        self.output_name = None

    def _load_model(self):
        if self.session is not None:
            return
        try:
            model = onnx.load(self.wp)
            onnx.checker.check_model(model)
            self.session = ort.InferenceSession(self.wp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Load ONNX model error: {e}")

    def warmup(self):
        self._load_model()
        dumppy_input = np.random.randn(1, 3, 112, 112).astype(np.float32)
        self.session.run([self.output_name], {self.input_name: dumppy_input})

    def _preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (112, 112))
        image = image.astype(np.float32) / 255.0
        mean, std = np.asarray([0.485, 0.456, 0.406], dtype=np.float32), np.asarray([0.229, 0.224, 0.225],
                                                                                    dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return np.expand_dims(image, axis=0)

    def extract(self, frame):
        self._load_model()
        img = self._preprocess(frame)
        embedding = self.session.run([self.output_name], {self.input_name: img})[0]
        return l2_normalize(embedding).squeeze(0)


if __name__ == "__main__":
    import time
    from src.database.repository import PGVectorStore
    wp = r"D:\private\face-recognition-system\checkpoints\final5\checkpoints\face_embedding_model.onnx"
    extractor = Extractor(wp)
    store = PGVectorStore()
    img_path = r"D:\private\task2_atin\pics\hoangnv.jpg"
    frame = cv2.imread(img_path)
    start = time.perf_counter()
    embedding = extractor.extract(frame)
    end = time.perf_counter()
    print(embedding.shape)
    print((end - start) / 60)
    print(embedding.shape)
    print(embedding.dtype)

    res = store.search_candidates(embedding, limit=3)
    print(res)
