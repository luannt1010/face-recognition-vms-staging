import cv2
import time
import logging
import threading
from queue import Queue, Empty, Full
import time
from datetime import datetime, timezone
import numpy as np


logger = logging.getLogger(__name__)


class CameraReader:
    def __init__(self, cam_id, rtsp_url):
        self.cam_id = cam_id
        self.rtsp_url = rtsp_url
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.running = False
        self.frames = Queue(maxsize=1)
        self.generation = 0
        self.cap = None
        self.thread = None

    def get_sample(self):
        with self.lock:
            try:
                return self.frames.get_nowait()
            except Empty:
                return None

    def get_frame(self):
        sample = self.get_sample()
        return sample["frame"] if sample is not None else None

    def _put_latest_frame(self, frame: np.ndarray):
        sample = {
            "frame": frame,
            # Đây là thời điểm máy AI đọc được ảnh, không phải timestamp camera.
            "captured_at": datetime.now(timezone.utc),
            "captured_monotonic": time.monotonic(),
            "generation": self.generation,
        }
        with self.lock:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            try:
                self.frames.put_nowait(sample)
            except Full:
                pass

    def _clear_frames(self):
        with self.lock:
            try:
                self.frames.get_nowait()
            except Empty:
                pass

    def _init_cap(self):
        if self.cap is not None:
            return True
        # Timeout là thuộc tính open-only của backend FFmpeg.
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            return False
        self.generation += 1
        logger.info("Camera %s connected (generation=%s)", self.cam_id, self.generation)
        return True

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self._clear_frames()
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(
            target=self._update_frame, name=f"camera-{self.cam_id}", daemon=True
        )
        self.thread.start()

    def _update_frame(self):
        try:
            while not self.stop_event.is_set():
                try:
                    opened = self._init_cap()
                    ret, frame = self.cap.read() if opened else (False, None)
                except cv2.error:
                    # Không ghi exception/URL vì có thể chứa user và password RTSP.
                    logger.warning("Camera %s could not read stream", self.cam_id)
                    ret, frame = False, None
                if self.stop_event.is_set():
                    break
                if not ret:
                    self._clear_frames()
                    if self.cap is not None:
                        self.cap.release()
                        self.cap = None
                    self.stop_event.wait(1)
                    continue
                self._put_latest_frame(frame)
        finally:
            # Chỉ reader thread sở hữu/release cap; stop() không đua với read().
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.running = False
            logger.info("Camera %s stopped", self.cam_id)

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                logger.warning("Camera %s is still closing its stream", self.cam_id)
        self._clear_frames()

def run_demo(line=None, *, rtsp_url, weights=None, device=None, debug=False,
             zones=None, config_path=None):
    """Demo local: YOLO + BoT-SORT theo người, SCRFD + InsightFace gắn tên.

    Mặc định đọc polygon từ configs/demo_zones.json.
    line: [[x1,y1], [x2,y2]] chuẩn hoá 0..1, IN theo pháp tuyến trái a -> b.
    zones: list polygon/line theo định dạng ai_zones của VMS (ưu tiên hơn line).
    weights: tên bộ model SCRFD của InsightFace, mặc định buffalo_s.
    Nhấn Q hoặc Esc để thoát.
    """
    import os
    from pathlib import Path
    import unicodedata
    from dotenv import load_dotenv
    from main import load_config, model_path, recognize_people
    from src.core_ai.detector import SCRFDDetector, PersonDetector, SCRFDTensorRTDetector
    from src.core_ai.event_processor import EventProcessor
    from src.core_ai.identity_cache import IdentityCache
    from src.core_ai.identity_manager import IdentityManager
    from src.core_ai.extractor import InsightFaceExtractor
    from src.core_ai.tracker import PersonTrackerBoTSORT, PersonTrackerDeepSort
    from src.database.verifier import Verifier

    root = Path(__file__).resolve().parents[2]
    load_dotenv(root / '.env')
    config = load_config(config_path or root / 'configs/config.yml')
    runtime = config['runtime']
    fps = float(runtime.get('processing_fps', 10))
    if fps <= 0:
        raise ValueError('processing_fps phải > 0')
    min_face_size = int(runtime.get('min_face_size', 32))
    if min_face_size <= 0:
        raise ValueError('min_face_size phải > 0')

    if zones is None and line is None:
        import json
        data = json.loads((root / 'configs/demo_zones.json').read_text(encoding='utf-8-sig'))
        zones = data['ai_zones']

    if zones is None:
        # Vạch demo ngang, đi từ phía trên xuống dưới là IN.
        line = line if line is not None else [[.2, .5], [.8, .5]]
        a, b = line
        # Chọn điểm IN theo pháp tuyến bên trái của vector a -> b.
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = np.hypot(dx, dy)
        if length == 0:
            raise ValueError('Hai điểm của line phải khác nhau')
        midpoint = [(a[0]+b[0])/2, (a[1]+b[1])/2]
        offset = .05
        direction = [midpoint[0]-dy/length*offset, midpoint[1]+dx/length*offset]
        if not all(0 <= value <= 1 for value in direction):
            raise ValueError('Line sát biên ảnh; truyền zones với direction_point cụ thể')
        zones = [{'id': 'demo-line', 'name': 'Demo line', 'ai_modules': 'FACE',
                  'shape_type': 'line_crossing',
                  'line': {'points': [{'x': a[0], 'y': a[1]}, {'x': b[0], 'y': b[1]}]},
                  'direction_point': {'x': direction[0], 'y': direction[1]}}]

    inference = config.get('inference', {})
    device = device if device is not None else inference.get('device', 0)
    imgsz = inference.get('imgsz', 640)
    det_size = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
    # detector = SCRFDDetector(model_name=weights or config.get('model_detect', {}).get('model_name', 'buffalo_s'),
    #                          det_size=det_size,
    #                          det_thresh=inference.get('conf', 0.5), device=device)
    detector = SCRFDTensorRTDetector(engine_path=r"D:\vms-combine-ai\det_500m_fp16.engine",
                             det_size=det_size,
                             det_thresh=inference.get('conf', 0.5), device=device)
    extractor = InsightFaceExtractor(model_path(config['model_extract']['weight']), device=device)
    detector.warmup()
    extractor.warmup()
    processor = EventProcessor(cooldown=runtime.get('event_cooldown', 1),
                               track_ttl=runtime.get('track_ttl', 2),
                               boundary_margin=runtime.get('boundary_margin', .003))
    person_config = dict(config.get('person_tracking', {}))
    person_weights = model_path(person_config.pop('weight', 'models/detector/yolo11n.pt'))
    person_detector = PersonDetector(model_path=person_weights, conf=person_config["conf"], imgsz=person_config["imgsz"], device=device)
    # tracker = PersonTracker(lost_track_buffer=person_config.get('lost_track_buffer', 30))
    tracker = PersonTrackerDeepSort()
    identity_cache = IdentityCache(interval=runtime.get('recognition_interval', .5),
                                    ttl=processor.track_ttl)
    identity_manager = IdentityManager(ttl=processor.track_ttl)
    camera = CameraReader(cam_id='demo', rtsp_url=rtsp_url)
    verifier = None
    generation, last_frame = None, None
    labels = {}  # track_id -> (direction, thời điểm hết hạn hiển thị)
    next_debug = 0.0
    window = 'Face demo - Q / Esc to exit'

    try:
        verifier = Verifier(threshold=runtime['recognition_threshold'])
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        camera.start()
        logger.info('Demo local đang chạy. Nhấn Q hoặc Esc để thoát.')
        timing_samples = []
        timing_window_started = time.monotonic()
        while True:
            loop_started = time.perf_counter()
            started = time.monotonic()
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            sample = camera.get_sample()
            if sample is None:
                time.sleep(.01)
                continue
            captured = sample['captured_monotonic']
            if started - captured > processor.track_ttl:
                continue
            if (sample['generation'] != generation
                    or (last_frame is not None and captured-last_frame > processor.track_ttl)):
                tracker.reset()
                processor.reset('demo')
                identity_cache.clear()
                identity_manager.clear()
                labels.clear()
            generation, last_frame = sample['generation'], captured
            frame = sample['frame']

            t0 = time.perf_counter()
            # Track chính theo người. Mặt chỉ cung cấp tên cho person track.
            person_result = person_detector.detect(frame)
            t1 = time.perf_counter()
            tracks = tracker.update(person_result, frame)
            alive_track_ids = tracker.get_track_ids()
            t2 = time.perf_counter()
            detections = detector.detect_many_faces(frame)
            t3 = time.perf_counter()
            events = processor.process('demo', tracks, frame.shape, zones, now=captured)
            t4 = time.perf_counter()
            _, identities = recognize_people(
                frame, tracks, detections, extractor, verifier, min_face_size,
                identity_manager=identity_manager, identity_cache=identity_cache, now=captured,
                force_ids={event['track_id'] for event in events}, debug=debug, alive_track_ids=alive_track_ids)
            t5 = time.perf_counter()
            now = time.monotonic()
            labels = {key: value for key, value in labels.items() if value[1] > now}
            for event in events:
                track_id = event['track_id']
                identity = identities[track_id]
                labels[track_id] = (event['direction'], now+2)
                name = identity.get('user_name') or 'Nguoi la'
                print(f"[{event['direction']}] track={track_id} name={name} "
                      f"zone={event['zone_id']} score={identity.get('score_match')}", flush=True)
            if debug and now >= next_debug:
                logger.info('faces=%s person_tracks=%s identities=%s',
                            len(detections['bbox']), len(tracks), identities)
                next_debug = now+1

            # Vẽ sau khi nhận diện để ảnh extract không dính chữ và khung.
            display = frame.copy()
            height, width = frame.shape[:2]
            for zone in zones:
                if not zone.get('is_active', True) or zone.get('ai_modules', 'FACE') != 'FACE':
                    continue
                is_line = 'line' in zone
                points = zone['line']['points'] if is_line else zone['points']
                pixels = np.array([[round(p['x']*width), round(p['y']*height)]
                                   for p in points], dtype=np.int32)
                cv2.polylines(display, [pixels], not is_line, (0, 255, 255), 2)
                if not is_line:
                    anchor = (int(pixels[0][0]), max(20, int(pixels[0][1])-10))
                    cv2.putText(display, 'Polygon: enter IN / leave OUT', anchor,
                                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
                if is_line:
                    point = zone.get('direction_point') or zone.get('point')
                    midpoint = tuple(np.mean(pixels, axis=0).astype(int))
                    target = (round(point['x']*width), round(point['y']*height))
                    cv2.arrowedLine(display, midpoint, target, (0, 255, 255), 2)
                    cv2.putText(display, 'IN', target, cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            for bbox in detections['bbox']:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 200, 0), 1)
            for track in tracks:
                track_id = track['track_id']
                identity = identities[track_id]
                name = identity.get('user_name') or 'Nguoi la'
                # OpenCV putText chỉ hỗ trợ ASCII; log vẫn giữ tên tiếng Việt.
                name = unicodedata.normalize('NFKD', name.replace('đ', 'd').replace('Đ', 'D'))
                name = name.encode('ascii', 'ignore').decode()
                score = identity.get('score_match')
                score_text = '' if score is None else f' {score:.2f}'
                direction = labels.get(track_id, ('', 0))[0]
                text = f'{track_id}: {name}{score_text} {direction}'
                color = (0, 255, 0) if identity['matched'] else (0, 0, 255)
                x1, y1, x2, y2 = map(int, track['bbox'])
                # Demo xét IN/OUT theo tâm bbox người, kể cả khi không thấy mặt.
                cv2.circle(display, ((x1+x2)//2, (y1+y2)//2), 4, color, -1)
                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display, text, (max(0, x1), max(20, y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
            scale = min(1, 1000/width, 800/height)
            if scale < 1:
                display = cv2.resize(display, (round(width*scale), round(height*scale)))
            cv2.imshow(window, display)
            t6 = time.perf_counter()
            time.sleep(max(0, 1/fps - (time.monotonic()-started)))
            t7 = time.perf_counter()
            timing_samples.append({
                "person": (t1 - t0) * 1000,
                "tracking": (t2 - t1) * 1000,
                "face_detect": (t3 - t2) * 1000,
                "event": (t4 - t3) * 1000,
                "recognition": (t5 - t4) * 1000,
                "output": (t6 - t5) * 1000,
                "sleep": (t7 - t6) * 1000,
                "processing": (t6 - loop_started) * 1000,
                "loop": (t7 - loop_started) * 1000,
            })
            timing_now = time.monotonic()
            elapsed = timing_now - timing_window_started
            if elapsed >= 1.0:
                count = len(timing_samples)
                averages = {
                    key: sum(sample[key] for sample in timing_samples) / count
                    for key in timing_samples[0]
                }
                worst = max(sample["processing"] for sample in timing_samples)
                details = " ".join(
                    f"{key}={value:.1f}ms"
                    for key, value in averages.items()
                )
                logger.info(
                    "PERF demo fps=%.1f %s max_processing=%.1fms",
                    count / elapsed, details, worst,
                )
                timing_samples.clear()
                timing_window_started = timing_now
    except KeyboardInterrupt:
        logger.info('Dừng demo.')
    finally:
        camera.stop()
        if verifier is not None:
            verifier.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Local face recognition and IN/OUT demo')
    parser.add_argument('--rtsp', help='Camera RTSP URL', default=r"rtsp://192.168.1.198:28537/io")
    parser.add_argument('--detector-model', '--weights', dest='weights',
                        help='InsightFace SCRFD model pack name (default: buffalo_s)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--config', type=Path, help='Config YAML path')
    parser.add_argument('--zones', type=Path, help='VMS JSON: ai_zones array or object with ai_zones')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    demo_zones = None
    if args.zones:
        import json
        data = json.loads(args.zones.read_text(encoding='utf-8-sig'))
        demo_zones = data['ai_zones'] if isinstance(data, dict) else data
    run_demo(rtsp_url=args.rtsp, weights=args.weights, device=args.device,
             debug=args.debug, zones=demo_zones, config_path=args.config)
