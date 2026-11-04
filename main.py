import argparse
import logging
import os
from pathlib import Path
import time
from dotenv import load_dotenv

from src.core_ai.event_processor import EventProcessor
from src.core_ai.identity_cache import IdentityCache
from src.core_ai.identity_manager import IdentityManager, associate_faces
from src.core_ai.camera_reader import CameraReader
from src.core_ai.detector import InferenceWorker, SCRFDDetector, PersonDetector
from src.core_ai.extractor import InsightFaceExtractor
from src.core_ai.tracker import PersonTrackerBoTSORT, PersonTrackerDeepSort
from src.database.verifier import Verifier
from src.vms_service.vms_client import VMSClient

ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


def load_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def model_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    return str(path)


def recognize_faces(frame, tracks, extractor, verifier, min_face_size, debug=False,
                    cache=None, now=None, force_ids=()):
    """Cắt mặt từ ảnh gốc; ghép identity bằng track_id, không bằng thứ tự list."""
    height, width = frame.shape[:2]
    identities = {}
    valid_tracks = []
    now = time.monotonic() if now is None else now
    if cache is not None:
        cache.prune(now)
    for track in tracks:
        if not track.get("is_detection", False):
            continue
        x1, y1, x2, y2 = map(int, track["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2-x1 < min_face_size or y2-y1 < min_face_size:
            continue
        track_id = track["track_id"]
        identity = cache.get(track_id, now, force=track_id in force_ids) if cache is not None else None
        if identity is None:
            # Vẫn extract từng mặt; chỉ chạy khi chưa có cache hoặc cần làm mới.
            landmarks = track.get('landmarks')
            if landmarks is not None:
                # Landmarks ở hệ tọa độ ảnh gốc, chỉ align khi cần làm mới identity.
                embedding = extractor.extract(frame, landmarks, align=True)
            else:
                embedding = extractor.extract(frame[y1:y2, x1:x2])
            identity = verifier.verify(embedding, debug=True) if debug else verifier.verify(embedding)
            # stt database local không phải person_id trên VMS; dùng user_code.
            identity.pop("person_id", None)
            if cache is not None:
                cache.put(track_id, identity, now)
        identities[track_id] = identity
        valid_tracks.append(track)
    return valid_tracks, identities


def recognize_people(frame, tracks, detections, extractor, verifier, min_face_size,
                     identity_manager, identity_cache=None, now=None, force_ids=(), debug=False, alive_track_ids=None):
    """Ghép mặt vào person track, nhận diện rồi giữ tên theo vòng đời track."""
    now = time.monotonic() if now is None else now
    face_tracks = associate_faces(detections, tracks)
    valid_faces, face_identities = recognize_faces(
        frame, face_tracks, extractor, verifier, min_face_size, debug=debug,
        cache=identity_cache, now=now, force_ids=force_ids)
    return valid_faces, identity_manager.update(tracks, face_identities, now, alive_track_ids)


def process_frame(code, sample, tracker, face_detector, person_detector, extractor, verifier, processor,
                  vms, min_face_size, upload_url, identity_manager, identity_cache=None):
    """Một frame: person track + face ID → IN/OUT theo người → publish VMS."""
    frame = sample["frame"]
    t0 = time.perf_counter()

    raw_detections = person_detector.detect(frame)
    t1 = time.perf_counter()
    tracks = tracker.update(raw_detections, frame)
    alive_track_ids = None
    if hasattr(tracker, "get_track_ids"):
        alive_track_ids = tracker.get_track_ids()
    t2 = time.perf_counter()
    detections = face_detector.detect_many_faces(frame)
    t3 = time.perf_counter()
    tracked_count = len(tracks)

    # 2. IN/OUT theo bbox người, kể cả khi không thấy mặt.
    zones = vms.get_zones(code)
    events = processor.process(code, tracks, frame.shape, zones,
                               now=sample["captured_monotonic"])
    t4 = time.perf_counter()
    # 3. Event có mặt thì xác minh lại ngay; không có mặt thì giữ tên trên person track.
    valid_faces, identities = recognize_people(
        frame, tracks, detections, extractor, verifier, min_face_size,
        identity_manager=identity_manager, identity_cache=identity_cache,
        now=sample["captured_monotonic"], force_ids={event["track_id"] for event in events}, alive_track_ids=alive_track_ids)
    t5 = time.perf_counter()
    # 4. Bbox người gửi mỗi frame; snapshot chỉ upload khi có sự kiện.
    vms.publish_bbox(code, tracks, frame.shape, identities=identities, detection_class="person")
    snapshot_url = None
    if events and upload_url:
        import cv2
        import requests
        try:
            ok, jpeg = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("Không encode được JPEG")
            snapshot_url = vms.upload_snapshot(code, jpeg.tobytes(), upload_url)
        except (cv2.error, requests.RequestException, ValueError, KeyError):
            # Không log URL/response vì có thể chứa thông tin kết nối.
            logger.warning("Camera %s: upload ảnh lỗi, gửi sự kiện không ảnh", code)

    for event in events:
        identity = identities[event["track_id"]]
        success = vms.publish_face(code, event, identity, snapshot_url=snapshot_url,
                                   event_time=sample["captured_at"])
        logger.info("camera=%s track=%s direction=%s matched=%s published=%s",
                    code, event["track_id"], event["direction"], identity["matched"], success)
    t6 = time.perf_counter()
    timings_ms = {
        "person": (t1 - t0) * 1000,
        "tracking": (t2 - t1) * 1000,
        "face_detect": (t3 - t2) * 1000,
        "event": (t4 - t3) * 1000,
        "recognition": (t5 - t4) * 1000,
        "output": (t6 - t5) * 1000,
        "total": (t6 - t0) * 1000,
    }
    return {"detections": len(detections["bbox"]), "tracks": tracked_count,
            "valid_faces": len(valid_faces), "matched": sum(bool(i["matched"]) for i in identities.values()),
            "zones": sum(z.get("is_active", True) and z.get("ai_modules", "FACE") == "FACE"
                         for z in zones), "events": len(events),
            "timing_ms": timings_ms}


def main():
    parser = argparse.ArgumentParser(description="Person tracking, face recognition and IN/OUT events for VMS")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--detector-model", help="InsightFace SCRFD model pack name (default: buffalo_s)")
    parser.add_argument("--debug", action="store_true", help="Log each bbox publish")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    runtime = config["runtime"]
    fps = float(runtime.get("processing_fps", 10))
    if fps <= 0:
        raise ValueError("processing_fps must > 0")
    min_face_size = int(runtime.get("min_face_size", 32))
    if min_face_size <= 0:
        raise ValueError("min_face_size must > 0")
    upload_url = os.getenv("VMS_STORAGE_UPLOAD_URL")
    if not upload_url:
        logger.warning("lack VMS_STORAGE_UPLOAD_URL: event wont return snapshot")

    inference = config.get("inference", {})
    device = args.device
    imgsz = inference.get("imgsz", 640)
    det_size = (imgsz, imgsz) if isinstance(imgsz, int) else tuple(imgsz)
    detector = SCRFDDetector(
        model_name=args.detector_model or config.get("model_detect", {}).get("model_name", "buffalo_s"),
        det_size=det_size, det_thresh=inference.get("conf", .5), device=device)
    extractor_weights = model_path(config["model_extract"]["weight"])
    extractor = InsightFaceExtractor(extractor_weights, device=device)
    detector.warmup()
    extractor.warmup()
    # YOLO dùng chung;
    person_config = dict(config.get("person_tracking", {}))
    person_detector = PersonDetector(model_path=model_path(person_config.get("weight", "yolo11n.pt")), conf=person_config.get("conf", 0.5),
                                     imgsz=person_config.get("imgsz", 640), device=device)
    person_detector.warmup()
    processor = EventProcessor(cooldown=runtime.get("event_cooldown", 1),
                               track_ttl=runtime.get("track_ttl", 2),
                               boundary_margin=runtime.get("boundary_margin", .003))
    vms = VMSClient.from_env()
    sessions = {}
    verifier = None

    try:
        verifier = Verifier(threshold=runtime["recognition_threshold"])
        vms.start()
        logger.info("Pipeline running, wait for camera turn on FACE from VMS. Ctrl+C to stop.")
        while True:
            cycle_started = time.monotonic()
            cameras = {camera["code"]: camera for camera in vms.get_cameras()}

            # Camera bị xóa/tắt FACE/đổi URL: đóng reader và bỏ lịch sử cũ.
            for code in list(sessions):
                camera = cameras.get(code)
                url = (camera.get("restream_urls", {}).get("FACE") or camera.get("rtsp_url")) if camera else None
                if not camera or url != sessions[code]["url"]:
                    sessions.pop(code)["reader"].stop()
                    processor.reset(code)
                    logger.info("Dừng camera %s", code)

            for code, camera in cameras.items():
                url = camera.get("restream_urls", {}).get("FACE") or camera.get("rtsp_url")
                if not url:
                    continue  # Camera ảo chưa có RTSP.
                if code not in sessions:
                    reader = CameraReader(camera["id"], url)
                    sessions[code] = {"reader": reader, "url": url,
                                      "tracker": PersonTrackerDeepSort(),
                                      "generation": None,
                                      "identity_cache": IdentityCache(
                                          interval=runtime.get("recognition_interval", .5),
                                          ttl=processor.track_ttl),
                                      "identity_manager": IdentityManager(ttl=processor.track_ttl),
                                      "last_frame": None}
                    reader.start()
                    logger.info("Bắt đầu camera %s", code)

                session = sessions[code]
                # Đọc frame mới nhất, không tích lũy hàng đợi ảnh cũ.
                sample = session["reader"].get_sample()
                if sample is None:
                    continue
                captured = sample["captured_monotonic"]
                if time.monotonic() - captured > processor.track_ttl:
                    continue
                # Reconnect RTSP hoặc gián đoạn quá lâu: không nối track cũ với người mới.
                if (sample["generation"] != session["generation"]
                        or (session["last_frame"] is not None
                            and captured - session["last_frame"] > processor.track_ttl)):
                    session["tracker"].reset()
                    processor.reset(code)
                    session["identity_cache"].clear()
                    session["identity_manager"].clear()
                session["generation"] = sample["generation"]
                session["last_frame"] = captured
                timing_state = session.setdefault("timing", {
                    "started": time.monotonic(),
                    "samples": [],
                })
                status = process_frame(code, sample, session["tracker"], detector, person_detector, extractor,
                                       verifier, processor, vms, min_face_size, upload_url,
                                       identity_cache=session["identity_cache"],
                                       identity_manager=session["identity_manager"])
                timing_state["samples"].append(status["timings_ms"])

                now = time.monotonic()
                elapsed = now - timing_state["started"]

                if elapsed >= 1.0:
                    samples = timing_state["samples"]
                    count = len(samples)
                    averages = {
                        key: sum(sample[key] for sample in samples) / count
                        for key in samples[0]
                    }
                    worst_total = max(sample["total"] for sample in samples)
                    details = " ".join(
                        f"{key}={value:.1f}ms"
                        for key, value in averages.items()
                    )
                    logger.info(
                        "PERF camera=%s fps=%.1f %s max_total=%.1fms",
                        code, count / elapsed, details, worst_total,
                    )
                    timing_state["samples"].clear()
                    timing_state["started"] = now

                if time.monotonic() >= session.get("next_status", 0):
                    logger.info("pipeline camera=%s %s", code, status)
                    if status["zones"] == 0:
                        logger.warning("camera=%s: have not line/polygon FACE; can't process IN/OUT", code)
                    session["next_status"] = time.monotonic()+2

            # Giới hạn FPS xử lý; model dùng chung và các camera xử lý lần lượt.
            time.sleep(max(0, 1/fps - (time.monotonic() - cycle_started)))
    except KeyboardInterrupt:
        logger.info("STOPPING pipeline...")
    finally:
        for session in sessions.values():
            session["reader"].stop()
        vms.stop()
        if verifier is not None:
            verifier.close()


if __name__ == "__main__":
    main()
