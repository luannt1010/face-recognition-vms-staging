"""Chạy: python main.py --config configs/config.yml. Dừng bằng Ctrl+C."""

import argparse
import logging
import os
from pathlib import Path
import time

from src.core_ai.event_processor import EventProcessor
from src.core_ai.identity_cache import IdentityCache

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
            face = frame[y1:y2, x1:x2]
            embedding = extractor.extract(face)
            identity = verifier.verify(embedding, debug=True) if debug else verifier.verify(embedding)
            # stt database local không phải person_id trên VMS; dùng user_code.
            identity.pop("person_id", None)
            if cache is not None:
                cache.put(track_id, identity, now)
        identities[track_id] = identity
        valid_tracks.append(track)
    return valid_tracks, identities


def process_frame(code, sample, tracker, detector, extractor, verifier, processor,
                  vms, min_face_size, upload_url, identity_cache=None):
    """Một frame: detect → track → identity → event → publish."""
    frame = sample["frame"]

    # 1. Phát hiện mặt trên ảnh gốc.
    detections = detector.predict([frame])[0]

    # 2. Gán track_id. Mỗi camera có tracker riêng.
    tracks = tracker.update(detections)
    tracked_count = len(tracks)

    # 3. Track mới nhận diện ngay; track cũ dùng cache đến kỳ làm mới.
    tracks, identities = recognize_faces(frame, tracks, extractor, verifier, min_face_size,
                                         cache=identity_cache, now=sample["captured_monotonic"])

    # 4. Lấy vùng mới nhất từ VMS rồi xác định IN/OUT.
    zones = vms.get_zones(code)
    events = processor.process(code, tracks, frame.shape, zones,
                               now=sample["captured_monotonic"])
    if identity_cache is not None and events:
        event_ids = {event["track_id"] for event in events}
        event_tracks = [track for track in tracks if track["track_id"] in event_ids]
        _, fresh = recognize_faces(frame, event_tracks, extractor, verifier, min_face_size,
                                    cache=identity_cache, now=sample["captured_monotonic"],
                                    force_ids=event_ids)
        identities.update(fresh)

    # 5. Bbox gửi mỗi frame; snapshot chỉ upload khi có sự kiện.
    vms.publish_bbox(code, tracks, frame.shape, identities=identities)
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
    return {"detections": len(detections["bbox"]), "tracks": tracked_count,
            "valid_faces": len(tracks), "matched": sum(bool(i["matched"]) for i in identities.values()),
            "zones": sum(z.get("is_active", True) and z.get("ai_modules", "FACE") == "FACE"
                         for z in zones), "events": len(events)}


def main():
    parser = argparse.ArgumentParser(description="Face recognition and IN/OUT events for VMS")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--debug", action="store_true", help="Log each bbox publish")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Import model tại đây để python main.py --help không cần nạp model/GPU.
    from dotenv import load_dotenv
    from src.core_ai.camera_reader import CameraReader
    from src.core_ai.detector import InferenceWorker
    from src.core_ai.extractor import Extractor
    from src.core_ai.tracker import FaceTracker
    from src.database.verifier import Verifier
    from src.vms_service.vms_client import VMSClient

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    runtime = config["runtime"]
    fps = float(runtime.get("processing_fps", 10))
    if fps <= 0:
        raise ValueError("processing_fps phải > 0")
    min_face_size = int(runtime.get("min_face_size", 32))
    if min_face_size <= 0:
        raise ValueError("min_face_size phải > 0")
    upload_url = os.getenv("VMS_STORAGE_UPLOAD_URL")
    if not upload_url:
        logger.warning("Thiếu VMS_STORAGE_UPLOAD_URL: sự kiện sẽ không có snapshot")

    detector_weights = model_path(os.getenv("FACE_DETECTOR_WEIGHTS") or config["model_detect"]["weight"])
    extractor_weights = model_path(os.getenv("FACE_EXTRACTOR_WEIGHTS") or config["model_extract"]["weight"])
    detector = InferenceWorker(wp=detector_weights, **config["inference"])
    extractor = Extractor(extractor_weights)
    extractor.warmup()
    processor = EventProcessor(cooldown=runtime.get("event_cooldown", 1),
                               track_ttl=runtime.get("track_ttl", 2),
                               boundary_margin=runtime.get("boundary_margin", .003))
    vms = VMSClient.from_env()
    sessions = {}
    verifier = None

    try:
        verifier = Verifier(threshold=runtime["recognition_threshold"])
        vms.start()
        logger.info("Pipeline đang chạy, chờ camera bật FACE từ VMS. Ctrl+C để dừng.")
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
                                      "tracker": FaceTracker(fps=fps), "generation": None,
                                      "identity_cache": IdentityCache(
                                          interval=runtime.get("recognition_interval", .5),
                                          ttl=processor.track_ttl),
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
                    session["tracker"] = FaceTracker(fps=fps)
                    processor.reset(code)
                    session["identity_cache"].clear()
                session["generation"] = sample["generation"]
                session["last_frame"] = captured
                status = process_frame(code, sample, session["tracker"], detector, extractor,
                                       verifier, processor, vms, min_face_size, upload_url,
                                       identity_cache=session["identity_cache"])
                if time.monotonic() >= session.get("next_status", 0):
                    logger.info("pipeline camera=%s %s", code, status)
                    if status["zones"] == 0:
                        logger.warning("camera=%s: chưa có vùng/vạch FACE hoạt động; chưa thể sinh IN/OUT", code)
                    session["next_status"] = time.monotonic()+2

            # Giới hạn FPS xử lý; model dùng chung và các camera xử lý lần lượt.
            time.sleep(max(0, 1/fps - (time.monotonic() - cycle_started)))
    except KeyboardInterrupt:
        logger.info("Đang dừng pipeline...")
    finally:
        for session in sessions.values():
            session["reader"].stop()
        vms.stop()
        if verifier is not None:
            verifier.close()


if __name__ == "__main__":
    main()
