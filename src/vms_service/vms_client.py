from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from threading import Event, Lock
import time
from uuid import uuid4

import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)
VN_TIMEZONE = timezone(timedelta(hours=7))


def topic_code(code):
    # Giữ nguyên Unicode/dấu cách theo VMS, chỉ thay / # +.
    return str(code).replace("/", "_").replace("#", "_").replace("+", "_")


class VMSClient:
    def __init__(self, host, username, password, client_id=None, zone_id=283,
                 organization_id=103, port=443):
        self.host, self.port = host, port
        self.zone_id, self.organization_id = zone_id, organization_id
        self.camera_topic = f"smart_vms/cameras/zone/{zone_id}"
        self.lock = Lock()
        self.connected = Event()
        self.cameras = {}
        self.zones = {}
        client_id = client_id or f"tts_luan_{uuid4().hex[:8]}"
        if client_id.startswith("vms_staging_backend"):
            raise ValueError("AI edge phải dùng client_id riêng")
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                                  client_id=client_id, transport="websockets")
        self.client.ws_set_options(path="/mqtt")
        self.client.tls_set()
        self.client.username_pw_set(username, password)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    @classmethod
    def from_env(cls):
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[2] / ".env")
        return cls(host=os.environ["MQTT_HOST"], username=os.environ["MQTT_USERNAME"],
                   password=os.environ["MQTT_PASSWORD"], client_id=os.getenv("MQTT_CLIENT_ID"),
                   zone_id=int(os.getenv("VMS_ZONE_ID", "283")),
                   organization_id=int(os.getenv("VMS_ORGANIZATION_ID", "103")),
                   port=int(os.getenv("MQTT_PORT", "443")))

    def start(self, timeout=10):
        self.client.connect_async(self.host, self.port, keepalive=30)
        self.client.loop_start()
        if not self.connected.wait(timeout):
            self.stop()
            raise TimeoutError("Chưa kết nối được MQTT VMS; kiểm tra host và tài khoản")

    def stop(self):
        self.client.disconnect()
        self.client.loop_stop()
        self.connected.clear()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            logger.error("MQTT từ chối kết nối: %s", reason_code)
            return
        # Chạy lại sau reconnect: subscribe và xin camera lại.
        client.subscribe(self.camera_topic, qos=1)
        client.subscribe("smart_vms/cameras/+/ai_zones/FACE", qos=1)
        client.publish("smart_vms/cameras/getdevice", json.dumps({
            "zone_id": self.zone_id, "organization_id": self.organization_id}), qos=1)
        logger.info("VMS camera scope: topic=%s zone_id=%s organization_id=%s",
                    self.camera_topic, self.zone_id, self.organization_id)
        self.connected.set()

    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        self.connected.clear()
        with self.lock:
            self.cameras.clear()
            self.zones.clear()

    def _on_message(self, client, userdata, message):
        try:
            data = json.loads(message.payload.decode("utf-8")) if message.payload else {}
            with self.lock:
                if message.topic == self.camera_topic:
                    cameras = data.get("cameras", [])
                    self.cameras = {c["code"]: c for c in cameras
                                    if "FACE" in c.get("ai_modules", [])}
                    logger.info("VMS config: topic=%s received=%s cameras_FACE=%s",
                                message.topic, len(cameras), list(self.cameras))
                    if not cameras:
                        logger.warning("VMS trả 0 camera cho zone_id=%s; kiểm tra zone chứa camera "
                                       "và organization_id=%s", self.zone_id, self.organization_id)
                    elif not self.cameras:
                        # Chỉ log mã camera/module, không log RTSP hoặc payload có mật khẩu.
                        modules = [(c.get("code"), c.get("ai_modules", [])) for c in cameras[:10]]
                        logger.warning("VMS trả %s camera nhưng không camera nào bật FACE; "
                                       "camera_modules (tối đa 10)=%s", len(cameras), modules)
                else:
                    # topic này chứa cả polygon và line crossing.
                    code = message.topic.split("/")[2]
                    self.zones[code] = data.get("ai_zones", [])
                    selected = any(topic_code(camera_code) == code for camera_code in self.cameras)
                    logger.info("VMS config: camera=%s zones_FACE=%s camera_selected=%s",
                                code, len(self.zones[code]), selected)
        except (ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Bỏ payload cấu hình không hợp lệ từ %s", message.topic)

    def get_cameras(self):
        """Chỉ camera bật FACE. Vòng xử lý phải dừng reader không còn trong list."""
        with self.lock:
            return deepcopy(list(self.cameras.values()))

    def get_zones(self, camera_code):
        with self.lock:
            if camera_code not in self.cameras:
                return []
            return deepcopy(self.zones.get(topic_code(camera_code), []))

    def _publish(self, topic, data, qos):
        """True = broker nhận (QoS 1), không phải backend đã lưu sự kiện."""
        if not self.connected.is_set():
            logger.warning("Chưa kết nối: topic=%s camera=%s", topic, data.get("camera_code"))
            return False
        result = self.client.publish(topic, json.dumps(data, ensure_ascii=False, allow_nan=False),
                                     qos=qos, retain=False)
        try:
            if qos == 1 and result.rc == mqtt.MQTT_ERR_SUCCESS:
                result.wait_for_publish(timeout=3)
            success = result.rc == mqtt.MQTT_ERR_SUCCESS and (qos == 0 or result.is_published())
        except (RuntimeError, ValueError):
            success = False
        log = logger.debug if qos == 0 and success else logger.info
        log("publish topic=%s camera=%s object_key=%s success=%s",
            topic, data.get("camera_code"), data.get("snapshot_url"), success)
        return success

    def publish_face(self, camera_code, event, identity, snapshot_url=None, event_time=None):
        """event từ EventProcessor, identity từ Verifier. Người lạ bỏ user_code."""
        with self.lock:
            camera = deepcopy(self.cameras.get(camera_code))
        if camera is None:
            return False
        if event["direction"] not in ("IN", "OUT"):
            raise ValueError("direction phải là IN hoặc OUT")
        when = event_time or datetime.now(VN_TIMEZONE)
        if when.tzinfo is None:
            raise ValueError("event_time phải có timezone")
        timestamp = when.astimezone(VN_TIMEZONE).isoformat()
        data = {"ai_modules": "FACE", "camera_id": camera["id"], "camera_code": camera_code,
                "organization_id": self.organization_id, "access_time": timestamp,
                "event_time": timestamp, "direction": event["direction"],
                "score_match": identity.get("score_match"),
                "payload": {"direction": event["direction"], "zone_id": event["zone_id"],
                            "zone_name": event.get("zone_name", "")}}
        if identity.get("matched"):
            for field in ("user_code", "user_name", "department_name", "person_id"):
                if identity.get(field) is not None:
                    data[field] = identity[field]
        if snapshot_url:
            data["snapshot_url"] = snapshot_url
        return self._publish("smart_vms/ai_events/FACE", data, qos=1)

    def publish_bbox(self, camera_code, tracks, frame_shape, identities=None, detection_class="face"):
        with self.lock:
            camera = deepcopy(self.cameras.get(camera_code))
        if camera is None:
            return False
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Kích thước frame phải > 0")
        detections = []
        identities = identities or {}
        for track in tracks:
            if not track.get("is_detection", False):
                continue
            bbox = [max(0.0, min(1.0, float(v)/size))
                    for v, size in zip(track["bbox"], (width, height, width, height))]
            identity = identities.get(track["track_id"], {})
            detections.append({"id": str(track["track_id"]), "track_id": int(track["track_id"]),
                               "class": detection_class, "confidence": float(track["score"]), "bbox": bbox,
                               "color": "#22c55e" if identity.get("matched") else "#ef4444"})
        return self._publish(f"smart_vms/ai/bbox/{topic_code(camera_code)}", {
            "ai_modules": "FACE", "camera_id": camera["id"], "camera_code": camera_code,
            "timestamp": time.time(), "detections": detections}, qos=0)

    def upload_snapshot(self, camera_code, jpeg_bytes, upload_url):
        """jpeg_bytes lấy từ cv2.imencode('.jpg', frame)[1].tobytes()."""
        import requests
        with self.lock:
            camera = deepcopy(self.cameras[camera_code])
        response = requests.post(upload_url, data={"camera_id": camera["id"], "category": "face"},
                                 files={"file": ("snapshot.jpg", jpeg_bytes, "image/jpeg")}, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data.get("success") or not data.get("object_key"):
            raise ValueError("Storage API không trả object_key thành công")
        return data["object_key"]
