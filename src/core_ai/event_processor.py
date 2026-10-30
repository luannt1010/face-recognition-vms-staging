from copy import deepcopy
import math
import time


def _point(value):
    x, y = float(value["x"]), float(value["y"])
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
        raise ValueError("Toạ độ VMS phải nằm trong khoảng 0..1")
    return x, y


def _cross(a, b, p):
    return (b[0]-a[0]) * (p[1]-a[1]) - (b[1]-a[1]) * (p[0]-a[0])


def _distance(p, a, b):
    """Khoảng cách từ điểm p đến đoạn thẳng ab."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx + dy*dy
    t = 0 if length2 == 0 else max(0, min(1, ((p[0]-a[0])*dx + (p[1]-a[1])*dy) / length2))
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def _inside(p, points):
    """Ray casting: tia ngang cắt số lẻ cạnh thì điểm ở trong polygon."""
    inside = False
    for a, b in zip(points, points[1:] + points[:1]):
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1]-a[1]) * (b[0]-a[0]) / (b[1]-a[1])
            if p[0] < x:
                inside = not inside
    return inside


class EventProcessor:
    """Xác định IN/OUT từ tâm bbox khuôn mặt. Không chạy model hay gửi MQTT."""
    def __init__(self, cooldown=1.0, track_ttl=2.0, boundary_margin=0.003):
        if (not all(math.isfinite(v) for v in (cooldown, track_ttl, boundary_margin))
                or cooldown < 0 or track_ttl <= 0 or boundary_margin < 0):
            raise ValueError("cooldown/margin >= 0 và track_ttl > 0")
        self.cooldown = cooldown
        self.track_ttl = track_ttl
        self.margin = boundary_margin
        self.cameras = {}

    def reset(self, camera_code):
        """Gọi khi camera reconnect hoặc tracker được tạo lại."""
        self.cameras.pop(camera_code, None)

    def process(self, camera_code, tracks, frame_shape, zones, now=None):
        """Trả list {track_id, zone_id, zone_name, direction} cho frame hiện tại.

        tracks: đầu ra FaceTracker (bbox pixel, is_detection=True).
        zones: mảng ai_zones từ topic .../ai_zones/FACE.
        frame_shape: frame.shape; now dùng time.monotonic(), có thể truyền khi test.
        """
        now = time.monotonic() if now is None else now
        height, width = frame_shape[:2]
        if height <= 0 or width <= 0:
            raise ValueError("Kích thước frame phải > 0")
        # Sửa vùng không được coi là người di chuyển qua vùng.
        if camera_code not in self.cameras or self.cameras[camera_code]["zones"] != zones:
            self.cameras[camera_code] = {"zones": deepcopy(zones), "tracks": {}}
        history = self.cameras[camera_code]["tracks"]
        for key in list(history):
            if now - history[key]["seen"] > self.track_ttl:
                del history[key]

        events = []
        for zone in zones:
            if not zone.get("is_active", True) or zone.get("ai_modules", "FACE") != "FACE":
                continue
            zone_id = zone.get("id") or zone["code"]
            is_line = zone.get("shape_type") == "line_crossing" or "line" in zone
            points = [_point(p) for p in (zone["line"]["points"] if is_line else zone["points"])]
            if is_line:
                a, b = points
                in_point = _point(zone.get("direction_point") or zone["point"])
                in_side = _cross(a, b, in_point)

            for track in tracks:
                if not track.get("is_detection", False):
                    continue
                x1, y1, x2, y2 = map(float, track["bbox"])
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                    continue
                p = ((x1+x2)/(2*width), (y1+y2)/(2*height))
                key = (zone_id, int(track["track_id"]))
                old = history.get(key)
                if old:
                    old["seen"] = now
                # Điểm trong dải biên giữ phía ổn định trước đó để giảm rung bbox.
                if is_line:
                    side = _cross(a, b, p)
                    if abs(side) / math.dist(a, b) <= self.margin:
                        continue
                    inside = side * in_side > 0
                else:
                    edges = zip(points, points[1:] + points[:1])
                    if min(_distance(p, a, b) for a, b in edges) <= self.margin:
                        continue
                    inside = _inside(p, points)

                last_event = old["last_event"] if old else -math.inf
                crossed = old is not None and old["inside"] != inside
                if crossed and is_line:
                    # Chỉ cắt đoạn vạch thật, không tính đường kéo dài vô hạn.
                    crossed = _cross(old["point"], p, a) * _cross(old["point"], p, b) <= 0
                if crossed and now - last_event >= self.cooldown:
                    events.append({"track_id": key[1], "zone_id": zone_id,
                                   "zone_name": zone.get("name", ""),
                                   "direction": "IN" if inside else "OUT"})
                    last_event = now
                history[key] = {"point": p, "inside": inside, "seen": now, "last_event": last_event}
        return events
