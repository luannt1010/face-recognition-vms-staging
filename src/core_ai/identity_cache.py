import math


class IdentityCache:
    """Cache danh tính cho một camera. Không giữ ảnh hoặc embedding."""
    def __init__(self, interval=0.5, ttl=2.0):
        self.interval = interval
        self.ttl = ttl
        self.entries = {}

    def clear(self):
        self.entries.clear()

    def prune(self, now):
        """Quên track đã mất detection quá lâu."""
        for track_id in list(self.entries):
            if now - self.entries[track_id]["last_seen"] > self.ttl:
                del self.entries[track_id]

    def get(self, track_id, now, force=False):
        entry = self.entries.get(track_id)
        if entry is None:
            return None
        entry["last_seen"] = now
        # Event cần kết quả của frame hiện tại. Nếu vừa extract frame này thì dùng lại.
        if force:
            return entry["identity"].copy() if entry["recognized_at"] == now else None
        if now - entry["recognized_at"] >= self.interval:
            return None
        return entry["identity"].copy()

    def put(self, track_id, identity, now):
        self.entries[track_id] = {"identity": identity.copy(),
                                  "recognized_at": now, "last_seen": now}
