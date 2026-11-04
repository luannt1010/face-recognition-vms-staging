import numpy as np


def associate_faces(faces, people, min_containment=.8):
    """Chỉ ghép khi mặt thuộc rõ ràng một người, mỗi người tối đa một mặt.

    Trả face tracks dùng person track_id, bbox/landmarks vẫn là của khuôn mặt.
    Không đoán danh tính khi các bbox người chồng lấn gây nhập nhằng.
    """
    matches = {}
    for index, bbox in enumerate(faces['bbox']):
        x1, y1, x2, y2 = map(float, bbox)
        area = (x2-x1) * (y2-y1)
        if area <= 0:
            continue
        cx, cy = (x1+x2)/2, (y1+y2)/2
        candidates = []
        for person in people:
            if not person.get('is_detection', False):
                continue
            px1, py1, px2, py2 = map(float, person['bbox'])
            # Mặt cần ở nửa trên (cho phép tới 60%) của bbox người, và tâm phải nằm giữa cạnh trái và phải của bbox người
            if not (px1 <= cx <= px2 and py1 <= cy <= py1+0.6*(py2-py1)):
                continue
            intersection = max(0, min(x2, px2)-max(x1, px1)) * max(0, min(y2, py2)-max(y1, py1))
            if intersection / area >= min_containment:
                candidates.append(person['track_id'])
        if len(candidates) == 1:
            matches.setdefault(candidates[0], []).append(index)

    tracks = []
    for track_id, indices in matches.items():
        if len(indices) != 1:
            continue
        index = indices[0]
        tracks.append({'track_id': track_id, 'bbox': np.asarray(faces['bbox'][index]).copy(),
                       'landmarks': np.asarray(faces['landmarks'][index]).copy(),
                       'score': float(faces['det_score'][index]), 'is_detection': True})
    return tracks


class IdentityManager:
    """Một camera: giữ tên đã xác nhận khi còn person track, dù không thấy mặt.

    Không phục hồi danh tính giữa hai track_id khác nhau hoặc giữa các camera.
    """
    def __init__(self, ttl=2):
        self.ttl = ttl
        self.entries = {}

    def clear(self):
        self.entries.clear()

    def update(self, people, face_identities, now, alive_track_ids=None):
        # Nếu có alive track ids thì theo lần update của tracker ngược lại theo ttl
        if alive_track_ids is not None:
            alive_track_ids = set(alive_track_ids)
            self.entries = {track_id: entry for track_id, entry in self.entries.items() if track_id in alive_track_ids}
        else:
            self.entries = {track_id: entry for track_id, entry in self.entries.items() if now-entry['last_seen'] <= self.ttl}
        identities = {}
        for person in people:
            if not person.get('is_detection', False):
                continue
            track_id = person['track_id']
            entry = self.entries.get(track_id)
            identity = face_identities.get(track_id)
            # Một lần không match mặt không xóa tên đã xác nhận trên cùng person track.
            if identity is None or not identity['matched']:
                if entry is not None and entry['identity']['matched']:
                    identity = entry['identity']
            if identity is None:
                identity = {'matched': False, 'score_match': None}
            self.entries[track_id] = {'identity': identity.copy(), 'last_seen': now}
            identities[track_id] = identity.copy()
        return identities
