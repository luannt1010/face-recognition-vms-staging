import numpy as np
import supervision as sv
from trackers import ByteTrackTracker


class FaceTracker:
    def __init__(self, fps=30, tracker=None, min_detection_iou=0.5):
        self.min_detection_iou = float(min_detection_iou)
        self.tracker = tracker if tracker is not None else ByteTrackTracker(
            frame_rate=fps,
            track_activation_threshold=0.5,
            high_conf_det_threshold=0.6,
            lost_track_buffer=30,
            minimum_consecutive_frames=2,
            minimum_iou_threshold=0.1
        )

    def update(self, faces):
        bboxes = np.asarray(faces["bbox"] if faces else [], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(faces["det_score"] if faces else [], dtype=np.float32)
        if len(bboxes) != len(scores):
            raise ValueError("Detection boxes and scores must have the same length")
        detections = sv.Detections(xyxy=bboxes, confidence=scores)
        tracked = self.tracker.update(detections)
        if len(bboxes) == 0 or tracked.tracker_id is None:
            return []

        # Ghép 1-1 với detection của chính frame này. Chỉ giữ ID của tracker;
        # bbox/score luôn lấy từ detector, không dùng hộp Kalman dự đoán.
        candidates = []
        for track_index, (bbox, track_id) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
            if track_id is None or int(track_id) < 0:
                continue
            top_left = np.maximum(bbox[:2], bboxes[:, :2])
            bottom_right = np.minimum(bbox[2:], bboxes[:, 2:])
            intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
            area = np.prod(np.maximum(bbox[2:] - bbox[:2], 0))
            areas = np.prod(np.maximum(bboxes[:, 2:] - bboxes[:, :2], 0), axis=1)
            ious = intersection / np.maximum(area + areas - intersection, 1e-6)
            for detection_index in np.flatnonzero(ious >= self.min_detection_iou):
                candidates.append((float(ious[detection_index]), track_index, int(detection_index)))

        used_tracks, used_detections = set(), set()
        results = []
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            results.append({
                "track_id": int(tracked.tracker_id[track_index]),
                "bbox": bboxes[detection_index].copy(),
                "score": float(scores[detection_index]),
                "is_detection": True,
            })
        return results
