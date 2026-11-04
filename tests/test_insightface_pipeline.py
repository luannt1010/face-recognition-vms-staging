import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


class SCRFDResultTests(unittest.TestCase):
    def detector_with_faces(self, faces):
        from src.core_ai.detector import SCRFDDetector

        detector = SCRFDDetector.__new__(SCRFDDetector)
        detector.model = Mock()
        detector.model.get.return_value = faces
        return detector

    def test_single_face_list_returns_landmarks_with_batch_dimension(self):
        landmarks = np.arange(10, dtype=np.float32).reshape(5, 2)
        face = SimpleNamespace(bbox=[10, 20, 60, 80], det_score=.9, kps=landmarks)
        detector = self.detector_with_faces([face])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_one_face(frame)
        detector.model.get.assert_called_once_with(frame, max_num=1)
        self.assertEqual(result['bbox'].shape, (1, 4))
        self.assertEqual(result['det_score'].shape, (1,))
        self.assertEqual(result['landmarks'].shape, (1, 5, 2))
        np.testing.assert_array_equal(result['landmarks'][0], landmarks)

    def test_no_face_returns_empty_arrays_for_both_methods(self):
        detector = self.detector_with_faces([])
        for detect in (detector.detect_one_face, detector.detect_many_faces):
            result = detect(np.zeros((100, 100, 3), dtype=np.uint8))
            self.assertEqual(result['bbox'].shape, (0, 4))
            self.assertEqual(result['det_score'].shape, (0,))
            self.assertEqual(result['landmarks'].shape, (0, 5, 2))

    def test_many_faces_keeps_all_faces_for_enrollment_count_validation(self):
        face = SimpleNamespace(bbox=[10, 20, 60, 80], det_score=.9, kps=np.zeros((5, 2)))
        detector = self.detector_with_faces([face, face])
        result = detector.detect_many_faces(np.zeros((100, 100, 3), dtype=np.uint8))
        self.assertEqual(result['bbox'].shape, (2, 4))
        self.assertEqual(result['landmarks'].shape, (2, 5, 2))
        self.assertEqual(detector.model.get.call_args.kwargs['max_num'], 0)


@unittest.skipUnless(importlib.util.find_spec('supervision') and importlib.util.find_spec('trackers'),
                     'Tracking requires supervision and trackers')
class LandmarkTrackingTests(unittest.TestCase):
    def test_landmarks_follow_detection_when_tracker_reorders_faces(self):
        import supervision as sv
        from src.core_ai.tracker import FaceTracker

        boxes = np.array([[10, 10, 50, 50], [60, 10, 100, 50]], dtype=np.float32)
        landmarks = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
        backend = Mock()
        backend.update.return_value = sv.Detections(
            xyxy=boxes[::-1].copy(), tracker_id=np.array([42, 7]))
        tracks = FaceTracker(tracker=backend).update({
            'bbox': boxes, 'det_score': np.array([.9, .8]), 'landmarks': landmarks})

        by_id = {track['track_id']: track for track in tracks}
        np.testing.assert_array_equal(by_id[7]['landmarks'], landmarks[0])
        np.testing.assert_array_equal(by_id[42]['landmarks'], landmarks[1])

    def test_boxes_without_landmarks_and_empty_detections_still_work(self):
        import supervision as sv
        from src.core_ai.tracker import FaceTracker

        boxes = np.array([[10, 10, 50, 50]], dtype=np.float32)
        backend = Mock()
        backend.update.return_value = sv.Detections(xyxy=boxes, tracker_id=np.array([7]))
        tracker = FaceTracker(tracker=backend)
        tracks = tracker.update({'bbox': boxes, 'det_score': [.9]})
        self.assertEqual(tracks[0]['track_id'], 7)
        self.assertNotIn('landmarks', tracks[0])
        self.assertEqual(tracker.update({
            'bbox': np.empty((0, 4)), 'det_score': np.empty(0),
            'landmarks': np.empty((0, 5, 2))}), [])


@unittest.skipUnless(importlib.util.find_spec('insightface'), 'Extractor requires InsightFace')
class InsightFaceAlignmentTests(unittest.TestCase):
    def test_numpy_landmarks_align_before_extraction_and_normalization(self):
        from src.core_ai.extractor import InsightFaceExtractor

        extractor = InsightFaceExtractor.__new__(InsightFaceExtractor)
        extractor.model = Mock()
        extractor.model.get_feat.return_value = np.array([[3., 4.]], dtype=np.float32)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = np.ones((5, 2), dtype=np.float32)
        aligned = np.zeros((112, 112, 3), dtype=np.uint8)

        with patch('src.core_ai.extractor.face_align.norm_crop', return_value=aligned) as align:
            embedding = extractor.extract(frame, landmarks, align=True)
            self.assertIs(align.call_args.args[0], frame)
            self.assertIs(align.call_args.kwargs['landmark'], landmarks)
            self.assertEqual(align.call_args.kwargs['image_size'], 112)
        self.assertIs(extractor.model.get_feat.call_args.args[0], aligned)
        np.testing.assert_allclose(embedding, [.6, .8])
        with self.assertRaises(ValueError):
            extractor.extract(frame, align=True)
