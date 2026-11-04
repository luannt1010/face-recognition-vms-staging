import importlib.util
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.core_ai.identity_manager import IdentityManager, associate_faces


def person(track_id, bbox, real=True):
    return {'track_id': track_id, 'bbox': bbox, 'is_detection': real}


def face_detections(boxes):
    return {'bbox': np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
            'landmarks': np.zeros((len(boxes), 5, 2), dtype=np.float32),
            'det_score': np.full(len(boxes), .9)}


class FacePersonAssociationTests(unittest.TestCase):
    def test_face_keeps_person_id_and_own_bbox_and_landmarks(self):
        faces = face_detections([[65, 5, 85, 25], [10, 5, 30, 25]])
        people = [person(7, [0, 0, 40, 100]), person(42, [60, 0, 100, 100])]
        tracks = associate_faces(faces, people)
        self.assertEqual([t['track_id'] for t in tracks], [42, 7])
        np.testing.assert_array_equal(tracks[0]['bbox'], faces['bbox'][0])
        np.testing.assert_array_equal(tracks[1]['landmarks'], faces['landmarks'][1])

    def test_ambiguous_faces_and_multiple_faces_in_one_body_are_skipped(self):
        faces = face_detections([[10, 5, 30, 25]])
        overlapping = [person(1, [0, 0, 50, 100]), person(2, [5, 0, 55, 100])]
        self.assertEqual(associate_faces(faces, overlapping), [])
        two_faces = face_detections([[10, 5, 30, 25], [40, 5, 60, 25]])
        self.assertEqual(associate_faces(two_faces, [person(1, [0, 0, 100, 100])]), [])

    def test_lower_body_outside_and_predicted_people_do_not_receive_faces(self):
        faces = face_detections([[10, 70, 30, 90], [110, 5, 130, 25]])
        self.assertEqual(associate_faces(faces, [person(1, [0, 0, 50, 100])]), [])
        self.assertEqual(associate_faces(face_detections([[10, 5, 30, 25]]),
                                        [person(1, [0, 0, 50, 100], real=False)]), [])
        self.assertEqual(associate_faces(face_detections([]), []), [])


class PersonIdentityTests(unittest.TestCase):
    def test_name_survives_missing_face_and_temporary_unmatched_face(self):
        manager = IdentityManager(ttl=2)
        tracks = [person(7, [0, 0, 40, 100])]
        known = {'matched': True, 'user_code': 'A', 'user_name': 'An'}
        manager.update(tracks, {7: known}, now=0)
        for now in (1, 2, 3, 4):
            self.assertEqual(manager.update(tracks, {}, now)[7]['user_code'], 'A')
        self.assertEqual(manager.update(tracks, {7: {'matched': False}}, now=5)[7]['user_code'], 'A')

    def test_expiry_new_id_reset_and_camera_isolation_do_not_inherit_name(self):
        manager = IdentityManager(ttl=2)
        tracks = [person(7, [0, 0, 40, 100])]
        known = {'matched': True, 'user_code': 'A'}
        manager.update(tracks, {7: known}, now=0)
        self.assertFalse(manager.update([person(8, [0, 0, 40, 100])], {}, 1)[8]['matched'])
        self.assertFalse(manager.update(tracks, {}, 3)[7]['matched'])
        manager.update(tracks, {7: known}, now=4)
        manager.clear()
        self.assertFalse(manager.update(tracks, {}, 4.1)[7]['matched'])
        other = IdentityManager()
        self.assertFalse(other.update(tracks, {}, 4.1)[7]['matched'])


@unittest.skipUnless(importlib.util.find_spec('supervision') and importlib.util.find_spec('trackers'),
                     'Tracking requires supervision and trackers')
class PersonTrackerTests(unittest.TestCase):
    def test_shared_detector_keeps_tracker_state_separate_for_each_camera(self):
        from src.core_ai.tracker import PersonTracker

        detector = Mock()
        detector.predict.return_value = [{'bbox': np.array([[10, 10, 50, 90]], dtype=np.float32),
                                          'det_score': np.array([.9])}]
        with patch('src.core_ai.detector.InferenceWorker') as model:
            first = PersonTracker(fps=10, detector=detector)
            second = PersonTracker(fps=10, detector=detector)
        model.assert_not_called()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first.update(frame)
        self.assertEqual(first.tracker.frame_id, 1)
        self.assertEqual(second.tracker.frame_id, 0)
        self.assertIs(first.detector, second.detector)
        first.reset()
        self.assertIs(first.detector, detector)

    def test_yolo_person_filter_stable_ids_warmup_and_reset(self):
        from src.core_ai.tracker import PersonTracker

        detector = Mock()
        detector.predict.return_value = [{'bbox': np.array([[10, 10, 50, 90]], dtype=np.float32),
                                          'det_score': np.array([.9])}]
        with patch('src.core_ai.detector.InferenceWorker', return_value=detector) as model:
            tracker = PersonTracker(model_path='fake.pt', fps=10, device='cpu')
        model.assert_called_once_with(wp='fake.pt', conf=.1, imgsz=640, classes=[0], device='cpu')
        tracker.warmup()
        self.assertEqual(tracker.tracker.frame_id, 0)  # warmup does not consume a tracking frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        first = tracker.update(frame)
        second = tracker.update(frame)
        self.assertEqual(first[0]['track_id'], second[0]['track_id'])
        self.assertTrue(first[0]['is_detection'])
        detected = detector.predict.return_value
        detector.predict.return_value = [{'bbox': np.empty((0, 4)), 'det_score': np.empty(0)}]
        self.assertEqual(tracker.update(frame), [])
        detector.predict.return_value = detected
        recovered = tracker.update(frame)
        self.assertEqual(recovered[0]['track_id'], first[0]['track_id'])
        tracker.reset()
        self.assertEqual(tracker.tracker.frame_id, 0)
        self.assertEqual(tracker.tracker.tracks, [])
        self.assertIs(tracker.detector, detector)
        detector.predict.return_value = [{'bbox': np.empty((0, 4)), 'det_score': np.empty(0)}]
        self.assertEqual(tracker.update(frame), [])
