from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, Mock

import numpy as np

from main import process_frame, recognize_faces
from src.core_ai.identity_cache import IdentityCache
from src.core_ai.identity_manager import IdentityManager


class PipelineTests(TestCase):
    def test_pipeline_order_and_identity_by_track_id(self):
        frame = MagicMock()
        frame.shape = (100, 100, 3)
        sample = {"frame": frame, "captured_monotonic": 12,
                  "captured_at": datetime.now(timezone.utc)}
        tracks = [{"track_id": 8, "bbox": [0, 0, 45, 80], "is_detection": True},
                  {"track_id": 3, "bbox": [50, 20, 100, 100], "is_detection": True}]
        detector, tracker, extractor, verifier, processor, vms = [Mock() for _ in range(6)]
        order = Mock()
        for name, obj in (("detector", detector), ("tracker", tracker), ("extractor", extractor),
                          ("verifier", verifier), ("processor", processor), ("vms", vms)):
            order.attach_mock(obj, name)
        detector.detect_many_faces.return_value = {
            "bbox": [[0, 0, 40, 40], [50, 25, 90, 65]],
            "landmarks": np.zeros((2, 5, 2)), "det_score": [.9, .8]}
        tracker.update.return_value = tracks
        extractor.extract.return_value = "embedding"
        verifier.verify.side_effect = [{"matched": True, "user_code": "A", "person_id": 99},
                                       {"matched": False, "score_match": .2}]
        # Đảo thứ tự event để đảm bảo ghép identity theo ID.
        processor.process.return_value = [
            {"track_id": 3, "zone_id": "z", "direction": "OUT"},
            {"track_id": 8, "zone_id": "z", "direction": "IN"}]
        vms.get_zones.return_value = []
        process_frame("CAM", sample, tracker, detector, extractor, verifier, processor,
                      vms, min_face_size=32, upload_url=None, identity_manager=IdentityManager())
        self.assertEqual([c[0] for c in order.mock_calls], [
            "tracker.update", "detector.detect_many_faces", "vms.get_zones", "processor.process",
            "extractor.extract", "verifier.verify", "extractor.extract", "verifier.verify",
            "vms.publish_bbox", "vms.publish_face", "vms.publish_face"])
        tracker.update.assert_called_once_with(frame)
        self.assertTrue(extractor.extract.call_args.kwargs['align'])
        self.assertIs(extractor.extract.call_args.args[0], frame)
        self.assertEqual(vms.publish_bbox.call_args.kwargs['detection_class'], 'person')
        self.assertIs(vms.publish_bbox.call_args.args[1], tracks)
        published = vms.publish_face.call_args_list
        self.assertFalse(published[0].args[2]["matched"])
        self.assertEqual(published[1].args[2]["user_code"], "A")
        self.assertNotIn("person_id", published[1].args[2])
        self.assertEqual(published[1].kwargs["event_time"], sample["captured_at"])

    def test_person_event_keeps_identity_when_face_is_missing(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector, tracker, extractor, verifier, processor, vms = [Mock() for _ in range(6)]
        tracks = [{'track_id': 7, 'bbox': [10, 0, 80, 100], 'is_detection': True}]
        tracker.update.return_value = tracks
        detector.detect_many_faces.return_value = {
            'bbox': [[20, 10, 60, 50]], 'landmarks': np.zeros((1, 5, 2)), 'det_score': [.9]}
        verifier.verify.return_value = {'matched': True, 'user_code': 'A'}
        processor.process.return_value = []
        vms.get_zones.return_value = []
        manager, cache = IdentityManager(), IdentityCache()
        sample = {'frame': frame, 'captured_monotonic': 0, 'captured_at': 'timestamp'}
        process_frame('CAM', sample, tracker, detector, extractor, verifier, processor, vms,
                      32, None, identity_cache=cache, identity_manager=manager)
        detector.detect_many_faces.return_value = {
            'bbox': [], 'landmarks': np.empty((0, 5, 2)), 'det_score': []}
        processor.process.return_value = [{'track_id': 7, 'zone_id': 'z', 'direction': 'IN'}]
        sample['captured_monotonic'] = 1
        status = process_frame('CAM', sample, tracker, detector, extractor, verifier, processor, vms,
                               32, None, identity_cache=cache, identity_manager=manager)
        self.assertEqual(vms.publish_face.call_args.args[2]['user_code'], 'A')
        self.assertEqual(vms.publish_bbox.call_args.kwargs['identities'][7]['user_code'], 'A')
        self.assertEqual(status['valid_faces'], 0)
        self.assertEqual(status['tracks'], 1)
        extractor.extract.assert_called_once()

    def test_person_without_face_still_publishes_unknown_event(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detector, tracker, extractor, verifier, processor, vms = [Mock() for _ in range(6)]
        tracker.update.return_value = [
            {'track_id': 7, 'bbox': [10, 0, 80, 100], 'is_detection': True}]
        detector.detect_many_faces.return_value = {
            'bbox': [], 'landmarks': np.empty((0, 5, 2)), 'det_score': []}
        processor.process.return_value = [{'track_id': 7, 'zone_id': 'z', 'direction': 'IN'}]
        vms.get_zones.return_value = []
        process_frame('CAM', {'frame': frame, 'captured_monotonic': 0, 'captured_at': 'timestamp'},
                      tracker, detector, extractor, verifier, processor, vms, 32, None,
                      identity_manager=IdentityManager())
        self.assertFalse(vms.publish_face.call_args.args[2]['matched'])
        extractor.extract.assert_not_called()

    def test_small_and_predicted_faces_are_skipped(self):
        frame = MagicMock()
        frame.shape = (100, 100, 3)
        tracks = [{"track_id": 1, "bbox": [0, 0, 10, 10], "is_detection": True},
                  {"track_id": 2, "bbox": [0, 0, 50, 50], "is_detection": False},
                  {"track_id": 3, "bbox": [90, 90, 120, 120], "is_detection": True}]
        extractor, verifier = Mock(), Mock()
        self.assertEqual(recognize_faces(frame, tracks, extractor, verifier, 32), ([], {}))
        extractor.extract.assert_not_called()
        verifier.verify.assert_not_called()
