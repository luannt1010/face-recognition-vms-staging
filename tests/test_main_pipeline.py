from datetime import datetime, timezone
from unittest import TestCase
from unittest.mock import MagicMock, Mock

from main import process_frame, recognize_faces


class PipelineTests(TestCase):
    def test_pipeline_order_and_identity_by_track_id(self):
        frame = MagicMock()
        frame.shape = (100, 100, 3)
        sample = {"frame": frame, "captured_monotonic": 12,
                  "captured_at": datetime.now(timezone.utc)}
        tracks = [{"track_id": 8, "bbox": [0, 0, 40, 40], "is_detection": True},
                  {"track_id": 3, "bbox": [50, 50, 90, 90], "is_detection": True}]
        detector, tracker, extractor, verifier, processor, vms = [Mock() for _ in range(6)]
        order = Mock()
        for name, obj in (("detector", detector), ("tracker", tracker), ("extractor", extractor),
                          ("verifier", verifier), ("processor", processor), ("vms", vms)):
            order.attach_mock(obj, name)
        detector.predict.return_value = [{"bbox": [[0, 0, 40, 40], [50, 50, 90, 90]]}]
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
                      vms, min_face_size=32, upload_url=None)
        self.assertEqual([c[0] for c in order.mock_calls], [
            "detector.predict", "tracker.update", "extractor.extract", "verifier.verify",
            "extractor.extract", "verifier.verify", "vms.get_zones", "processor.process",
            "vms.publish_bbox", "vms.publish_face", "vms.publish_face"])
        published = vms.publish_face.call_args_list
        self.assertFalse(published[0].args[2]["matched"])
        self.assertEqual(published[1].args[2]["user_code"], "A")
        self.assertNotIn("person_id", published[1].args[2])
        self.assertEqual(published[1].kwargs["event_time"], sample["captured_at"])

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
