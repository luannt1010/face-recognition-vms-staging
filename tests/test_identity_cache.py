from unittest import TestCase
from unittest.mock import MagicMock, Mock

from main import recognize_faces, process_frame
from src.core_ai.identity_cache import IdentityCache


class CacheTests(TestCase):
    def setUp(self):
        self.frame = MagicMock()
        self.frame.shape = (100, 100, 3)
        self.tracks = [{'track_id': 1, 'bbox': [20, 20, 70, 70], 'is_detection': True}]
        self.extractor, self.verifier = Mock(), Mock()
        self.verifier.verify.side_effect = [
            {'matched': True, 'user_code': 'A'}, {'matched': False, 'score_match': .2},
            {'matched': True, 'user_code': 'B'}]
        self.cache = IdentityCache(interval=.5, ttl=2)

    def recognize(self, now, force_ids=(), tracks=None):
        return recognize_faces(self.frame, self.tracks if tracks is None else tracks,
                               self.extractor, self.verifier, 32,
                               cache=self.cache, now=now, force_ids=force_ids)[1]

    def test_reuses_identity_until_interval_then_refreshes(self):
        self.assertEqual(self.recognize(0)[1]['user_code'], 'A')
        self.assertEqual(self.recognize(.1)[1]['user_code'], 'A')
        self.extractor.extract.assert_called_once()
        self.assertFalse(self.recognize(.5)[1]['matched'])
        self.assertFalse(self.recognize(.6)[1]['matched'])
        self.assertEqual(self.extractor.extract.call_count, 2)
        self.assertEqual(self.verifier.verify.call_count, 2)

    def test_event_forces_current_frame_but_not_duplicate_extract(self):
        self.recognize(0)
        self.recognize(.1)
        self.assertFalse(self.recognize(.1, force_ids={1})[1]['matched'])
        self.recognize(.1, force_ids={1})
        self.assertEqual(self.extractor.extract.call_count, 2)

    def test_expiry_reset_and_camera_isolation(self):
        self.recognize(0)
        self.recognize(3)
        self.assertEqual(self.extractor.extract.call_count, 2)
        self.cache.clear()
        self.recognize(3.1)
        self.assertEqual(self.extractor.extract.call_count, 3)
        other_camera = IdentityCache()
        self.assertIsNone(other_camera.get(1, 3.1))

    def test_predictions_do_not_keep_cache_alive(self):
        self.recognize(0)
        predicted = [dict(self.tracks[0], is_detection=False)]
        self.recognize(1, tracks=predicted)
        self.recognize(2.1, tracks=predicted)
        self.assertEqual(self.cache.entries, {})

    def test_event_publish_uses_fresh_identity(self):
        self.recognize(0)
        detector, tracker, processor, vms = Mock(), Mock(), Mock(), Mock()
        detector.predict.return_value = [{'bbox': [[20, 20, 70, 70]]}]
        tracker.update.return_value = self.tracks
        processor.process.return_value = [{'track_id': 1, 'zone_id': 'z', 'direction': 'IN'}]
        vms.get_zones.return_value = []
        sample = {'frame': self.frame, 'captured_monotonic': .1, 'captured_at': 'timestamp'}
        process_frame('CAM', sample, tracker, detector, self.extractor, self.verifier,
                      processor, vms, 32, None, identity_cache=self.cache)
        self.assertFalse(vms.publish_face.call_args.args[2]['matched'])
        self.assertFalse(vms.publish_bbox.call_args.kwargs['identities'][1]['matched'])
        self.assertEqual(self.extractor.extract.call_count, 2)
