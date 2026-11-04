from contextlib import ExitStack
import importlib.util
import time
from types import ModuleType
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('dotenv'),
                     'Demo requires OpenCV and python-dotenv')
class DemoTests(unittest.TestCase):
    def run_fake_demo(self, fail=False, hide_face=False):
        import numpy as np
        from src.core_ai import camera_reader as demo

        detector, tracker, extractor, verifier, camera = [Mock() for _ in range(5)]
        landmarks = np.array([[40, 20], [60, 20], [50, 30], [40, 40], [60, 40]], dtype=np.float32)
        first_faces = {'bbox': np.array([[30, 10, 70, 50]]), 'det_score': np.array([.9]),
                       'landmarks': landmarks[None]}
        next_faces = {'bbox': np.array([[30, 50, 70, 90]]), 'det_score': np.array([.9]),
                      'landmarks': (landmarks+40)[None]}
        if hide_face:
            next_faces = {'bbox': np.empty((0, 4)), 'det_score': np.empty(0),
                          'landmarks': np.empty((0, 5, 2))}
        detector.detect_many_faces.side_effect = [first_faces, next_faces, first_faces]
        if fail:
            detector.detect_many_faces.side_effect = RuntimeError('inference failed')
        tracker.update.side_effect = [
            [{'track_id': 1, 'bbox': [20, 0, 80, 80], 'is_detection': True}],
            [{'track_id': 1, 'bbox': [20, 40, 80, 100], 'is_detection': True}],
            [{'track_id': 1, 'bbox': [20, 0, 80, 80], 'is_detection': True}]]
        verifier.verify.side_effect = [{'matched': True, 'user_name': 'Test', 'score_match': .9},
                                      {'matched': True, 'user_name': 'Test', 'score_match': .9},
                                      {'matched': True, 'user_name': 'Test', 'score_match': .9}]
        now = time.monotonic()
        camera.get_sample.side_effect = [
            {'frame': np.zeros((100, 100, 3), dtype=np.uint8),
             'captured_monotonic': now, 'generation': 1},
            {'frame': np.zeros((100, 100, 3), dtype=np.uint8),
             'captured_monotonic': now+.01, 'generation': 1},
            {'frame': np.zeros((100, 100, 3), dtype=np.uint8),
             'captured_monotonic': now+.02, 'generation': 1}]
        config = {'runtime': {'processing_fps': 1000, 'recognition_threshold': .4, 'event_cooldown': 0},
                  'inference': {}, 'model_detect': {'weight': 'fake.pt'},
                  'model_extract': {'weight': 'fake.onnx'}}
        modules = {}
        for name, class_name, obj in [
            ('src.core_ai.detector', 'SCRFDDetector', detector),
            ('src.core_ai.tracker', 'PersonTracker', tracker),
            ('src.core_ai.extractor', 'InsightFaceExtractor', extractor),
            ('src.database.verifier', 'Verifier', verifier)]:
            module = ModuleType(name)
            setattr(module, class_name, Mock(return_value=obj))
            modules[name] = module

        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', modules))
            stack.enter_context(patch('main.load_config', return_value=config))
            stack.enter_context(patch('main.model_path', side_effect=lambda value: value))
            stack.enter_context(patch.object(demo, 'CameraReader', return_value=camera))
            printed = stack.enter_context(patch('builtins.print'))
            polygon_draw = stack.enter_context(patch.object(demo.cv2, 'polylines', wraps=demo.cv2.polylines))
            for function in ('namedWindow', 'imshow', 'destroyAllWindows'):
                stack.enter_context(patch.object(demo.cv2, function))
            stack.enter_context(patch.object(demo.cv2, 'getWindowProperty', return_value=1))
            stack.enter_context(patch.object(demo.cv2, 'waitKey', side_effect=[0, 0, 0, ord('q')]))
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                    demo.run_demo(rtsp_url='fake', device='cpu')
            else:
                demo.run_demo(rtsp_url='fake', device='cpu')
                self.assertEqual(printed.call_count, 2)
                self.assertTrue(printed.call_args_list[0].args[0].startswith('[IN] track=1'))
                self.assertTrue(printed.call_args_list[1].args[0].startswith('[OUT] track=1'))
                self.assertIn('zone=demo-polygon', printed.call_args.args[0])
                self.assertTrue(polygon_draw.call_args.args[2])  # closed polygon
                self.assertIn('name=Test', printed.call_args_list[0].args[0])
                self.assertEqual(extractor.extract.call_count, 2 if hide_face else 3)
                for call in extractor.extract.call_args_list:
                    self.assertEqual(call.args[0].shape, (100, 100, 3))  # full frame for alignment
                    self.assertTrue(call.kwargs['align'])
                if not hide_face:
                    np.testing.assert_array_equal(extractor.extract.call_args_list[1].args[1], landmarks+40)
            modules['src.core_ai.detector'].SCRFDDetector.assert_called_once_with(
                model_name='buffalo_s', det_size=(640, 640), det_thresh=.5, device='cpu')
            modules['src.core_ai.extractor'].InsightFaceExtractor.assert_called_once_with(
                'fake.onnx', device='cpu')
            detector.warmup.assert_called_once()
            extractor.warmup.assert_called_once()
            tracker.warmup.assert_called_once()
            tracker.reset.assert_called_once()  # first stream generation
            modules['src.core_ai.tracker'].PersonTracker.assert_called_once_with(
                model_path='models/detector/yolo11n.pt', fps=1000, device='cpu')
        camera.stop.assert_called_once()
        verifier.close.assert_called_once()

    def test_local_demo_emits_in_and_closes_resources(self):
        self.run_fake_demo()

    def test_inference_error_still_closes_resources(self):
        self.run_fake_demo(fail=True)

    def test_person_crossing_keeps_name_without_visible_face(self):
        self.run_fake_demo(hide_face=True)
