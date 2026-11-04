from contextlib import ExitStack
from types import ModuleType
import time
import unittest
from unittest.mock import Mock, patch

import main


class MainRuntimeTests(unittest.TestCase):
    def test_default_combo_shared_models_and_camera_reconnect_cleanup(self):
        camera_a, camera_b, tracker_a, tracker_b = [Mock() for _ in range(4)]
        scrfd, person_detector, extractor, verifier, vms = [Mock() for _ in range(5)]
        config = {'inference': {'device': 'cpu', 'conf': .4, 'imgsz': 320},
                  'model_detect': {'weight': 'legacy-unused.engine', 'model_name': 'buffalo_s'},
                  'model_extract': {'weight': 'face.onnx'},
                  'person_tracking': {'weight': 'person.pt', 'conf': .15, 'imgsz': 320,
                                      'enable_cmc': False},
                  'runtime': {'processing_fps': 10, 'recognition_threshold': .4}}
        factories = {
            'src.core_ai.camera_reader': {'CameraReader': Mock(side_effect=[camera_a, camera_b])},
            'src.core_ai.detector': {'SCRFDDetector': Mock(return_value=scrfd),
                                     'InferenceWorker': Mock(return_value=person_detector)},
            'src.core_ai.extractor': {'InsightFaceExtractor': Mock(return_value=extractor)},
            'src.core_ai.tracker': {'PersonTracker': Mock(side_effect=[tracker_a, tracker_b])},
            'src.database.verifier': {'Verifier': Mock(return_value=verifier)},
            'src.vms_service.vms_client': {'VMSClient': Mock() },
            'dotenv': {'load_dotenv': Mock()},
        }
        factories['src.vms_service.vms_client']['VMSClient'].from_env.return_value = vms
        modules = {}
        for name, attrs in factories.items():
            modules[name] = ModuleType(name)
            for key, value in attrs.items():
                setattr(modules[name], key, value)
        cameras = [{'id': 1, 'code': 'A', 'rtsp_url': 'fake-a'},
                   {'id': 2, 'code': 'B', 'rtsp_url': 'fake-b'}]
        vms.get_cameras.side_effect = [cameras, cameras, KeyboardInterrupt]
        now = time.monotonic()
        for camera, generations in ((camera_a, [1, 2]), (camera_b, [1, 1])):
            camera.get_sample.side_effect = [
                {'frame': object(), 'generation': generation,
                 'captured_monotonic': now + i*.01} for i, generation in enumerate(generations)]
        visits = []

        def process(code, *args, identity_cache, identity_manager):
            visits.append((code, identity_cache, identity_manager,
                           bool(identity_cache.entries), bool(identity_manager.entries)))
            identity_cache.entries[7] = {'seed': True}
            identity_manager.entries[7] = {'seed': True}
            return {'zones': 1}

        with ExitStack() as stack:
            stack.enter_context(patch.dict('sys.modules', modules))
            stack.enter_context(patch('sys.argv', ['main.py']))
            stack.enter_context(patch.object(main, 'load_config', return_value=config))
            paths = stack.enter_context(patch.object(main, 'model_path', side_effect=lambda value: value))
            stack.enter_context(patch.object(main.os, 'getenv', return_value=None))
            stack.enter_context(patch.object(main, 'process_frame', side_effect=process))
            stack.enter_context(patch.object(main.logging, 'basicConfig'))
            stack.enter_context(patch.object(main.time, 'sleep'))
            main.main()

        factories['src.core_ai.detector']['SCRFDDetector'].assert_called_once_with(
            model_name='buffalo_s', det_size=(320, 320), det_thresh=.4, device='cpu')
        factories['src.core_ai.extractor']['InsightFaceExtractor'].assert_called_once_with(
            'face.onnx', device='cpu')
        factories['src.core_ai.detector']['InferenceWorker'].assert_called_once_with(
            wp='person.pt', conf=.15, imgsz=320, classes=[0], device='cpu')
        self.assertEqual([call.args[0] for call in paths.call_args_list], ['face.onnx', 'person.pt'])
        for call in factories['src.core_ai.tracker']['PersonTracker'].call_args_list:
            self.assertIs(call.kwargs['detector'], person_detector)
        scrfd.warmup.assert_called_once()
        extractor.warmup.assert_called_once()
        person_detector.warmup.assert_called_once()
        self.assertEqual([visit[0] for visit in visits], ['A', 'B', 'A', 'B'])
        self.assertIsNot(visits[0][1], visits[1][1])
        self.assertIsNot(visits[0][2], visits[1][2])
        self.assertEqual(visits[2][3:], (False, False))  # A reconnect clears both states
        self.assertEqual(visits[3][3:], (True, True))  # B keeps its state
        self.assertEqual(tracker_a.reset.call_count, 2)
        tracker_b.reset.assert_called_once()
        camera_a.stop.assert_called_once()
        camera_b.stop.assert_called_once()
        verifier.close.assert_called_once()
        vms.stop.assert_called_once()
