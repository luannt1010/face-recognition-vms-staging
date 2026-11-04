from contextlib import ExitStack
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import MagicMock, Mock, patch

from scripts.enroll_employees import read_employees, save_employee


class EnrollTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('cv2', 'dotenv', 'psycopg', 'pgvector')),
                         'Enrollment requires OpenCV, dotenv and database libraries')
    def test_scrfd_enrollment_skips_invalid_faces_and_aligns_valid_face_on_original_frame(self):
        import numpy as np
        from scripts import enroll_employees as enroll

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = np.array([[35, 25], [65, 25], [50, 40], [35, 55], [65, 55]], dtype=np.float32)
        detector, extractor, connection = Mock(), Mock(), Mock()
        empty = {'bbox': np.empty((0, 4)), 'landmarks': np.empty((0, 5, 2)), 'det_score': np.empty(0)}
        detector.detect_many_faces.side_effect = [
            empty, empty, {'bbox': [[0, 0, 40, 40], [50, 0, 90, 40]]},
            {'bbox': [[0, 0, 10, 10]]},
            {'bbox': [[20, 10, 80, 70]], 'landmarks': landmarks[None]}]
        embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
        extractor.extract.return_value = embedding
        database = MagicMock()
        database.__enter__.return_value = connection
        detector_module, extractor_module = ModuleType('src.core_ai.detector'), ModuleType('src.core_ai.extractor')
        detector_module.SCRFDDetector = Mock(return_value=detector)
        extractor_module.InsightFaceExtractor = Mock(return_value=extractor)
        config = {'model_detect': {'model_name': 'buffalo_s'}, 'model_extract': {'weight': 'face.onnx'},
                  'inference': {'imgsz': 320, 'conf': .4, 'device': 0}, 'runtime': {'min_face_size': 32}}
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder, ExitStack() as stack:
            root = Path(folder)
            (root / 'photo.jpg').write_bytes(b'placeholder')
            csv_path = root / 'employees.csv'
            csv_path.write_text('user_code,user_name,image_path\n' + ''.join(
                f'{i},Test {i},photo.jpg\n' for i in range(4)), encoding='utf-8')
            stack.enter_context(patch.dict('sys.modules', {'src.core_ai.detector': detector_module,
                                                          'src.core_ai.extractor': extractor_module}))
            stack.enter_context(patch('sys.argv', ['enroll', '--csv', str(csv_path),
                                                  '--image-root', str(root), '--device', 'cpu']))
            stack.enter_context(patch('main.load_config', return_value=config))
            stack.enter_context(patch('main.model_path', side_effect=lambda value: value))
            stack.enter_context(patch('cv2.imread', return_value=frame))
            stack.enter_context(patch('src.utils.get_connection', return_value=database))
            stack.enter_context(patch.object(enroll.os, 'getenv', return_value=None))
            stack.enter_context(patch.object(enroll.logging, 'basicConfig'))
            save = stack.enter_context(patch.object(enroll, 'save_employee'))
            with self.assertLogs(enroll.logger, level='INFO') as logs:
                self.assertEqual(enroll.main(), 1)
            output = ' '.join(logs.output)
            self.assertIn('tìm được 0', output)
            self.assertIn('tìm được 2', output)
            self.assertIn('Khuôn mặt quá nhỏ', output)
        save.assert_called_once()
        self.assertEqual(save.call_args.args[1]['code'], '3')
        self.assertIs(save.call_args.args[2], embedding)
        self.assertEqual(connection.rollback.call_count, 3)
        extractor.extract.assert_called_once()
        self.assertIs(extractor.extract.call_args.args[0], frame)
        np.testing.assert_array_equal(extractor.extract.call_args.kwargs['kps'], landmarks)
        self.assertTrue(extractor.extract.call_args.kwargs['align'])
        detector_module.SCRFDDetector.assert_called_once_with(
            model_name='buffalo_s', det_size=(320, 320), det_thresh=.4, device='cpu')
        extractor_module.InsightFaceExtractor.assert_called_once_with('face.onnx', device='cpu')
        detector.warmup.assert_called_once()
        extractor.warmup.assert_called_once()

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'Padding requires OpenCV')
    def test_portrait_padding_maps_face_and_landmarks_back_to_original_frame(self):
        import numpy as np
        from scripts.enroll_employees import detect_enrollment_faces

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        landmarks = np.array([[35, 25], [65, 25], [50, 40], [35, 55], [65, 55]], dtype=np.float32)
        detector = Mock()
        detector.detect_many_faces.side_effect = [
            {'bbox': np.empty((0, 4))},
            {'bbox': np.array([[45, 35, 105, 95]], dtype=np.float32),
             'landmarks': (landmarks+25)[None], 'det_score': np.array([.84])}]
        faces = detect_enrollment_faces(detector, frame)
        self.assertIs(detector.detect_many_faces.call_args_list[0].args[0], frame)
        self.assertEqual(detector.detect_many_faces.call_args_list[1].args[0].shape, (150, 150, 3))
        np.testing.assert_array_equal(faces['bbox'][0], [20, 10, 80, 70])
        np.testing.assert_array_equal(faces['landmarks'][0], landmarks)
        self.assertAlmostEqual(float(faces['det_score'][0]), .84)

    def test_csv_bom_names_and_relative_image_root(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            csv_path = Path(folder) / 'employees.csv'
            csv_path.write_text('user_code,user_name,image_path\n001,Nguyễn Văn A,pics/a.jpg\n', encoding='utf-8-sig')
            employee = read_employees(csv_path, Path(folder))[0]
            self.assertEqual(employee['code'], '001')
            self.assertEqual(employee['name'], 'Nguyễn Văn A')
            self.assertEqual(employee['image_path'], Path(folder) / 'pics/a.jpg')

    def test_duplicate_codes_rejected(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            csv_path = Path(folder) / 'employees.csv'
            csv_path.write_text('user_code,user_name,image_path\n1,A,a.jpg\n1,B,b.jpg\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'user_code=1'):
                read_employees(csv_path, folder)

    def test_save_updates_by_code_without_removing_other_fields(self):
        connection = Mock()
        embedding = object()
        save_employee(connection, {'code': '001', 'name': 'A'}, embedding)
        sql, params = connection.execute.call_args.args
        self.assertIn('ON CONFLICT (ma_nv) DO UPDATE', sql)
        self.assertNotIn('DELETE', sql)
        self.assertEqual(params, ('001', 'A', embedding))
        connection.commit.assert_called_once()
