from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from scripts.enroll_employees import read_employees, save_employee


class EnrollTests(unittest.TestCase):
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
