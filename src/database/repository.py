import psycopg
import numpy as np
from src.utils import get_connection

class PGVectorStore:
    def __init__(self):
        self.conn = get_connection()

    def insert_embedding(self,
                         embedding: np.ndarray,
                         ma_nv: str,
                         name: str,
                         dob:str=None,
                         det_name:str=None,
                         chuc_vu:str=None,
                         sdt:str=None,
                         email:str=None):
        insert_query = """
        INSERT INTO employees (ma_nv, ho_va_ten, embedding)
        VALUES(%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(insert_query, (ma_nv, name, embedding))
            self.conn.commit()
        except psycopg.Error as e:
            print(f"DB error: {e}")
            self.conn.rollback()
            raise

    def search(self, embedding: np.ndarray):
        """Tìm cosine similarity cao nhất; giữ nguyên giá trị embedding đầu vào."""
        candidates = self.search_candidates(embedding, limit=1)
        return candidates[0] if candidates else None

    def search_candidates(self, embedding: np.ndarray, limit=3):
        """Lấy các ứng viên gần nhất để kiểm tra thứ hạng và điểm similarity."""
        query = """
        SELECT stt, ma_nv, ho_va_ten, phong_ban,
               1 - (embedding <=> %s) AS similarity
        FROM employees 
        ORDER BY embedding <=> %s LIMIT %s;
        """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, (embedding, embedding, limit))
                people = cursor.fetchall()
            self.conn.commit()
            return people
        except psycopg.Error as e:
            self.conn.rollback()
            raise

    def get_all(self):
        query = """
                SELECT * FROM employees
                """
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query)
                return cursor.fetchall()
        except psycopg.Error as e:
            print(f"DB error: {e}")
            self.conn.rollback()
            raise

    def close(self):
        self.conn.close()

if __name__ == "__main__":
    import pandas as pd
    import os
    import cv2
    from src.core_ai.extractor import Extractor
    from src.core_ai.detector import SCRFDDetector

    store = PGVectorStore()
    detector = SCRFDDetector()
    extractor = Extractor(r"D:\private\face-recognition-system\checkpoints\test_SGD\checkpoints\face_embedding_model.onnx")
    csv_path = r"D:\private\task2_atin\configs\employees.csv"
    df = pd.read_csv(csv_path)
    paths = df["image_path"].tolist()
    user_codes = df["user_code"].tolist()
    user_names = df["user_name"].tolist()
    cnt = 0
    for path, code, name in zip(paths, user_codes, user_names):
        frame = cv2.imread(os.path.join(r"D:\private\task2_atin", path))
        results = detector.detect_one_face(frame)
        print(results)
        x1, y1, x2, y2 = map(int, list(results["bbox"][0]))
        cropped = frame[y1:y2, x1:x2]
        embedding = extractor.extract(cropped)
        store.insert_embedding(embedding, code, name)
        cnt += 1
    print(cnt)

