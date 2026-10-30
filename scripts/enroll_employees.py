"""Đọc CSV -> phát hiện một mặt -> tạo embedding -> lưu vào employees."""

import argparse
import csv
import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


def read_employees(csv_path, image_root):
    """image_path tương đối được tính từ image_root, mặc định là root project."""
    employees = []
    seen_codes = set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"user_code", "user_name", "image_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV phải có user_code, user_name, image_path")
        for line_number, row in enumerate(reader, start=2):
            code = (row["user_code"] or "").strip()
            name = (row["user_name"] or "").strip()
            image_path = (row["image_path"] or "").strip()
            if not code or not name or not image_path:
                raise ValueError(f"Dòng {line_number}: thiếu mã, tên hoặc đường dẫn ảnh")
            if code in seen_codes:
                raise ValueError(f"Dòng {line_number}: trùng user_code={code}")
            seen_codes.add(code)
            image_path = Path(image_path)
            if not image_path.is_absolute():
                image_path = Path(image_root) / image_path
            employees.append({"code": code, "name": name, "image_path": image_path})
    if not employees:
        raise ValueError("CSV không có nhân viên")
    return employees


def save_employee(connection, employee, embedding):
    """Một ma_nv có một embedding theo UNIQUE trong sql/init.sql."""
    connection.execute(
        """
        INSERT INTO employees (ma_nv, ho_va_ten, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (ma_nv) DO UPDATE SET
            ho_va_ten = EXCLUDED.ho_va_ten,
            embedding = EXCLUDED.embedding
        """,
        (employee["code"], employee["name"], embedding),
    )
    connection.commit()


def main():
    parser = argparse.ArgumentParser(description="Enroll employee faces from CSV into PostgreSQL")
    parser.add_argument("--csv", type=Path, default=ROOT / "configs/employees.csv")
    parser.add_argument("--image-root", type=Path, default=ROOT, help="Base directory for CSV image paths")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.yml")
    parser.add_argument("--device", help="cpu or GPU index, e.g. 0")
    parser.add_argument("--check", action="store_true", help="Check CSV and image paths only; no DB writes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    employees = read_employees(args.csv, args.image_root)

    if args.check:
        missing = 0
        for employee in employees:
            exists = employee["image_path"].is_file()
            missing += not exists
            logger.info("code=%s image=%s exists=%s", employee["code"], employee["image_path"], exists)
        logger.info("CSV: %s nhân viên, %s ảnh thiếu", len(employees), missing)
        return 1 if missing else 0

    # Cùng model và cách crop/tiền xử lý với main, không dùng model enroll khác.
    import cv2
    import numpy as np
    import psycopg
    from dotenv import load_dotenv
    from main import load_config, model_path
    from src.core_ai.detector import InferenceWorker
    from src.core_ai.extractor import Extractor
    from src.utils import get_connection

    load_dotenv(ROOT / ".env")
    config = load_config(args.config)
    inference = dict(config["inference"])
    if args.device is not None:
        inference["device"] = int(args.device) if args.device.isdigit() else args.device
    detector = InferenceWorker(wp=model_path(
        os.getenv("FACE_DETECTOR_WEIGHTS") or config["model_detect"]["weight"]), **inference)
    extractor = Extractor(model_path(
        os.getenv("FACE_EXTRACTOR_WEIGHTS") or config["model_extract"]["weight"]))
    extractor.warmup()
    min_size = int(config["runtime"].get("min_face_size", 32))
    saved, skipped = 0, 0

    with get_connection() as connection:
        for employee in employees:
            try:
                if not employee["image_path"].is_file():
                    raise ValueError("Không tìm thấy file ảnh")
                frame = cv2.imread(str(employee["image_path"]))
                if frame is None:
                    raise ValueError("Không đọc được ảnh")
                faces = detector.predict([frame])[0]
                if len(faces["bbox"]) != 1:
                    raise ValueError(f"Ảnh phải có đúng 1 khuôn mặt, tìm được {len(faces['bbox'])}")
                height, width = frame.shape[:2]
                x1, y1, x2, y2 = map(int, faces["bbox"][0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2-x1 < min_size or y2-y1 < min_size:
                    raise ValueError("Khuôn mặt quá nhỏ")
                embedding = extractor.extract(frame[y1:y2, x1:x2])
                if embedding.shape != (512,) or not np.isfinite(embedding).all() or not np.any(embedding):
                    raise ValueError("Embedding không hợp lệ")
                save_employee(connection, employee, embedding)
                saved += 1
                logger.info("Đã lưu code=%s name=%s", employee["code"], employee["name"])
            except (ValueError, cv2.error, psycopg.Error) as error:
                connection.rollback()
                skipped += 1
                # Lỗi SQL chỉ log tên loại lỗi để tránh in connection string.
                reason = type(error).__name__ if isinstance(error, psycopg.Error) else str(error)
                logger.warning("Bỏ qua code=%s: %s", employee["code"], reason)

    logger.info("Hoàn tất: đã lưu=%s, bỏ qua=%s", saved, skipped)
    return 1 if skipped else 0


if __name__ == "__main__":
    raise SystemExit(main())
