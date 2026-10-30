import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class Verifier:
    def __init__(self, threshold=0.7, store=None):
        self.threshold = float(threshold)
        self.owns_store = store is None
        if store is None:
            from src.database.repository import PGVectorStore
            store = PGVectorStore()
        self.store = store
        self.next_debug = 0.0

    def verify(self, embedding, debug=False):
        embedding = np.asarray(embedding, dtype=np.float32)
        # Khi debug, xem top 3 tối đa mỗi giây; vẫn dùng top 1 để quyết định.
        if debug and time.monotonic() >= self.next_debug:
            candidates = self.store.search_candidates(embedding, limit=3)
            person = candidates[0] if candidates else None
            summary = [(p["ma_nv"], p["ho_va_ten"], round(float(p["similarity"]), 4))
                       for p in candidates]
            logger.info("recognition threshold=%.3f top3(code, name, score)=%s", self.threshold, summary)
            self.next_debug = time.monotonic()+1
        else:
            person = self.store.search(embedding)
        if person is None:
            return {"matched": False, "score_match": None}
        score = float(person["similarity"])
        if not np.isfinite(score):
            raise ValueError("similarity từ database phải là số hữu hạn")
        if score < self.threshold:
            return {"matched": False, "score_match": score}
        return {
            "matched": True,
            "person_id": int(person["stt"]),
            "user_code": person["ma_nv"],
            "user_name": person["ho_va_ten"],
            "department_name": person["phong_ban"],
            "score_match": score,
        }

    def close(self):
        if self.owns_store:
            self.store.close()
