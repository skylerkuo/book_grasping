import re
from typing import Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util


class TextSimilarityMatcher:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"[Info] Loading sentence embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text or ""
        text = text.strip()
        text = re.sub(r"\s+", "", text)
        return text

    def get_best_window_similarity(
        self,
        source_text: str,
        target_text: str,
        padding: int = 3,
        bonus_weight: float = 1,
    ) -> Tuple[float, str]:
        source_text = self.normalize_text(source_text)
        target_text = self.normalize_text(target_text)

        if not source_text or not target_text:
            return 0.0, ""

        window_size = max(len(target_text) + padding, len(target_text))

        if len(source_text) <= window_size:
            segments = [source_text]
        else:
            segments = [
                source_text[i : i + window_size]
                for i in range(len(source_text) - window_size + 1)
            ]

        if not segments:
            return 0.0, ""

        target_embedding = self.model.encode(target_text, convert_to_tensor=True)
        segment_embeddings = self.model.encode(segments, convert_to_tensor=True)
        semantic_scores = util.cos_sim(segment_embeddings, target_embedding).flatten()

        target_chars = set(target_text)
        final_scores = []

        for idx, segment in enumerate(segments):
            overlap_count = sum(1 for ch in target_chars if ch in segment)
            overlap_ratio = overlap_count / len(target_chars) if target_chars else 0.0
            bonus = overlap_ratio * bonus_weight
            final_scores.append(float(semantic_scores[idx].item()) + bonus)

        best_idx = int(np.argmax(final_scores))
        return float(final_scores[best_idx]), segments[best_idx]