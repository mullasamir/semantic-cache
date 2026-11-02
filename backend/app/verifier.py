import threading


class Verifier:
    """Second-stage check for borderline similarity scores."""

    def __init__(self, settings, llm):
        self.s = settings
        self.llm = llm
        self._ce = None
        self._lock = threading.Lock()

    def _cross_encoder(self):
        with self._lock:
            if self._ce is None:
                from sentence_transformers import CrossEncoder

                self._ce = CrossEncoder(self.s.cross_encoder_model)
            return self._ce

    def approve(self, new_query: str, cached_query: str) -> bool:
        mode = self.s.verify_mode
        if mode == "crossencoder":
            score = float(self._cross_encoder().predict([(new_query, cached_query)])[0])
            return score >= self.s.cross_encoder_threshold
        if mode == "llm":
            return self.llm.judge_same(new_query, cached_query)
        return True
