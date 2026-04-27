# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""etlantis.match.semantic — embedding-cosine string matching.

Best-fit when surface-form matching fails despite the strings referring
to the same entity. Cases:

  * Acronyms: ``"USAF"`` ↔ ``"United States Air Force"``.
  * Brand vs legal name: ``"McDonald's"`` ↔ ``"Golden Arches"``.
  * Word-order independence with semantic equivalence:
    ``"acme catering services llc"`` ↔ ``"llc acme catering"``.

Edit-distance (FuzzyMatcher) handles typos and abbreviations but not
the above. SemanticMatcher uses dense sentence-transformer embeddings
+ cosine similarity to capture the semantic relationship.

Optional install — heavy dep tree (torch + sentence-transformers ≈ 1 GB
on disk):

    pip install etlantis[semantic]

Without that extras group, importing `etlantis.match.semantic` raises
ImportError. The `etlantis.match` package's `__init__` catches it so
users who only need exact + fuzzy don't see a hard error.

Performance: GPU dramatically helps. CPU 10K×10K ≈ 20 s; GPU 10K×10K
≈ 1–2 s. Device detection is delegated to `clio.runtime.hardware.
detect_device` so the choice is consistent with the rest of the fleet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from etlantis.match.base import _validate_threshold


class SemanticMatcher:
    """Embedding-cosine matcher backed by sentence-transformers.

    The model is loaded lazily on the first `match` call. Apps that
    want to pay the load cost upfront (so the first user request
    isn't slow) can call `warm_up()` during initialization.

    Args:
        model_name: HuggingFace model identifier. Default
            ``"all-MiniLM-L6-v2"`` — 384-dim, fast (~22 MB on disk),
            good quality. Alternative: ``"all-mpnet-base-v2"`` (slower
            but higher recall on long descriptions).
        device: Override device detection. ``None`` (default) delegates
            to ``clio.runtime.hardware.detect_device`` for fleet-wide
            consistency. Pass ``"cpu"``, ``"cuda"``, or ``"mps"`` to
            force a specific device.
        batch_size: How many strings to embed per forward pass. Larger
            is faster on GPU and rarely a memory issue for the typical
            10K-100K row workloads etlantis sees.

    Threshold semantics: float in [0.0, 1.0] (cosine similarity, both
    embeddings normalized so cosine = dot product). Typical values:
        * 0.85 strict — only strong semantic equivalents match
        * 0.75 moderate — typical business-name matching workload
        * 0.65 lenient — recall-focused (with manual review of edges)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 512,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device_override = device
        self._model: Any = None  # SentenceTransformer; typed Any to avoid
        # forcing the import at decl time.
        self._device_resolved: str | None = None

    def warm_up(self) -> None:
        """Force-load the embedding model now.

        Safe to call repeatedly — the first call loads, subsequent
        calls are no-ops. Useful in long-running adapters where the
        first user request shouldn't pay the model-load latency.
        """
        self._load_model()

    @property
    def device(self) -> str | None:
        """The device this matcher will use (or already loaded onto)."""
        return self._device_resolved or self._device_override

    # ------------------------------------------------------------------
    # Internals: lazy model + embedding
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover — extras enforced at import time
            raise ImportError(
                "SemanticMatcher requires the `semantic` extras group. "
                "Install with: pip install 'etlantis[semantic]'"
            ) from exc

        # Resolve device. clio.runtime.hardware.detect_device returns the
        # same string sentence-transformers expects ('cpu', 'cuda', 'mps').
        # If clio isn't importable for some reason, fall back to letting
        # sentence-transformers do its own auto-detection.
        device = self._device_override
        if device is None:
            try:
                from clio.runtime.hardware import detect_device

                device = detect_device()
            except Exception:  # pragma: no cover — clio is a hard dep but defensive
                device = None
        self._device_resolved = device
        self._model = SentenceTransformer(self.model_name, device=device)

    def _embed(self, texts: Sequence[str]) -> Any:
        """Return a normalized embedding matrix of shape (len(texts), dim).

        `normalize_embeddings=True` means cosine similarity collapses to
        dot product — the matrix multiply at match-time runs faster than
        a full cosine and stays numerically stable.
        """
        self._load_model()
        return self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
        threshold: float = 0.75,
    ) -> dict[int, int]:
        """Best-match each left value against the right list.

        For each left row, returns the right row with the highest
        cosine similarity, provided that score meets or exceeds
        ``threshold``. Returns ``{left_idx: right_idx}`` only for rows
        with an accepting match.

        Args:
            left_values: Primary list. ``None``/empty entries are
                excluded (won't match anything).
            right_values: Enrichment list. Same exclusion rules.
            threshold: Minimum cosine similarity in [0.0, 1.0]. Out of
                range raises ValueError.

        Returns:
            ``{left_idx: right_idx}`` mapping into the original input
            indices (NOT into the post-filter compressed lists).
        """
        _validate_threshold(threshold)
        if not left_values or not right_values:
            return {}

        # Filter empties on both sides while remembering the mapping
        # back to original indices. Same trick as FuzzyMatcher: if we
        # passed empty strings to the embedder they'd produce near-zero
        # vectors which can spuriously match other near-zero vectors at
        # high cosine, masking real candidates.
        left_filtered: list[tuple[int, str]] = [
            (i, v) for i, v in enumerate(left_values) if v and v.strip()
        ]
        right_filtered: list[tuple[int, str]] = [
            (i, v) for i, v in enumerate(right_values) if v and v.strip()
        ]
        if not left_filtered or not right_filtered:
            return {}

        left_indices, left_texts = zip(*left_filtered, strict=True)
        right_indices, right_texts = zip(*right_filtered, strict=True)

        # Embed both sides. Embeddings are normalized so cosine = dot.
        left_emb = self._embed(left_texts)
        right_emb = self._embed(right_texts)

        # Avoid forcing numpy as a hard etlantis dep — it's pulled in
        # via the `semantic` extras (sentence-transformers requires it).
        import numpy as np

        # (n_left, n_right) similarity matrix.
        sim = np.asarray(left_emb) @ np.asarray(right_emb).T

        # Best right index + best score per left row.
        best_right_in_filtered = np.argmax(sim, axis=1)
        best_scores = sim[np.arange(len(left_texts)), best_right_in_filtered]

        matches: dict[int, int] = {}
        for filtered_left_idx, orig_left_idx in enumerate(left_indices):
            score = float(best_scores[filtered_left_idx])
            if score >= threshold:
                filtered_right_idx = int(best_right_in_filtered[filtered_left_idx])
                matches[orig_left_idx] = right_indices[filtered_right_idx]
        return matches

    def get_similarity_scores(
        self,
        left_values: Sequence[str | None],
        right_values: Sequence[str | None],
    ) -> Any:
        """Return the full (n_left × n_right) cosine-similarity matrix.

        Diagnostic helper for threshold tuning. Empty rows are dropped
        from the matrix; the returned shape is `(n_left_filtered,
        n_right_filtered)`. Apps that need the original indices can
        compute the parallel filtered-index lists with the same
        `if v and v.strip()` predicate this function uses.
        """
        if not left_values or not right_values:
            import numpy as np

            return np.zeros((0, 0))
        left_texts = [v for v in left_values if v and v.strip()]
        right_texts = [v for v in right_values if v and v.strip()]
        if not left_texts or not right_texts:
            import numpy as np

            return np.zeros((0, 0))
        left_emb = self._embed(left_texts)
        right_emb = self._embed(right_texts)
        import numpy as np

        return np.asarray(left_emb) @ np.asarray(right_emb).T


__all__ = ["SemanticMatcher"]
