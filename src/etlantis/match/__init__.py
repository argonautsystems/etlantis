# Copyright 2026 etlantis Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License").
# See LICENSE in the repository root for full terms.

"""etlantis.match — pluggable string matching strategies.

Three matchers ship under the same Matcher Protocol:

    ExactMatcher      O(n+m) hash-table lookup of normalized strings.
                      Right tool for license numbers, IDs, codes — any
                      field that should match exactly after normalization.
                      v0.2.0+.

    FuzzyMatcher      Edit-distance scoring via rapidfuzz. Right tool for
                      business names, addresses, anywhere typos and
                      abbreviations are expected. v0.2.0+.

    SemanticMatcher   Embedding-cosine matching for cases where the
                      surface forms diverge but meaning is shared
                      (``"MCDONALDS"`` ↔ ``"GOLDEN ARCHES"``). Optional
                      install: `pip install etlantis[semantic]`. v0.2.1+.

All three accept the same ``match(left, right, threshold)`` shape and
return ``dict[left_idx -> right_idx]``. Concrete matchers can be passed
interchangeably wherever a Matcher Protocol is expected.
"""

from etlantis.match.base import Matcher, NormalizeFn, default_normalize
from etlantis.match.exact import ExactMatcher
from etlantis.match.fuzzy import FuzzyMatcher

__all__ = [
    "Matcher",
    "NormalizeFn",
    "default_normalize",
    "ExactMatcher",
    "FuzzyMatcher",
]

# SemanticMatcher is optional — requires the `semantic` extras dep
# group (sentence-transformers + torch). Apps that don't need it skip
# the install and never see this import error.
try:
    from etlantis.match.semantic import SemanticMatcher  # noqa: F401

    __all__.append("SemanticMatcher")
except ImportError:  # pragma: no cover — only triggers when extras absent
    pass
