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

"""etlantis.config.env_substitute — environment-variable substitution.

Recursively walks a JSON/YAML-loaded structure (dict/list/str) and replaces
${VAR} or ${VAR:default} tokens in string values with the corresponding
environment variable value. Lifted from the rvmaps prototype loader pattern.

The substitution is intentionally simple — no shell-style expansion, no
arithmetic, no command interpolation. Just literal env-var lookup with an
optional inline default.

Usage:
    from clio.config.env_substitute  # noqa, just for the example
    from etlantis.config.env_substitute import substitute

    raw = {"api_key": "${OPENAI_API_KEY}", "timeout": 30}
    resolved = substitute(raw)
    # resolved["api_key"] is now os.environ["OPENAI_API_KEY"], or "" if unset.

    raw_with_default = {"region": "${AWS_REGION:us-east-1}"}
    resolved = substitute(raw_with_default)
    # resolved["region"] is os.environ["AWS_REGION"] if set, else "us-east-1".
"""

from __future__ import annotations

import os
import re
from typing import Any

# ${VAR} or ${VAR:default}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def substitute(obj: Any) -> Any:
    """Recursively substitute ${VAR} and ${VAR:default} env vars in strings.

    Walks dict / list / str inputs. Non-string scalars (int / float / bool /
    None) pass through unchanged. Substitution happens token-by-token within
    a string — multiple env vars can appear in the same string and each is
    resolved independently.

    Args:
        obj: Object to walk. dict / list / str / scalar.

    Returns:
        New object with substitutions applied. Original is not mutated.
    """
    if isinstance(obj, str):
        return _ENV_VAR_PATTERN.sub(_replace_var, obj)
    if isinstance(obj, dict):
        return {k: substitute(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(item) for item in obj]
    return obj


def _replace_var(match: re.Match[str]) -> str:
    """Resolve a single ${VAR} or ${VAR:default} match against os.environ.

    If the env var is unset and no inline default is provided, returns
    empty string. This matches the rvmaps prototype behavior; callers that
    need stricter "must be set" semantics should validate post-substitution
    via the pydantic schema.
    """
    var_name = match.group(1)
    default_value = match.group(2) or ""
    return os.environ.get(var_name, default_value)
