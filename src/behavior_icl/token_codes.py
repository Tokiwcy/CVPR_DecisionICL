from __future__ import annotations

import itertools
import string
from typing import Any


def make_single_token_codes(tokenizer: Any, count: int) -> list[str]:
    """Return deterministic, readable class codes that each occupy one token."""
    candidates = list(string.ascii_uppercase)
    candidates.extend(
        "".join(chars)
        for chars in itertools.product(string.ascii_uppercase, repeat=2)
    )
    codes: list[str] = []
    token_ids: set[int] = set()
    for candidate in candidates:
        encoded = tokenizer.encode(candidate, add_special_tokens=False)
        if len(encoded) != 1 or encoded[0] in token_ids:
            continue
        codes.append(candidate)
        token_ids.add(encoded[0])
        if len(codes) == count:
            return codes
    raise ValueError(f"Could not construct {count} unique single-token class codes")
