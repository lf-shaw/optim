#!/usr/bin/env python3
"""Require PIQP 0.6.4+ and report the effective inequality representation."""

from __future__ import annotations

import json
import sys

from factor_qcqp import piqp_distribution_version, resolve_piqp_inequality_form


MINIMUM_VERSION = (0, 6, 4)


def _release_tuple(value: str | None) -> tuple[int, int, int] | None:
    if value is None:
        return None
    core = value.split("+", 1)[0]
    parts = core.split(".")
    if len(parts) < 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def main() -> int:
    version = piqp_distribution_version()
    resolved = resolve_piqp_inequality_form("auto")
    payload = {
        "piqp_version": version,
        "requested_inequality_form": "auto",
        "resolved_inequality_form": resolved,
        "minimum_supported_version": "0.6.4",
        "compact_active": resolved == "compact",
    }
    print(json.dumps(payload, sort_keys=True))
    if (_release_tuple(version) or (0, 0, 0)) < MINIMUM_VERSION:
        print(
            f"PIQP >= 0.6.4 is required; found {version!r}.",
            file=sys.stderr,
        )
        return 1
    if resolved != "compact":
        print(
            "PIQP auto mode must resolve to compact.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
