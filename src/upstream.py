"""Import the unmodified upstream code of Wang & Wang (2026).

The upstream repository (github.com/gagawjbytw/hedging_rough_hawkes_heston) is
cloned into ``external/`` and must not be edited. Its scripts import each other
as top-level modules, so we put its ``numerics/`` folder on ``sys.path`` and
re-export the two modules we build on:

* ``base``    -> numerics/original_market_stability.py
                 (model parameters, Riccati--Volterra solver, market simulation)
* ``hedging`` -> numerics/hedging_performance_experiment.py
                 (terminal-loss evaluation of a single hedge)

Importing these modules has no side effects (no files are written).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_NUMERICS = (
    PROJECT_ROOT / "external" / "hedging_rough_hawkes_heston" / "numerics"
)

if not UPSTREAM_NUMERICS.is_dir():
    raise ImportError(
        f"upstream code not found at {UPSTREAM_NUMERICS}; clone "
        "https://github.com/gagawjbytw/hedging_rough_hawkes_heston into external/"
    )
if str(UPSTREAM_NUMERICS) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_NUMERICS))

import original_market_stability as base  # noqa: E402
import hedging_performance_experiment as hedging  # noqa: E402

__all__ = ["base", "hedging", "PROJECT_ROOT", "UPSTREAM_NUMERICS"]
