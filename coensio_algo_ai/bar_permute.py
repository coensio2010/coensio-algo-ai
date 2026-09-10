"""Bar-permutation Monte Carlo (neurotrader888 style).

Preserves OHLC bar shape statistics while destroying temporal order.
Volume is permuted with the body (same shuffle as H/L/C relatives).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def permute_ohlcv(df: pd.DataFrame, *, start_index: int = 0, seed: int | None = None) -> pd.DataFrame:
    """Return a permuted OHLCV copy with the same DatetimeIndex."""
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    need = {"open", "high", "low", "close"}
    missing = need - {c.lower() for c in df.columns}
    if missing:
        raise ValueError(f"permute_ohlcv needs columns {sorted(need)}, missing {sorted(missing)}")

    cols = {c.lower(): c for c in df.columns}
    o = df[cols["open"]].to_numpy(dtype=float)
    h = df[cols["high"]].to_numpy(dtype=float)
    l = df[cols["low"]].to_numpy(dtype=float)
    c = df[cols["close"]].to_numpy(dtype=float)
    if "volume" in cols:
        v = df[cols["volume"]].to_numpy(dtype=float)
    else:
        v = np.ones(len(df), dtype=float)

    n = len(df)
    if n < start_index + 3:
        raise ValueError("not enough bars to permute")

    rng = np.random.default_rng(seed)
    log_o = np.log(np.clip(o, 1e-12, None))
    log_h = np.log(np.clip(h, 1e-12, None))
    log_l = np.log(np.clip(l, 1e-12, None))
    log_c = np.log(np.clip(c, 1e-12, None))

    perm_index = start_index + 1
    r_o = log_o[perm_index:] - log_c[perm_index - 1 : -1]
    r_h = log_h[perm_index:] - log_o[perm_index:]
    r_l = log_l[perm_index:] - log_o[perm_index:]
    r_c = log_c[perm_index:] - log_o[perm_index:]
    v_tail = v[perm_index:].copy()

    idx = np.arange(len(r_o))
    body = rng.permutation(idx)
    gap = rng.permutation(idx)
    r_h, r_l, r_c = r_h[body], r_l[body], r_c[body]
    v_tail = v_tail[body]
    r_o = r_o[gap]

    out_o = np.empty(n)
    out_h = np.empty(n)
    out_l = np.empty(n)
    out_c = np.empty(n)
    out_v = v.copy()

    out_o[:perm_index] = log_o[:perm_index]
    out_h[:perm_index] = log_h[:perm_index]
    out_l[:perm_index] = log_l[:perm_index]
    out_c[:perm_index] = log_c[:perm_index]

    for i, k in enumerate(range(perm_index, n)):
        out_o[k] = out_c[k - 1] + r_o[i]
        out_h[k] = out_o[k] + r_h[i]
        out_l[k] = out_o[k] + r_l[i]
        out_c[k] = out_o[k] + r_c[i]
        out_v[k] = v_tail[i]

    # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    out_h = np.maximum(out_h, np.maximum(out_o, out_c))
    out_l = np.minimum(out_l, np.minimum(out_o, out_c))

    perm = pd.DataFrame(
        {
            "open": np.exp(out_o),
            "high": np.exp(out_h),
            "low": np.exp(out_l),
            "close": np.exp(out_c),
            "volume": out_v,
        },
        index=df.index,
    )
    perm.attrs.update(getattr(df, "attrs", {}) or {})
    return perm
