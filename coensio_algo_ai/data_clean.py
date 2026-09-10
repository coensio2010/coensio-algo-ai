"""OHLC sanitization (drop / repair malformed bars)."""

from __future__ import annotations


def clean_ohlc_df(df, max_jump: float = 5.0):
    """Return (clean_df, n_fixed_bars). Non-destructive."""
    o, h, l, c, fixed = clean_ohlc_arrays(
        df["open"].to_numpy(dtype=float).tolist(),
        df["high"].to_numpy(dtype=float).tolist(),
        df["low"].to_numpy(dtype=float).tolist(),
        df["close"].to_numpy(dtype=float).tolist(),
        max_jump=max_jump,
    )
    if fixed == 0:
        return df, 0
    out = df.copy()
    out["open"] = o
    out["high"] = h
    out["low"] = l
    out["close"] = c
    return out, fixed


def clean_ohlc_arrays(
    open_: list[float],
    high: list[float],
    low: list[float],
    close: list[float],
    max_jump: float = 5.0,
) -> tuple[list[float], list[float], list[float], list[float], int]:
    """Return cleaned copies and count of repaired bars."""
    o = list(open_)
    h = list(high)
    l = list(low)
    c = list(close)
    n = len(o)
    if n == 0:
        return o, h, l, c, 0

    ref = c[0] if c[0] == c[0] and c[0] > 0 else 1.0
    fixed = 0
    for i in range(n):
        r = ref if ref == ref and ref > 0 else 1.0
        lo_band = r / max_jump
        hi_band = r * max_jump
        bad = False
        for arr in (o, h, l, c):
            v = arr[i]
            if not (v > 0) or v < lo_band or v > hi_band:
                arr[i] = r
                bad = True
        if bad:
            fixed += 1
        hi = max(o[i], h[i], l[i], c[i])
        lo = min(o[i], h[i], l[i], c[i])
        if h[i] != hi or l[i] != lo:
            fixed += 1
        h[i] = hi
        l[i] = lo
        ref = c[i]
    return o, h, l, c, fixed
