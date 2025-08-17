import numpy as np
import pandas as pd
from scipy.stats import entropy
from ..dependencies import get_settings

PAGES_CANON = [
    "NextSong",
    "Thumbs Up",
    "Thumbs Down",
    "Add to Playlist",
    "Add Friend",
    "Help",
    "Error",
    "Downgrade",
    "Submit Downgrade",
    "Home",
    "Roll Advert",
]


def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.zeros_like(a, dtype=float)
    mask = b != 0
    out[mask] = a[mask] / b[mask]
    return out


def _cyclical(n, period):
    return np.sin(2 * np.pi * n / period), np.cos(2 * np.pi * n / period)


def build_features_for_user_date(
    df_all: pd.DataFrame, user_id: str, cutoff_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Return a single engineered feature row for (user_id, cutoff_date),
    using only data <= cutoff_date (no leakage).
    """
    cfg = get_settings()
    S = int(cfg["model"]["recent_window"])
    L = int(cfg["model"]["long_window"])

    # slice user + time
    d = df_all[df_all["userId"] == user_id].copy()
    if d.empty:
        raise ValueError(f"user_id '{user_id}' not found")
    d = d[d["date"] <= cutoff_date].copy()
    if d.empty:
        raise ValueError(f"No activity for user {user_id} up to {cutoff_date.date()}")

    # per-day base
    daily = (
        d.groupby("date")
        .agg(events=("ts", "count"), listen_sec=("length", "sum"))
        .fillna(0)
    )

    # per-day page counts
    pages = (
        d[d["page"].isin(PAGES_CANON)]
        .groupby(["date", "page"])
        .size()
        .unstack(fill_value=0)
    )
    for p in PAGES_CANON:
        if p not in pages.columns:
            pages[p] = 0
    pages = pages[PAGES_CANON]  # column order
    daily = daily.join(pages, how="left").fillna(0)

    # continuous daily index (from first activity to cutoff)
    full_idx = pd.date_range(daily.index.min(), cutoff_date, freq="D")
    daily = daily.reindex(full_idx, fill_value=0)
    daily.index.name = "date"

    # cumulative
    cum = daily.cumsum()
    cum.columns = [f"cum_{c}" for c in cum.columns]

    # rolling sums
    winS = daily.rolling(S, min_periods=1).sum()
    winS.columns = [f"win{S}_{c}" for c in winS.columns]

    winL = daily.rolling(L, min_periods=1).sum()
    winL.columns = [f"win{L}_{c}" for c in winL.columns]

    # artist entropy per-day (only where artist present)
    uda = d.groupby(["date", "artist"]).size().rename("cnt").reset_index()
    if not uda.empty:
        uda["p"] = uda.groupby("date")["cnt"].transform(lambda x: x / x.sum())
        ent = (
            uda.groupby("date")["p"]
            .apply(lambda p: entropy(p, base=2))
            .reindex(full_idx)
            .fillna(0.0)
        )
    else:
        ent = pd.Series(0.0, index=full_idx)
    artist_entropy_day = ent.rename("artist_entropy_day")

    # calendar/static
    reg_dt = d["registration_datetime"].dropna().min()
    days_since_join = (
        (pd.Series(full_idx, index=full_idx) - reg_dt.floor("D")).dt.days
        if pd.notna(reg_dt)
        else pd.Series(0, index=full_idx)
    )
    dow = pd.Series(full_idx, index=full_idx).dt.dayofweek + 1
    dom = pd.Series(full_idx, index=full_idx).dt.day
    dow_sin, dow_cos = _cyclical(dow, 7)
    dom_sin, dom_cos = _cyclical(dom, 31)

    # gender code (F=0, M=1, other=-1)
    gender = (
        d.dropna(subset=["userId"])
        .drop_duplicates("userId")
        .set_index("userId")["gender"]
        .map({"F": 0, "M": 1})
        .fillna(-1)
        .iloc[0]
    )
    static = pd.DataFrame(
        {
            "days_since_join": days_since_join.values,
            "gender_code": float(gender),
            "day_of_month": dom.values,
            "day_of_week": dow.values,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "dom_sin": dom_sin,
            "dom_cos": dom_cos,
        },
        index=full_idx,
    )

    # assemble
    feat_daily = pd.concat(
        [daily, cum, winS, winL, artist_entropy_day, static], axis=1
    ).fillna(0)

    # ratios & changes (use last row only)
    last = feat_daily.iloc[[-1]].copy()

    # lifetime ratios
    last["ratio_thumbs_up"] = safe_div(
        last.get("cum_Thumbs Up", 0), last.get("cum_NextSong", 0)
    )
    last["ratio_playlist"] = safe_div(
        last.get("cum_Add to Playlist", 0), last.get("cum_NextSong", 0)
    )
    last["ratio_friend"] = safe_div(
        last.get("cum_Add Friend", 0), last.get("cum_NextSong", 0)
    )

    # recent ratios (S & L)
    last[f"ratio{S}_thumbs_up"] = safe_div(
        last.get(f"win{S}_Thumbs Up", 0), last.get(f"win{S}_NextSong", 0)
    )
    last[f"ratio{L}_thumbs_up"] = safe_div(
        last.get(f"win{L}_Thumbs Up", 0), last.get(f"win{L}_NextSong", 0)
    )
    last[f"ratio{S}_playlist"] = safe_div(
        last.get(f"win{S}_Add to Playlist", 0), last.get(f"win{S}_NextSong", 0)
    )
    last[f"ratio{L}_playlist"] = safe_div(
        last.get(f"win{L}_Add to Playlist", 0), last.get(f"win{L}_NextSong", 0)
    )
    last[f"ratio{S}_friend"] = safe_div(
        last.get(f"win{S}_Add Friend", 0), last.get(f"win{S}_NextSong", 0)
    )
    last[f"ratio{L}_friend"] = safe_div(
        last.get(f"win{L}_Add Friend", 0), last.get(f"win{L}_NextSong", 0)
    )

    # changes
    last["chg_thumbs_up"] = last[f"ratio{S}_thumbs_up"] - last["ratio_thumbs_up"]
    last["chg_playlist"] = last[f"ratio{S}_playlist"] - last["ratio_playlist"]
    last["chg_friend"] = last[f"ratio{S}_friend"] - last["ratio_friend"]
    last[f"delta{S}_{L}_thumbs"] = (
        last[f"ratio{S}_thumbs_up"] - last[f"ratio{L}_thumbs_up"]
    )
    last[f"delta{S}_{L}_playlist"] = (
        last[f"ratio{S}_playlist"] - last[f"ratio{L}_playlist"]
    )

    # error/help intensity (recent)
    last[f"win{S}_error_rate"] = safe_div(
        last.get(f"win{S}_Error", 0), last.get(f"win{S}_events", 0)
    )
    last[f"win{S}_help_rate"] = safe_div(
        last.get(f"win{S}_Help", 0), last.get(f"win{S}_events", 0)
    )

    # downgrade without submit (lifetime)
    last["downgrade_wo_submit"] = (
        (last.get("cum_Downgrade", 0) > 0) & (last.get("cum_Submit Downgrade", 0) == 0)
    ).astype(int)

    # engagement proxy
    last["prop_nextsong"] = safe_div(
        last.get("cum_NextSong", 0), last.get("cum_events", 1)
    )

    last.index = pd.Index([cutoff_date], name="date")
    return last
