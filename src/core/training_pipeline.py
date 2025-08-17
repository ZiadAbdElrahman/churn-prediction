import logging
import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Tuple
from scipy.stats import entropy

from .data_access import load_events
from ..dependencies import get_settings
from .model_io import save_artifacts
from .train import train_from_features
from .features import build_features_for_user_date  # the single-row builder you added

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

import logging
import numpy as np
import pandas as pd
from typing import Tuple

from .data_access import load_events
from ..dependencies import get_settings

logger = logging.getLogger(__name__)

# keep in sync with your feature builder & model artifacts
FEATURES = [
    "days_since_join",
    "gender_code",
    "day_of_month",
    "day_of_week",
    "dow_sin",
    "dow_cos",
    "dom_sin",
    "dom_cos",
    "artist_entropy_day",
    "cum_events",
    "cum_listen_sec",
    "cum_NextSong",
    "cum_Thumbs Up",
    "cum_Thumbs Down",
    "cum_Add to Playlist",
    "cum_Add Friend",
    "win7_events",
    "win7_listen_sec",
    "win7_NextSong",
    "win7_Thumbs Up",
    "win7_Thumbs Down",
    "win7_Add to Playlist",
    "win7_Add Friend",
    "ratio_thumbs_up",
    "ratio_playlist",
    "ratio_friend",
    "ratio7_thumbs_up",
    "ratio7_playlist",
    "ratio7_friend",
    "chg_thumbs_up",
    "chg_playlist",
    "chg_friend",
    "win7_error_rate",
    "win7_help_rate",
    "downgrade_wo_submit",
    "prop_nextsong",
    "win14_events",
    "win14_listen_sec",
    "win14_NextSong",
    "win14_Thumbs Up",
    "win14_Thumbs Down",
    "win14_Add to Playlist",
    "win14_Add Friend",
    "ratio14_thumbs_up",
    "ratio14_playlist",
    "ratio14_friend",
    "delta7_14_thumbs",
    "delta7_14_playlist",
]

PAGES = [
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


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    bad = ~np.isfinite(b) | (b == 0)  # NaN, Inf, or zero denom
    out = np.zeros_like(a, dtype=float)
    # compute only where denom is good
    np.divide(a, b, out=out, where=~bad)
    return out


def _cyclical(n, period):
    return np.sin(2 * np.pi * n / period), np.cos(2 * np.pi * n / period)


def build_training_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fast, vectorized dataset build (no nested loops).
    Returns X (features) and y (labels).
    """
    cfg = get_settings()
    H = int(cfg["model"]["horizon_days"])
    S = int(cfg["model"]["recent_window"])  # should be 7
    L = int(cfg["model"]["long_window"])  # should be 14

    df = load_events().copy()
    logger.info("Loaded events: %d rows, %d users", len(df), df["userId"].nunique())

    # ----- Churn detection (first Submit Downgrade with paid->free) -----
    df = df.sort_values(["userId", "datetime"]).copy()
    prev_level = df.groupby("userId")["level"].shift()
    next_level = df.groupby("userId")["level"].shift(-1)
    is_churn = (
        (df["page"].eq("Submit Downgrade"))
        & (prev_level.eq("paid"))
        & (next_level.eq("free"))
    )
    user_churn_ts = df.loc[is_churn].groupby("userId")["datetime"].min()
    user_churn_date = user_churn_ts.dt.floor("D")

    # ----- Daily aggregates per user -----
    df["date"] = df["datetime"].dt.floor("D")

    daily_events = df.groupby(["userId", "date"]).size().rename("events").to_frame()
    daily_listen = (
        df.groupby(["userId", "date"])["length"]
        .sum(min_count=1)
        .rename("listen_sec")
        .to_frame()
    )

    daily_pages = (
        df[df["page"].isin(PAGES)]
        .groupby(["userId", "date", "page"])
        .size()
        .unstack(fill_value=0)
    )
    for p in PAGES:
        if p not in daily_pages.columns:
            daily_pages[p] = 0
    daily_pages = daily_pages[PAGES]

    daily = (
        daily_events.join(daily_listen, how="outer")
        .join(daily_pages, how="outer")
        .fillna(0)
    )

    # ----- Build continuous daily index per user -----
    user_first_date = df.groupby("userId")["date"].min()
    user_last_date = df.groupby("userId")["date"].max()

    frames = []
    for uid, g in daily.groupby(level=0, sort=False):
        start = user_first_date.loc[uid]
        # stop at churn date for leakage safety (don’t generate rows after churn day)
        end_candidate = user_last_date.loc[uid]
        cdate = user_churn_date.get(uid, pd.NaT)
        end = min(end_candidate, cdate) if pd.notna(cdate) else end_candidate
        idx = pd.MultiIndex.from_product(
            [[uid], pd.date_range(start, end, freq="D")], names=["userId", "date"]
        )
        frames.append(g.reindex(idx).fillna(0))
    daily_full = pd.concat(frames).sort_index()

    # ----- Attach per-user static / calendar -----
    reg_map = df.groupby("userId")["registration_datetime"].min()
    gender_map = (
        df.dropna(subset=["userId"])
        .drop_duplicates("userId")
        .set_index("userId")["gender"]
        .map({"F": 0, "M": 1})
        .fillna(-1)
    )

    # reset index for vector ops
    feat = daily_full.reset_index()
    feat["churn_ts"] = feat["userId"].map(user_churn_ts)
    feat["churn_date"] = feat["churn_ts"].dt.floor("D")
    feat["registration_datetime"] = feat["userId"].map(reg_map)
    feat["gender_code"] = feat["userId"].map(gender_map)

    # calendar
    feat["day_of_month"] = feat["date"].dt.day
    feat["day_of_week"] = feat["date"].dt.dayofweek + 1
    feat["days_since_join"] = (
        (feat["date"] - feat["registration_datetime"].dt.floor("D"))
        .dt.days.fillna(0)
        .astype(int)
    )

    feat["dow_sin"], feat["dow_cos"] = _cyclical(feat["day_of_week"], 7)
    feat["dom_sin"], feat["dom_cos"] = _cyclical(feat["day_of_month"], 31)

    # ----- Artist entropy per user-day (vectorized) -----
    artist_counts = (
        df.dropna(subset=["artist"])
        .groupby(["userId", "date", "artist"])
        .size()
        .rename("cnt")
        .reset_index()
    )
    if not artist_counts.empty:
        artist_counts["sum_day"] = artist_counts.groupby(["userId", "date"])[
            "cnt"
        ].transform("sum")
        artist_counts["p"] = artist_counts["cnt"] / artist_counts["sum_day"]
        ent = (
            artist_counts.groupby(["userId", "date"])["p"]
            .apply(lambda p: entropy(p, base=2))
            .rename("artist_entropy_day")
        )
        feat = feat.merge(ent.reset_index(), on=["userId", "date"], how="left")
        feat["artist_entropy_day"] = feat["artist_entropy_day"].fillna(0.0)
    else:
        feat["artist_entropy_day"] = 0.0

    # ----- Cumulative & rolling (vectorized by user) -----
    cols_core = ["events", "listen_sec"] + PAGES
    # cum
    feat = feat.sort_values(["userId", "date"])
    feat[[f"cum_{c}" for c in cols_core]] = feat.groupby("userId")[cols_core].cumsum()

    # rolling S & L
    rollS = (
        feat.groupby("userId")[cols_core]
        .rolling(window=S, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rollS.columns = [f"win{S}_{c}" for c in cols_core]
    feat = pd.concat([feat, rollS], axis=1)

    rollL = (
        feat.groupby("userId")[cols_core]
        .rolling(window=L, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    rollL.columns = [f"win{L}_{c}" for c in cols_core]
    feat = pd.concat([feat, rollL], axis=1)

    # ----- Ratios / deltas (vectorized) -----
    # lifetime
    feat["ratio_thumbs_up"] = _safe_div(feat["cum_Thumbs Up"], feat["cum_NextSong"])
    feat["ratio_playlist"] = _safe_div(
        feat["cum_Add to Playlist"], feat["cum_NextSong"]
    )
    feat["ratio_friend"] = _safe_div(feat["cum_Add Friend"], feat["cum_NextSong"])

    # windows
    feat[f"ratio{S}_thumbs_up"] = _safe_div(
        feat[f"win{S}_Thumbs Up"], feat[f"win{S}_NextSong"]
    )
    feat[f"ratio{S}_playlist"] = _safe_div(
        feat[f"win{S}_Add to Playlist"], feat[f"win{S}_NextSong"]
    )
    feat[f"ratio{S}_friend"] = _safe_div(
        feat[f"win{S}_Add Friend"], feat[f"win{S}_NextSong"]
    )

    feat[f"ratio{L}_thumbs_up"] = _safe_div(
        feat[f"win{L}_Thumbs Up"], feat[f"win{L}_NextSong"]
    )
    feat[f"ratio{L}_playlist"] = _safe_div(
        feat[f"win{L}_Add to Playlist"], feat[f"win{L}_NextSong"]
    )
    feat[f"ratio{L}_friend"] = _safe_div(
        feat[f"win{L}_Add Friend"], feat[f"win{L}_NextSong"]
    )

    # changes
    feat["chg_thumbs_up"] = feat[f"ratio{S}_thumbs_up"] - feat["ratio_thumbs_up"]
    feat["chg_playlist"] = feat[f"ratio{S}_playlist"] - feat["ratio_playlist"]
    feat["chg_friend"] = feat[f"ratio{S}_friend"] - feat["ratio_friend"]

    feat[f"delta{S}_{L}_thumbs"] = (
        feat[f"ratio{S}_thumbs_up"] - feat[f"ratio{L}_thumbs_up"]
    )
    feat[f"delta{S}_{L}_playlist"] = (
        feat[f"ratio{S}_playlist"] - feat[f"ratio{L}_playlist"]
    )

    # error/help rates (recent S)
    feat[f"win{S}_error_rate"] = _safe_div(
        feat[f"win{S}_Error"], feat[f"win{S}_events"]
    )
    feat[f"win{S}_help_rate"] = _safe_div(feat[f"win{S}_Help"], feat[f"win{S}_events"])

    # lifetime flags/proportions
    feat["downgrade_wo_submit"] = (
        (feat["cum_Downgrade"] > 0) & (feat["cum_Submit Downgrade"] == 0)
    ).astype(int)
    feat["prop_nextsong"] = _safe_div(
        feat["cum_NextSong"].to_numpy(),
        feat["cum_events"].replace(0, np.nan).to_numpy(),
    )

    # ----- Label: churn within next H days, and drop rows after churn day -----
    feat["label_churn_horizon"] = 0
    has_c = feat["churn_date"].notna()
    feat.loc[
        has_c
        & (feat["churn_date"] > feat["date"])
        & (feat["churn_date"] <= feat["date"] + pd.Timedelta(days=H)),
        "label_churn_horizon",
    ] = 1

    feat = feat[(~has_c) | (feat["date"] <= feat["churn_date"])].copy()

    # ----- Final matrix -----
    X = feat[FEATURES].replace([np.inf, -np.inf], 0).fillna(0)
    y = feat["label_churn_horizon"].astype(int)

    logger.info(
        "Built dataset fast: %d rows, %d features, positives: %d (%.2f%%)",
        X.shape[0],
        X.shape[1],
        y.sum(),
        100 * y.mean(),
    )
    return X, y


def train_from_events_json() -> dict:
    logger.info("Starting training from events.json (fast)")
    X, y = build_training_dataset()
    report = train_from_features(X, y)
    report["n_rows"] = int(X.shape[0])
    report["n_features"] = int(X.shape[1])
    logger.info("Training completed. Report: %s", report)
    return report
