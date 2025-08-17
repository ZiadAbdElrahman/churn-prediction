import logging
from functools import lru_cache
import pandas as pd
from pathlib import Path
from ..dependencies import get_settings

# --- Logging setup ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@lru_cache(maxsize=1)
def load_events() -> pd.DataFrame:
    cfg = get_settings()
    p = Path(cfg["paths"]["events_file"])

    if not p.exists():
        logger.error("Events file not found at %s", p)
        raise FileNotFoundError(p)

    logger.info("Loading events file: %s", p)

    # JSON Lines file
    df = pd.read_json(p, lines=True)
    logger.info("Loaded %d rows from events file", len(df))

    # canonical timestamps / helpers
    df["datetime"] = pd.to_datetime(df["ts"], unit="ms", errors="coerce")
    df["registration_datetime"] = pd.to_datetime(
        df["registration"], unit="ms", errors="coerce"
    )
    df["date"] = df["datetime"].dt.floor("D")

    # quick sanity check logging
    n_users = df["userId"].nunique() if "userId" in df.columns else 0
    null_ts = df["datetime"].isna().sum()
    logger.info(
        "Events cover %d unique users, with %d null timestamps", n_users, null_ts
    )

    return df
