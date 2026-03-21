"""
Data loader module for Calgary Traffic Incidents.

Fetches traffic incident data from Calgary Open Data portal using the
Socrata API (sodapy), caches locally as CSV, and provides preprocessing
and feature engineering utilities for clustering and classification.

Dataset: Traffic Incidents (35ra-9556)
Source: https://data.calgary.ca/Transportation-Transit/Traffic-Incidents/35ra-9556
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sodapy import Socrata

logger = logging.getLogger(__name__)

# Constants
SOCRATA_DOMAIN = "data.calgary.ca"
DATASET_ID = "35ra-9556"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "traffic_incidents.csv"

# Calgary city center coordinates for reference
CALGARY_LAT = 51.0447
CALGARY_LON = -114.0719

# Valid quadrants in Calgary
VALID_QUADRANTS = ["NW", "NE", "SW", "SE"]


def fetch_traffic_incidents(
    limit: int = 100000,
    force_refresh: bool = False,
    app_token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch traffic incident data from Calgary Open Data portal.

    Uses the Socrata API to download traffic incident records and caches
    them locally as a CSV file for subsequent use.

    Parameters
    ----------
    limit : int, optional
        Maximum number of records to fetch (default 100000).
    force_refresh : bool, optional
        If True, re-download even if cached file exists (default False).
    app_token : str, optional
        Socrata application token for higher rate limits.

    Returns
    -------
    pd.DataFrame
        Raw traffic incident data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_FILE.exists() and not force_refresh:
        logger.info("Loading cached data from %s", CACHE_FILE)
        df = pd.read_csv(CACHE_FILE, low_memory=False)
        logger.info("Loaded %d records from cache.", len(df))
        return df

    logger.info("Fetching data from Calgary Open Data portal...")
    try:
        client = Socrata(SOCRATA_DOMAIN, app_token, timeout=60)
        results = client.get(DATASET_ID, limit=limit)
        client.close()

        df = pd.DataFrame.from_records(results)
        df.to_csv(CACHE_FILE, index=False)
        logger.info(
            "Fetched and cached %d records to %s", len(df), CACHE_FILE
        )
        return df

    except Exception as exc:
        logger.error("Failed to fetch data from Socrata API: %s", exc)
        if CACHE_FILE.exists():
            logger.warning("Falling back to cached data.")
            return pd.read_csv(CACHE_FILE, low_memory=False)
        raise


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw traffic incident dataframe.

    Performs the following transformations:
      - Parse start_dt to datetime
      - Extract temporal features: hour, day_of_week, month, year
      - Clean and validate quadrant field
      - Convert latitude and longitude to numeric
      - Remove rows with missing critical fields

    Parameters
    ----------
    df : pd.DataFrame
        Raw traffic incident data.

    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed dataframe.
    """
    df = df.copy()

    # --- Parse datetime ------------------------------------------------
    if "start_dt" in df.columns:
        df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    else:
        logger.warning("Column 'start_dt' not found in dataframe.")
        # Attempt alternative column names
        for col in ["START_DT", "start_date", "incident_date"]:
            if col in df.columns:
                df["start_dt"] = pd.to_datetime(df[col], errors="coerce")
                break

    # --- Temporal feature extraction -----------------------------------
    if "start_dt" in df.columns and pd.api.types.is_datetime64_any_dtype(
        df["start_dt"]
    ):
        df["hour"] = df["start_dt"].dt.hour
        df["day_of_week"] = df["start_dt"].dt.dayofweek  # 0=Mon, 6=Sun
        df["day_name"] = df["start_dt"].dt.day_name()
        df["month"] = df["start_dt"].dt.month
        df["year"] = df["start_dt"].dt.year
        df["date"] = df["start_dt"].dt.date
    else:
        logger.warning("Could not extract temporal features.")

    # --- Quadrant cleaning ---------------------------------------------
    if "quadrant" in df.columns:
        df["quadrant"] = df["quadrant"].astype(str).str.strip().str.upper()
        df.loc[~df["quadrant"].isin(VALID_QUADRANTS), "quadrant"] = np.nan
    elif "QUADRANT" in df.columns:
        df["quadrant"] = df["QUADRANT"].astype(str).str.strip().str.upper()
        df.loc[~df["quadrant"].isin(VALID_QUADRANTS), "quadrant"] = np.nan

    # --- Latitude / Longitude ------------------------------------------
    for coord_col in ["latitude", "longitude"]:
        if coord_col in df.columns:
            df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
        else:
            alt = coord_col.upper()
            if alt in df.columns:
                df[coord_col] = pd.to_numeric(df[alt], errors="coerce")

    # --- Count column --------------------------------------------------
    if "count" in df.columns:
        df["incident_count"] = pd.to_numeric(df["count"], errors="coerce")
    elif "COUNT" in df.columns:
        df["incident_count"] = pd.to_numeric(df["COUNT"], errors="coerce")
    else:
        df["incident_count"] = 1

    # --- Drop rows missing critical spatial data -----------------------
    initial_len = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info(
            "Dropped %d rows with missing lat/lon (%.1f%%).",
            dropped,
            100.0 * dropped / max(initial_len, 1),
        )

    # Filter to reasonable Calgary bounding box
    df = df[
        (df["latitude"].between(50.8, 51.3))
        & (df["longitude"].between(-114.4, -113.8))
    ]

    df = df.reset_index(drop=True)
    return df


def create_clustering_features(df: pd.DataFrame) -> np.ndarray:
    """
    Create a feature matrix suitable for spatial clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with latitude and longitude columns.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2) with [latitude, longitude].
    """
    features = df[["latitude", "longitude"]].dropna().values
    return features


def create_classification_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Create features and target for incident severity / frequency classification.

    The target variable is a binary label indicating whether a given
    hour-location combination is a high-incident period (above median).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series or None]
        Feature dataframe (X) and target series (y).
    """
    required_cols = {"hour", "day_of_week", "month", "quadrant", "latitude", "longitude"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Missing columns for classification features: %s", missing)
        return pd.DataFrame(), None

    feature_df = df.copy()

    # Encode quadrant as numeric
    quadrant_map = {"NE": 0, "NW": 1, "SE": 2, "SW": 3}
    feature_df["quadrant_encoded"] = (
        feature_df["quadrant"].map(quadrant_map).fillna(-1).astype(int)
    )

    # Cyclical encoding for hour
    feature_df["hour_sin"] = np.sin(2 * np.pi * feature_df["hour"] / 24)
    feature_df["hour_cos"] = np.cos(2 * np.pi * feature_df["hour"] / 24)

    # Cyclical encoding for day of week
    feature_df["dow_sin"] = np.sin(2 * np.pi * feature_df["day_of_week"] / 7)
    feature_df["dow_cos"] = np.cos(2 * np.pi * feature_df["day_of_week"] / 7)

    # Cyclical encoding for month
    feature_df["month_sin"] = np.sin(2 * np.pi * feature_df["month"] / 12)
    feature_df["month_cos"] = np.cos(2 * np.pi * feature_df["month"] / 12)

    # Is weekend flag
    feature_df["is_weekend"] = (feature_df["day_of_week"] >= 5).astype(int)

    # Is rush hour flag (7-9 AM, 4-6 PM)
    feature_df["is_rush_hour"] = (
        feature_df["hour"].between(7, 9) | feature_df["hour"].between(16, 18)
    ).astype(int)

    # Create target: aggregate incident counts per hour-quadrant
    hourly_counts = (
        feature_df.groupby(["hour", "quadrant_encoded"])
        .size()
        .reset_index(name="hourly_count")
    )
    median_count = hourly_counts["hourly_count"].median()
    hourly_counts["high_incident"] = (
        hourly_counts["hourly_count"] > median_count
    ).astype(int)

    feature_df = feature_df.merge(
        hourly_counts[["hour", "quadrant_encoded", "high_incident"]],
        on=["hour", "quadrant_encoded"],
        how="left",
    )

    feature_columns = [
        "latitude",
        "longitude",
        "hour",
        "day_of_week",
        "month",
        "quadrant_encoded",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "is_weekend",
        "is_rush_hour",
    ]

    X = feature_df[feature_columns].dropna()
    y = feature_df.loc[X.index, "high_incident"].fillna(0).astype(int)

    return X, y


def load_and_prepare_data(
    limit: int = 100000,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Convenience function: fetch, cache, and preprocess traffic incident data.

    Parameters
    ----------
    limit : int, optional
        Maximum records to fetch (default 100000).
    force_refresh : bool, optional
        If True, re-download from the API.

    Returns
    -------
    pd.DataFrame
        Preprocessed dataframe ready for analysis.
    """
    raw_df = fetch_traffic_incidents(limit=limit, force_refresh=force_refresh)
    processed_df = preprocess_dataframe(raw_df)
    logger.info("Data ready: %d rows, %d columns.", *processed_df.shape)
    return processed_df
