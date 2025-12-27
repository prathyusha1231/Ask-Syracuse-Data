"""
Deterministic data loading and light cleaning for Ask Syracuse Data.
All loaders read static CSV snapshots from data/raw and normalize column names.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Sequence

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _parse_dates(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _load_csv(filename: str, date_cols: Sequence[str]) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    df = _clean_columns(df)
    df = _parse_dates(df, date_cols)
    return df


def load_code_violations() -> pd.DataFrame:
    """Load housing code violations; columns lowercased and date fields parsed."""
    return _load_csv(
        "Code_Violations_V2.csv",
        date_cols=["open_date", "violation_date", "status_date", "comply_by_date"],
    )


def load_rental_registry() -> pd.DataFrame:
    """Load rental registry records; columns lowercased and date fields parsed."""
    return _load_csv(
        "Syracuse_Rental_Registry.csv",
        date_cols=["completion_date", "valid_until"],
    )


def load_vacant_properties() -> pd.DataFrame:
    """Load vacant properties; columns lowercased and date fields parsed."""
    return _load_csv(
        "Vacant_Properties.csv",
        date_cols=["completion_date", "valid_until"],
    )


def load_crime_2022() -> pd.DataFrame:
    """Load Part 1 crime incidents for 2022; columns lowercased and date fields parsed."""
    return _load_csv(
        "Crime_Data_2022_(Part_1_Offenses).csv",
        date_cols=["dateend"],
    )


__all__ = [
    "load_code_violations",
    "load_rental_registry",
    "load_vacant_properties",
    "load_crime_2022",
]
