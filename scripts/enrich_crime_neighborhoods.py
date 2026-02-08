"""
One-time script: Reverse geocode crime data to fill in neighborhood names.
Uses Nominatim (free, 1 req/sec rate limit) via geopy.

Workflow:
1. Collect all unique (lat, lon) pairs from 2023-2025 crime CSVs
2. Reverse geocode each to get suburb/neighbourhood from OpenStreetMap
3. Map OSM names to our canonical Syracuse neighborhood names
4. Save enriched CSVs with neighborhood column filled in

Usage:
    python -m scripts.enrich_crime_neighborhoods
    python -m scripts.enrich_crime_neighborhoods --resume   # resume from cache
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path

import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_FILE = Path(__file__).resolve().parent.parent / "data" / "geocode_cache.json"

# Crime files that need neighborhood enrichment (2022 enriched already has it)
FILES_TO_ENRICH = [
    "Crime_Data_2023_(Part_1_Offenses).csv",
    "Crime_Data_2023_(Part_2_Offenses).csv",
    "Crime_Data_2024_(Part_1_Offenses).csv",
    "Crime_Data_2024_(Part_2_Offenses).csv",
    "Crime_Data_2025_(Part_1_Offenses).csv",
]

# Map OSM suburb/neighbourhood names to our canonical neighborhood names
OSM_TO_CANONICAL = {
    "brighton": "Brighton",
    "court-woodlawn": "Court-Woodlawn",
    "court woodlawn": "Court-Woodlawn",
    "downtown": "Downtown",
    "eastwood": "Eastwood",
    "elmwood": "Elmwood",
    "far westside": "Far Westside",
    "far west side": "Far Westside",
    "hawley-green": "Hawley Green",
    "hawley green": "Hawley Green",
    "lakefront": "Lakefront",
    "lincoln hill": "Lincoln Hill",
    "meadowbrook": "Meadowbrook",
    "near westside": "Near Westside",
    "near west side": "Near Westside",
    "north valley": "North Valley",
    "northside": "Northside",
    "north side": "Northside",
    "outer comstock": "Outer Comstock",
    "park ave": "Park Ave",
    "park avenue": "Park Ave",
    "salt springs": "Salt Springs",
    "sedgwick": "Sedgwick",
    "skunk city": "Skunk City",
    "southside": "Southside",
    "south side": "Southside",
    "strathmore": "Strathmore",
    "tipp hill": "Tipp Hill",
    "tipperary hill": "Tipp Hill",
    "university": "University",
    "university hill": "University",
    "university neighborhood": "University",
    "washington square": "Washington Square",
    "westcott": "Westcott",
    "wescott": "Westcott",
    # Common OSM names that map to our neighborhoods
    "lincoln park": "Lincoln Hill",
    "prospect hill": "Brighton",
    "southwest": "Strathmore",
    "the valley": "North Valley",
}


def load_cache() -> dict:
    """Load geocode cache from disk."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    """Save geocode cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


def normalize_neighborhood(raw: str | None) -> str | None:
    """Normalize an OSM neighborhood name to our canonical name."""
    if not raw:
        return None
    return OSM_TO_CANONICAL.get(raw.lower().strip(), raw.strip().title())


def reverse_geocode_batch(coords: list[tuple[float, float]], cache: dict) -> dict:
    """
    Reverse geocode a list of (lat, lon) pairs using Nominatim.
    Returns updated cache mapping "lat,lon" -> neighborhood name.
    """
    geolocator = Nominatim(user_agent="ask-syracuse-data-enrichment", timeout=10)

    total = len(coords)
    new_lookups = 0
    errors = 0

    for i, (lat, lon) in enumerate(coords):
        key = f"{lat},{lon}"
        if key in cache:
            continue

        try:
            location = geolocator.reverse(
                (lat, lon), exactly_one=True, language="en",
                addressdetails=True
            )
            if location and location.raw.get("address"):
                addr = location.raw["address"]
                # Try suburb first, then neighbourhood, then city_district
                neighborhood = (
                    addr.get("suburb")
                    or addr.get("neighbourhood")
                    or addr.get("city_district")
                )
                cache[key] = neighborhood
            else:
                cache[key] = None

            new_lookups += 1

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            errors += 1
            cache[key] = None
            if errors > 20:
                print(f"\n  Too many errors ({errors}), stopping. Last error: {e}")
                break

        except Exception as e:
            errors += 1
            cache[key] = None

        # Progress update
        if new_lookups % 50 == 0 and new_lookups > 0:
            print(f"  Geocoded {new_lookups} new lookups ({i+1}/{total} processed, {errors} errors)")
            save_cache(cache)  # periodic save

        # Rate limit: 1 request per second
        time.sleep(1.1)

    return cache


def enrich_file(filename: str, cache: dict) -> int:
    """Enrich a single crime CSV with neighborhood data. Returns count of records enriched."""
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  Skipping {filename} (not found)")
        return 0

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Identify lat/lon columns
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "long" if "long" in df.columns else "longitude"

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"  Skipping {filename} (no coordinates)")
        return 0

    # Look up neighborhoods from cache
    enriched = 0
    neighborhoods = []
    for _, row in df.iterrows():
        lat, lon = row.get(lat_col), row.get(lon_col)
        if pd.notna(lat) and pd.notna(lon):
            key = f"{lat},{lon}"
            raw = cache.get(key)
            canonical = normalize_neighborhood(raw) if raw else None
            neighborhoods.append(canonical)
            if canonical:
                enriched += 1
        else:
            neighborhoods.append(None)

    df["neighborhood"] = neighborhoods

    # Save enriched version
    enriched_name = filename.replace(".csv", "_enriched.csv")
    out_path = DATA_DIR / enriched_name
    df.to_csv(out_path, index=False)
    print(f"  Saved {enriched_name} ({enriched}/{len(df)} records with neighborhood)")

    return enriched


def main():
    print("=" * 60)
    print("Crime Data Neighborhood Enrichment (Nominatim)")
    print("=" * 60)

    # Step 1: Collect all unique coordinates
    print("\nStep 1: Collecting unique coordinates...")
    all_coords = set()
    for fname in FILES_TO_ENRICH:
        path = DATA_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()
        lat_col = "lat" if "lat" in df.columns else "latitude"
        lon_col = "long" if "long" in df.columns else "longitude"
        if lat_col in df.columns and lon_col in df.columns:
            valid = df[[lat_col, lon_col]].dropna()
            for _, row in valid.iterrows():
                all_coords.add((row[lat_col], row[lon_col]))

    print(f"  Found {len(all_coords)} unique coordinate pairs")

    # Step 2: Load cache and geocode
    cache = load_cache()
    cached = sum(1 for lat, lon in all_coords if f"{lat},{lon}" in cache)
    remaining = len(all_coords) - cached
    print(f"\nStep 2: Reverse geocoding ({cached} cached, {remaining} remaining)")
    if remaining > 0:
        est_minutes = remaining * 1.1 / 60
        print(f"  Estimated time: ~{est_minutes:.0f} minutes")

    coords_list = list(all_coords)
    cache = reverse_geocode_batch(coords_list, cache)
    save_cache(cache)
    print(f"  Cache now has {len(cache)} entries")

    # Step 3: Enrich each file
    print("\nStep 3: Enriching CSV files...")
    total_enriched = 0
    for fname in FILES_TO_ENRICH:
        print(f"\n  Processing {fname}...")
        total_enriched += enrich_file(fname, cache)

    print(f"\n{'=' * 60}")
    print(f"Done! Enriched {total_enriched} total records with neighborhood data.")
    print(f"Enriched CSVs saved to {DATA_DIR}")
    print(f"Cache saved to {CACHE_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
