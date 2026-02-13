"""
One-time script: Fix Unknown neighborhoods in crime_merged.csv.

Uses the crime data's own known-neighborhood records as a reference.
For each Unknown record with coordinates, finds the nearest known-neighborhood
coordinate via cKDTree and assigns that neighborhood if within threshold.

This avoids the cache precision mismatch problem and doesn't need Nominatim.

Usage:
    python -m scripts.fix_unknown_neighborhoods
    python -m scripts.fix_unknown_neighborhoods --dry-run   # preview without saving
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MERGED_CSV = DATA_DIR / "crime_merged.csv"

# Max distance in degrees for matching (~555m, well within a neighborhood)
MAX_DISTANCE_DEG = 0.005


def main():
    parser = argparse.ArgumentParser(description="Fix Unknown neighborhoods in crime_merged.csv")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    print("=" * 60)
    print("Fix Unknown Neighborhoods in crime_merged.csv")
    print("=" * 60)

    df = pd.read_csv(MERGED_CSV, low_memory=False)
    unknown_mask = df["neighborhood"] == "Unknown"
    has_coords = df["latitude"].notna() & df["longitude"].notna()
    fixable_mask = unknown_mask & has_coords
    known_mask = ~unknown_mask & has_coords

    print(f"\nTotal records: {len(df)}")
    print(f"Unknown neighborhood: {unknown_mask.sum()}")
    print(f"  With coordinates (fixable): {fixable_mask.sum()}")
    print(f"  Without coordinates: {(unknown_mask & ~has_coords).sum()}")
    print(f"Known records with coords (reference): {known_mask.sum()}")

    # Build reference from known records: for each unique coord, take the most common neighborhood
    known = df.loc[known_mask, ["latitude", "longitude", "neighborhood"]]
    known_agg = (
        known.groupby(["latitude", "longitude"])["neighborhood"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    print(f"Unique known coordinates: {len(known_agg)}")

    # Get unique unknown coords
    fixable = df.loc[fixable_mask, ["latitude", "longitude"]].drop_duplicates()
    unk_arr = fixable.values
    print(f"Unique unknown coordinates: {len(unk_arr)}")

    # Build cKDTree from known coords
    known_arr = known_agg[["latitude", "longitude"]].values
    tree = cKDTree(known_arr)
    distances, indices = tree.query(unk_arr)

    # Build coord -> neighborhood mapping
    coord_to_nb = {}
    matched = 0
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        lat, lon = unk_arr[i]
        if dist < MAX_DISTANCE_DEG:
            coord_to_nb[(lat, lon)] = known_agg.iloc[idx]["neighborhood"]
            matched += 1
        else:
            coord_to_nb[(lat, lon)] = None

    print(f"\nMatched {matched}/{len(unk_arr)} unique coords within {MAX_DISTANCE_DEG} deg (~{MAX_DISTANCE_DEG * 111000:.0f}m)")

    # Show unmatched
    unmatched = [(lat, lon, dist) for (lat, lon), nb in coord_to_nb.items()
                 if nb is None
                 for dist in [distances[np.where((unk_arr[:, 0] == lat) & (unk_arr[:, 1] == lon))[0][0]]]]
    if unmatched:
        print(f"\nUnmatched coords ({len(unmatched)}):")
        for lat, lon, dist in unmatched[:10]:
            print(f"  ({lat:.6f}, {lon:.6f}) nearest known: {dist:.4f} deg ({dist * 111000:.0f}m)")

    # Apply fixes
    fixed_count = 0
    for idx in df.index[fixable_mask]:
        lat = df.at[idx, "latitude"]
        lon = df.at[idx, "longitude"]
        new_nb = coord_to_nb.get((lat, lon))
        if new_nb:
            df.at[idx, "neighborhood"] = new_nb
            fixed_count += 1

    remaining_unknown = (df["neighborhood"] == "Unknown").sum()

    print(f"\n{'=' * 60}")
    print(f"Results:")
    print(f"  Fixed: {fixed_count} records")
    print(f"  Before: {unknown_mask.sum()} Unknown")
    print(f"  After:  {remaining_unknown} Unknown")
    print(f"  Reduction: {unknown_mask.sum() - remaining_unknown} fewer ({100 * (unknown_mask.sum() - remaining_unknown) / unknown_mask.sum():.1f}%)")

    # Show top neighborhoods assigned
    if fixed_count > 0:
        fixed_nbs = []
        for idx_val in df.index[fixable_mask]:
            nb = df.at[idx_val, "neighborhood"]
            if nb != "Unknown":
                fixed_nbs.append(nb)
        from collections import Counter
        top = Counter(fixed_nbs).most_common(15)
        print(f"\n  Top assigned neighborhoods:")
        for nb, count in top:
            print(f"    {nb}: {count}")

    if not args.dry_run and fixed_count > 0:
        df.to_csv(MERGED_CSV, index=False)
        print(f"\nSaved updated {MERGED_CSV}")
    elif args.dry_run:
        print("\n(dry-run mode - no files modified)")
    else:
        print("\nNo fixes to apply.")

    print("=" * 60)


if __name__ == "__main__":
    main()
