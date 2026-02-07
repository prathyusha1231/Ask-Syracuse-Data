"""
Geocode crime addresses to extract ZIP codes and map to neighborhoods.
Run once to enrich the crime dataset, then save the results.

Usage: python geocode_crime.py
"""
import pandas as pd
import time
import json
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "raw"
CRIME_FILE = DATA_DIR / "Crime_Data_2022_(Part_1_Offenses).csv"
OUTPUT_FILE = DATA_DIR / "Crime_Data_2022_enriched.csv"
CACHE_FILE = REPO_ROOT / "geocode_cache.json"

# ZIP to Neighborhood mapping (from violations data)
ZIP_TO_NEIGHBORHOOD = {
    13202: "Downtown",
    13203: "Northside",
    13204: "Near Westside",
    13205: "Brighton",
    13206: "Eastwood",
    13207: "Elmwood",
    13208: "Northside",
    13210: "Westcott",
    13214: "Meadowbrook",
    13215: "Winkworth",
    13219: "Skunk City",
    13224: "Salt Springs",
}


def load_cache():
    """Load geocoding cache from file."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    """Save geocoding cache to file."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def geocode_address(address: str, geolocator, geocode_fn, cache: dict) -> dict:
    """
    Geocode an address and extract ZIP code.

    Returns dict with: zip, neighborhood, lat, lon
    """
    # Check cache first
    if address in cache:
        return cache[address]

    # Add Syracuse, NY to the address
    full_address = f"{address}, Syracuse, NY"

    try:
        location = geocode_fn(full_address)

        if location:
            # Try to extract ZIP from the address components
            raw = location.raw
            address_parts = raw.get('address', {})

            # Try different fields for ZIP
            zip_code = address_parts.get('postcode', '')

            # Extract just the 5-digit ZIP
            if zip_code:
                zip_code = zip_code.split('-')[0].strip()
                if zip_code.isdigit():
                    zip_code = int(zip_code)
                else:
                    zip_code = None
            else:
                zip_code = None

            result = {
                'zip': zip_code,
                'neighborhood': ZIP_TO_NEIGHBORHOOD.get(zip_code, 'Unknown'),
                'lat': location.latitude,
                'lon': location.longitude,
            }
        else:
            result = {
                'zip': None,
                'neighborhood': 'Unknown',
                'lat': None,
                'lon': None,
            }
    except Exception as e:
        print(f"  Error geocoding '{address}': {e}")
        result = {
            'zip': None,
            'neighborhood': 'Unknown',
            'lat': None,
            'lon': None,
        }

    # Cache the result
    cache[address] = result
    return result


def main():
    print("Loading crime data...")
    df = pd.read_csv(CRIME_FILE)
    df.columns = df.columns.str.strip().str.lower()

    print(f"Total records: {len(df)}")
    print(f"Unique addresses: {df['address'].nunique()}")

    # Get unique addresses
    unique_addresses = df['address'].unique().tolist()

    # Load cache
    cache = load_cache()
    cached_count = sum(1 for addr in unique_addresses if addr in cache)
    print(f"Already cached: {cached_count}/{len(unique_addresses)}")

    # Set up geocoder with rate limiting
    geolocator = Nominatim(user_agent="syracuse_data_app", timeout=10)
    geocode_fn = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, return_value_on_exception=None)

    # Geocode each unique address
    to_geocode = [addr for addr in unique_addresses if addr not in cache]
    print(f"Addresses to geocode: {len(to_geocode)}")

    if to_geocode:
        print(f"Estimated time: {len(to_geocode) * 1.1 / 60:.1f} minutes")
        print()

        for i, address in enumerate(to_geocode):
            result = geocode_address(address, geolocator, geocode_fn, cache)

            # Progress update every 50 addresses
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(to_geocode)} - Last: {address} -> ZIP {result['zip']}")
                save_cache(cache)  # Save progress periodically

        # Final save
        save_cache(cache)
        print(f"\nGeocoding complete! Cached {len(cache)} addresses.")

    # Apply geocoding results to dataframe
    print("\nApplying results to dataset...")
    df['zip'] = df['address'].map(lambda a: cache.get(a, {}).get('zip'))
    df['neighborhood'] = df['address'].map(lambda a: cache.get(a, {}).get('neighborhood', 'Unknown'))
    df['latitude'] = df['address'].map(lambda a: cache.get(a, {}).get('lat'))
    df['longitude'] = df['address'].map(lambda a: cache.get(a, {}).get('lon'))

    # Summary
    print(f"\nResults:")
    print(f"  Records with ZIP: {df['zip'].notna().sum()}/{len(df)}")
    print(f"  Records with neighborhood: {(df['neighborhood'] != 'Unknown').sum()}/{len(df)}")

    print(f"\nNeighborhood distribution:")
    print(df['neighborhood'].value_counts())

    # Save enriched dataset
    print(f"\nSaving enriched dataset to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("Done!")


if __name__ == "__main__":
    main()
