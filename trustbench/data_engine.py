"""
Data Engineering Module for Trust Bench.
Loads raw UCI Online Retail II or synthesizes a high-fidelity 250-row benchmark table.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import hashlib
import json
import urllib.request
import zipfile

from trustbench.config import (
    TABLE_META_PATH,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TABLE_PATH,
    RAW_RETAIL_XLSX,
    RAW_RETAIL_CSV,
    SEED,
    TARGET_ROWS,
    DATE_SHIFT_DAYS,
)

RENAME_MAP = {
    "Invoice": "order_id",
    "InvoiceNo": "order_id",
    "StockCode": "product_code",
    "Description": "product_name",
    "Quantity": "qty",
    "InvoiceDate": "order_datetime",
    "Price": "unit_price",
    "UnitPrice": "unit_price",
    "Customer ID": "customer_id",
    "CustomerID": "customer_id",
    "Country": "country",
}

COLUMNS = [
    "order_id",
    "product_code",
    "product_name",
    "qty",
    "order_datetime",
    "unit_price",
    "customer_id",
    "country",
]


def generate_synthetic_uci_retail(seed: int = SEED) -> pd.DataFrame:
    """
    Generates a realistic e-commerce transaction dataset matching the UCI Online Retail II
    schema and statistical distribution (including cancelled orders starting with 'C',
    negative quantities, missing customer IDs, UK/Europe countries).
    """
    rng = np.random.default_rng(seed)

    products = [
        ("85048", "15CM CHRISTMAS GLASS BALL 20 LIGHTS", 6.95),
        ("79323P", "PINK CHERRY BLOSSOM LETTER RACK", 2.10),
        ("79323W", "WHITE CHERRY BLOSSOM LETTER RACK", 2.10),
        ("22041", "RECORD FRAME 7 INCH SINGLE SIZE", 2.55),
        ("21232", "STRAWBERRY CERAMIC TRINKET BOX", 1.25),
        ("22064", "DOUGHNUT LIP GLOSS", 1.65),
        ("21913", "VINTAGE UNION JACK MEMO BOARD", 4.95),
        ("21914", "BLUE HARMONICA IN BOX", 1.25),
        ("22423", "REGENCY CAKESTAND 3 TIER", 12.75),
        ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
        ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 4.95),
        ("POST", "POSTAGE", 18.00),
        ("21754", "HOME BUILDING BLOCK WORD", 5.95),
        ("21755", "LOVE BUILDING BLOCK WORD", 5.95),
        ("84991", "60 TEATIME BIRD GARDEN CAKE CASES", 0.55),
        ("22197", "SMALL POPCORN HOLDER", 0.85),
        ("21034", "REX CASH+CARRY JUMBO SHOPPER", 0.95),
        ("21212", "PACK OF 72 RETRO SPOT CAKE CASES", 0.55),
        ("20725", "LUNCH BAG RED RETROSPOT", 1.65),
        ("20727", "LUNCH BAG BLACK SKULL", 1.65),
        ("20728", "LUNCH BAG CARS BLUE", 1.65),
        ("22383", "LUNCH BAG SUKI DESIGN", 1.65),
        ("22384", "LUNCH BAG PINK POLKADOT", 1.65),
        ("22666", "RECIPE BOX PANTRY DESIGN", 2.95),
        ("22326", "ROUND SNACK BOXES SET OF 4 WOODLAND", 2.95),
    ]

    countries = ["United Kingdom"] * 18 + ["Germany", "France", "EIRE", "Spain", "Netherlands"]
    customers = [12346 + i * 37 for i in range(35)]

    base_date = datetime(2010, 1, 15, 10, 0, 0)
    records = []

    order_counter = 489434
    for i in range(110):
        # 10% cancelled orders
        is_cancelled = (i % 9 == 0)
        order_prefix = "C" if is_cancelled else ""
        current_order_id = f"{order_prefix}{order_counter}"
        order_counter += 1

        cust_id = rng.choice(customers) if rng.random() > 0.18 else None
        country = rng.choice(countries)
        order_dt = base_date + timedelta(
            days=int(rng.integers(0, 360)),
            hours=int(rng.integers(8, 18)),
            minutes=int(rng.integers(0, 60)),
        )

        # 1 to 5 items per order
        n_items = int(rng.integers(1, 5))
        chosen_items = rng.choice(len(products), size=n_items, replace=False)

        for item_idx in chosen_items:
            p_code, p_desc, p_price = products[item_idx]
            if is_cancelled:
                qty = -int(rng.integers(1, 10))
            else:
                qty = int(rng.integers(1, 48))

            records.append({
                "Invoice": current_order_id,
                "StockCode": p_code,
                "Description": p_desc,
                "Quantity": qty,
                "InvoiceDate": order_dt,
                "Price": p_price,
                "Customer ID": cust_id,
                "Country": country,
            })

    return pd.DataFrame(records)


UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def download_uci_dataset(dest_dir: str = str(RAW_DATA_DIR)) -> bool:
    """Downloads the official UCI Online Retail II zip and unpacks it into data/raw.
    Needs internet. If it fails, download the file by hand (UCI or Kaggle) and put
    online_retail_II.xlsx or .csv into data/raw/."""
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "online_retail_ii.zip")
    try:
        print("[TrustBench] Downloading", UCI_ZIP_URL)
        urllib.request.urlretrieve(UCI_ZIP_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        found = [f for f in os.listdir(dest_dir) if f.lower().startswith("online_retail")]
        print("[TrustBench] Files in data/raw:", found)
        if not (os.path.exists(RAW_RETAIL_XLSX) or os.path.exists(RAW_RETAIL_CSV)):
            print("[TrustBench] WARNING: expected online_retail_II.xlsx or .csv. Rename the extracted file to match.")
            return False
        return True
    except Exception as e:
        print("[TrustBench] Download failed:", e)
        return False


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:16]


def table_meta() -> dict:
    """Returns info about the current benchmark table (source: real or synthetic)."""
    if not os.path.exists(TABLE_META_PATH):
        return {"source": "unknown", "sha": file_sha256(TABLE_PATH) if os.path.exists(TABLE_PATH) else None}
    with open(TABLE_META_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_benchmark_table(save_path: str = TABLE_PATH, force_recreate: bool = False,
                            allow_synthetic: bool = False) -> pd.DataFrame:
    """
    Executes Step 2 & 3 of Trust Bench:
    - Loads the raw UCI retail file from data/raw. If it is missing this RAISES,
      unless allow_synthetic=True (then a clearly labelled synthetic table is made).
    - Picks small orders (up to 30 lines) with at least 5 cancelled orders.
    - Shifts dates by DATE_SHIFT_DAYS (1000 days).
    - Injects controlled noise: 6 missing product names, 4 exact duplicate rows.
    - Ensures customer_id is Int64.
    - Saves clean benchmark table to data/processed/benchmark_table.csv.
    """
    if os.path.exists(save_path) and not force_recreate:
        print(f"[TrustBench] Benchmark table already exists at: {save_path} (source: {table_meta().get('source')})")
        return pd.read_csv(
            save_path,
            parse_dates=["order_datetime"],
            dtype={"order_id": str, "product_code": str, "customer_id": "Int64"},
        )

    # 1. Load or synthesize raw data
    raw = None
    source = "real"
    if os.path.exists(RAW_RETAIL_XLSX):
        print(f"[TrustBench] Loading UCI dataset from {RAW_RETAIL_XLSX}...")
        sheets = pd.read_excel(RAW_RETAIL_XLSX, sheet_name=None)
        raw = pd.concat(sheets.values(), ignore_index=True)
    elif os.path.exists(RAW_RETAIL_CSV):
        print(f"[TrustBench] Loading UCI dataset from {RAW_RETAIL_CSV}...")
        raw = pd.read_csv(RAW_RETAIL_CSV)
    elif allow_synthetic:
        print("[TrustBench] WARNING: raw UCI file not found. Creating a SYNTHETIC table (demo only).")
        raw = generate_synthetic_uci_retail(seed=SEED)
        source = "synthetic"
    else:
        raise FileNotFoundError(
            "Real dataset not found. Download Online Retail II (UCI or Kaggle) and put "
            "online_retail_II.xlsx or online_retail_II.csv inside data/raw/. "
            "For a pipeline demo only, call with allow_synthetic=True (python run_pipeline.py --prepare-data --allow-synthetic).")

    # 2. Standardize column names
    df = raw.rename(columns=RENAME_MAP)[COLUMNS].copy()
    df["order_id"] = df["order_id"].astype(str)
    df["product_code"] = df["product_code"].astype(str)
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])

    # 3. Sample orders deterministically (Seed 42)
    sizes = df.groupby("order_id").size()
    shuffled = sizes.sample(frac=1.0, random_state=SEED)
    shuffled = shuffled[shuffled <= 30]

    cancelled_ids = [i for i in shuffled.index if i.startswith("C")][:5]
    normal_ids = [i for i in shuffled.index if not i.startswith("C")]

    chosen = list(cancelled_ids)
    total = int(shuffled[cancelled_ids].sum())
    for oid in normal_ids:
        if total >= TARGET_ROWS:
            break
        chosen.append(oid)
        total += int(shuffled[oid])

    bench = df[df["order_id"].isin(chosen)].copy()

    # 4. Shift dates by 1000 days so models cannot rely on memorized pretraining
    bench["order_datetime"] = bench["order_datetime"] + pd.Timedelta(days=DATE_SHIFT_DAYS)

    # 5. Add controlled real-world noise: 6 missing product names, 4 duplicate rows
    rng = np.random.default_rng(SEED)
    blank_idx = rng.choice(bench.index, size=min(6, len(bench)), replace=False)
    bench.loc[blank_idx, "product_name"] = np.nan

    dups = bench.sample(min(4, len(bench)), random_state=SEED)
    bench = pd.concat([bench, dups], ignore_index=True)

    # 6. Final tidy-up
    bench["customer_id"] = bench["customer_id"].astype("Int64")
    bench = bench.sort_values("order_datetime").reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bench.to_csv(save_path, index=False, encoding="utf-8")
    meta = {"source": source, "rows": int(len(bench)), "seed": SEED,
            "date_shift_days": DATE_SHIFT_DAYS, "sha": file_sha256(save_path)}
    with open(TABLE_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[TrustBench] Created benchmark table ({source}): {len(bench)} rows saved to {save_path}")
    return bench


if __name__ == "__main__":
    df = prepare_benchmark_table(allow_synthetic=False)
    print("Benchmark Table Sample:")
    print(df.head(5))

