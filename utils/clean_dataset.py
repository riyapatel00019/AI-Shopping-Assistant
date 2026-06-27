
"""
============================================================
AI SHOPPING ASSISTANT
Data Engineering Pipeline

============================================================
"""

import os
import re
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/amazon.csv"
OUTPUT_FILE = "data/products.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD DATASET
# ============================================================

def load_data(filepath: str) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Loading Dataset")
    logger.info("=" * 60)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found : {filepath}"
        )

    df = pd.read_csv(filepath)

    logger.info(f"Dataset Loaded Successfully")

    return df


# ============================================================
# DATASET INFORMATION
# ============================================================

def inspect_dataset(df):

    logger.info("=" * 60)
    logger.info("Dataset Inspection")
    logger.info("=" * 60)

    print("\nShape")
    print(df.shape)

    print("\nColumns")
    print(df.columns.tolist())

    print("\nData Types")
    print(df.dtypes)

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\nFirst Five Rows")
    print(df.head())


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_columns(df):

    required_columns = [

        "product_id",
        "product_name",
        "category",
        "discounted_price",
        "actual_price",
        "rating",
        "rating_count",
        "about_product",
        "img_link",
        "product_link"

    ]

    missing = []

    for col in required_columns:

        if col not in df.columns:
            missing.append(col)

    if len(missing) > 0:

        raise Exception(
            f"Missing Columns : {missing}"
        )

    logger.info("Required columns verified.")


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    logger.info(
        f"Duplicate Rows Removed : {before-after}"
    )

    return df


# ============================================================
# REMOVE DUPLICATE PRODUCTS
# ============================================================

def remove_duplicate_products(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=["product_name"]
    )

    after = len(df)

    logger.info(
        f"Duplicate Products Removed : {before-after}"
    )

    return df


# ============================================================
# REMOVE UNUSED COLUMNS
# ============================================================

def remove_unwanted_columns(df):

    remove_columns = [

        "user_id",
        "user_name",
        "review_id",
        "review_title",
        "review_content"

    ]

    available = []

    for col in remove_columns:

        if col in df.columns:
            available.append(col)

    df = df.drop(columns=available)

    logger.info(
        "Unused Columns Removed"
    )

    return df


# ============================================================
# RENAME COLUMNS
# ============================================================

def rename_columns(df):

    rename_dict = {

        "discounted_price": "price",
        "actual_price": "original_price",
        "about_product": "description",
        "img_link": "image_url",
        "product_link": "product_url"

    }

    df = df.rename(
        columns=rename_dict
    )

    logger.info(
        "Columns Renamed Successfully"
    )

    return df


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):

    logger.info("=" * 60)
    logger.info("Handling Missing Values")
    logger.info("=" * 60)

    text_columns = [

        "product_name",
        "description",
        "category"

    ]

    for col in text_columns:

        if col in df.columns:

            df[col] = df[col].fillna("Unknown")

    numeric_columns = [

        "price",
        "original_price",
        "rating",
        "rating_count"

    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = df[col].replace(
                "",
                np.nan
            )

    logger.info("Missing Values Processed")

    return df


# ============================================================
# REMOVE EMPTY PRODUCTS
# ============================================================

def remove_empty_products(df):

    before = len(df)

    df = df.dropna(
        subset=["product_name"]
    )

    after = len(df)

    logger.info(
        f"Empty Products Removed : {before-after}"
    )

    return df


# ============================================================
# BASIC TEXT CLEANING
# ============================================================

def basic_text_cleaning(df):

    logger.info("=" * 60)
    logger.info("Basic Text Cleaning")
    logger.info("=" * 60)

    def clean(text):

        if pd.isna(text):
            return ""

        text = str(text)

        text = text.replace("\n", " ")

        text = text.replace("\t", " ")

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        text = text.strip()

        return text

    columns = [

        "product_name",
        "description"

    ]

    for col in columns:

        if col in df.columns:

            df[col] = df[col].apply(clean)

    logger.info(
        "Text Cleaned Successfully"
    )

    return df


# ============================================================
# CLEAN PRICE
# ============================================================

def clean_prices(df):

    logger.info("=" * 60)
    logger.info("Cleaning Prices")
    logger.info("=" * 60)

    def clean(price):

        if pd.isna(price):
            return np.nan

        price = str(price)

        price = re.sub(r"[₹,]", "", price)
        price = re.sub(r"[^0-9.]", "", price)

        try:
            return float(price)
        except:
            return np.nan

    df["price"] = df["price"].apply(clean)
    df["original_price"] = df["original_price"].apply(clean)

    return df


# ============================================================
# CLEAN RATING
# ============================================================

def clean_ratings(df):

    logger.info("Cleaning Ratings")

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce"
    )

    return df


# ============================================================
# CLEAN RATING COUNT
# ============================================================

def clean_rating_count(df):

    logger.info("Cleaning Rating Count")

    def clean(x):

        if pd.isna(x):
            return 0

        x = str(x)

        x = x.replace(",", "")

        x = re.sub(r"[^0-9]", "", x)

        if x == "":
            return 0

        return int(x)

    df["rating_count"] = df["rating_count"].apply(clean)

    return df


# ============================================================
# NORMALIZE CATEGORY
# ============================================================

def normalize_categories(df):

    logger.info("=" * 60)
    logger.info("Normalizing Categories")
    logger.info("=" * 60)

    main = []
    sub = []

    for cat in df["category"]:

        if pd.isna(cat):

            main.append("Unknown")
            sub.append("Unknown")

            continue

        parts = str(cat).split("|")

        main.append(parts[0].strip())

        if len(parts) > 1:
            sub.append(parts[-1].strip())
        else:
            sub.append(parts[0].strip())

    df["main_category"] = main
    df["sub_category"] = sub

    return df


# ============================================================
# BRAND EXTRACTION
# ============================================================

BRANDS = [

    "Samsung",
    "Apple",
    "OnePlus",
    "Xiaomi",
    "Redmi",
    "Realme",
    "Motorola",
    "Vivo",
    "Oppo",
    "Nothing",
    "Google",

    "HP",
    "Dell",
    "Lenovo",
    "Asus",
    "Acer",
    "MSI",

    "Sony",
    "Boat",
    "JBL",
    "Noise",
    "Philips",
    "LG",
    "Canon",
    "Nikon",
    "Logitech"

]


def extract_brand(df):

    logger.info("=" * 60)
    logger.info("Extracting Brand")
    logger.info("=" * 60)

    brand_list = []

    for product in df["product_name"]:

        product = str(product)

        found = "Unknown"

        for brand in BRANDS:

            if re.search(
                rf"\b{re.escape(brand)}\b",
                product,
                re.IGNORECASE
            ):
                found = brand
                break

        brand_list.append(found)

    df["brand"] = brand_list

    return df


# ============================================================
# RAM EXTRACTION
# ============================================================

def extract_ram(text):

    text = str(text)

    match = re.search(
        r'(\d+)\s?GB\s?RAM',
        text,
        re.I
    )

    if match:
        return match.group(1) + "GB"

    return "Unknown"


# ============================================================
# STORAGE EXTRACTION
# ============================================================

def extract_storage(text):

    text = str(text)

    # Look for storage values only
    patterns = [
        r'(\d+)\s?GB\s?(Storage|ROM|SSD|HDD|Internal)',
        r'(\d+)\s?TB\s?(Storage|SSD|HDD)',
        r'(128|256|512)\s?GB',
        r'(1|2)\s?TB'
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:

            value = match.group(1)

            if "TB" in match.group(0).upper():
                return value + "TB"

            return value + "GB"

    return "Unknown"


# ============================================================
# BATTERY EXTRACTION
# ============================================================

def extract_battery(text):

    text = str(text)

    match = re.search(
        r'(\d+)\s?mAh',
        text,
        re.I
    )

    if match:
        return match.group(1) + "mAh"

    return "Unknown"


# ============================================================
# DISPLAY EXTRACTION
# ============================================================

def extract_display(text):

    text = str(text)

    patterns = [
        r'(\d+(\.\d+)?)\s?inch',
        r'(\d+(\.\d+)?)"',
        r'(\d+(\.\d+)?)\s?in'
    ]

    for pattern in patterns:

        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(1) + " inch"

    return "Unknown"


# ============================================================
# WARRANTY EXTRACTION
# ============================================================

def extract_warranty(text):

    text = str(text)

    match = re.search(
        r'(\d+)\s?year',
        text,
        re.I
    )

    if match:
        return match.group(1) + " Year"

    return "Unknown"

# ============================================================
# PROCESSOR EXTRACTION
# ============================================================

def extract_processor(text):

    text = str(text)

    processors = [
        "Snapdragon",
        "MediaTek",
        "Intel",
        "Ryzen",
        "Apple M1",
        "Apple M2",
        "Apple M3"
    ]

    for p in processors:

        if p.lower() in text.lower():
            return p

    return "Unknown"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(df):

    logger.info("=" * 60)
    logger.info("Extracting Product Features")
    logger.info("=" * 60)

    ram = []
    storage = []
    battery = []
    display = []
    warranty = []
    processor = []

    for desc in df["description"]:

        ram.append(extract_ram(desc))

        storage.append(extract_storage(desc))

        battery.append(extract_battery(desc))

        display.append(extract_display(desc))

        warranty.append(extract_warranty(desc))

        processor.append(extract_processor(desc))

    df["ram"] = ram
    df["storage"] = storage
    df["battery"] = battery
    df["display"] = display
    df["warranty"] = warranty
    df["processor"] = processor

    return df

def save_dataset(df):

    logger.info("=" * 60)
    logger.info("Saving Dataset")
    logger.info("=" * 60)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset Saved Successfully!")
    print(f"Saved to: {OUTPUT_FILE}")


def create_search_text(df):

    logger.info("=" * 60)
    logger.info("Creating Search Text")
    logger.info("=" * 60)

    search_text = []

    for _, row in df.iterrows():

        text = f"""
        Product : {row.get('product_name','')}
        Brand : {row.get('brand','')}
        Category : {row.get('main_category','')}
        Sub Category : {row.get('sub_category','')}
        Description : {row.get('description','')}
        RAM : {row.get('ram','')}
        Storage : {row.get('storage','')}
        Battery : {row.get('battery','')}
        Display : {row.get('display','')}
        Warranty : {row.get('warranty','')}
        Processor : {row.get('processor','')}
        """

        text = re.sub(r"\s+", " ", text)

        search_text.append(text.strip())

    df["search_text"] = search_text

    logger.info("Search Text Created Successfully")

    return df

def calculate_discount(df):

    logger.info("=" * 60)
    logger.info("Calculating Discount")
    logger.info("=" * 60)

    df["discount_percent"] = np.where(
        df["original_price"] > 0,
        ((df["original_price"] - df["price"]) / df["original_price"]) * 100,
        0
    )

    df["discount_percent"] = df["discount_percent"].round(2)

    return df



# ============================================================
# MAIN
# ============================================================

def main():
    print("===== MAIN FUNCTION STARTED =====")

    df = load_data(INPUT_FILE)

    inspect_dataset(df)

    validate_columns(df)

    df = remove_duplicates(df)

    df = remove_duplicate_products(df)

    df = remove_unwanted_columns(df)

    df = rename_columns(df)

    df = handle_missing_values(df)

    df = remove_empty_products(df)

    df = basic_text_cleaning(df)

    df = clean_prices(df)

    df = clean_ratings(df)

    df = clean_rating_count(df)

    df = normalize_categories(df)

    df = extract_brand(df)

    df = extract_features(df)

    df = create_search_text(df)

    df = calculate_discount(df)
    final_columns = [
    "product_id",
    "product_name",
    "brand",
    "main_category",
    "sub_category",
    "price",
    "original_price",
    "discount_percent",
    "rating",
    "rating_count",
    "description",
    "ram",
    "storage",
    "battery",
    "display",
    "warranty",
    "processor",
    "image_url",
    "product_url",
    "search_text"
]

    df = df[final_columns]

    save_dataset(df)

    print("\nPipeline Completed Successfully!")


if __name__ == "__main__":
    main()

