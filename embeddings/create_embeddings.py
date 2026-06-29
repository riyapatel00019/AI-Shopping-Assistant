"""
============================================================
AI SHOPPING ASSISTANT
Embedding Generation Pipeline
============================================================

This script generates semantic embeddings for products
using Sentence Transformers.

Input:
    data/shopping_products.csv

Output:
    embeddings/embeddings.npy
    embeddings/product_mapping.pkl
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import logging
import warnings
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "data/shopping_products.csv"

OUTPUT_EMBEDDINGS = "embeddings/embeddings.npy"

OUTPUT_MAPPING = "embeddings/product_mapping.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 64

# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(filepath: str) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Loading Products Dataset")
    logger.info("=" * 60)

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Dataset not found : {filepath}"
        )

    df = pd.read_csv(filepath)

    logger.info(
        f"Dataset Loaded Successfully : {len(df)} products"
    )

    return df

# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df):

    logger.info("=" * 60)
    logger.info("Validating Dataset")
    logger.info("=" * 60)

    required_columns = [

        "product_id",
        "product_name",
        "search_text"

    ]

    missing = []

    for col in required_columns:

        if col not in df.columns:

            missing.append(col)

    if len(missing) > 0:

        raise Exception(
            f"Missing Columns : {missing}"
        )

    logger.info("Dataset Validation Successful")

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    logger.info("=" * 60)
    logger.info("Loading Sentence Transformer")
    logger.info("=" * 60)

    model = SentenceTransformer(
        MODEL_NAME
    )

    logger.info(
        f"Model Loaded : {MODEL_NAME}"
    )

    return model

# ============================================================
# PREPARE SEARCH TEXT
# ============================================================

def prepare_search_text(df):

    logger.info("=" * 60)
    logger.info("Preparing Search Text")
    logger.info("=" * 60)

    texts = []

    for _, row in df.iterrows():

        text = f"""
Product Name: {row.get("product_name", "")}

Brand: {row.get("brand", "")}

Main Category: {row.get("main_category", "")}

Sub Category: {row.get("sub_category", "")}

Description: {row.get("description", "")}

Search Keywords: {row.get("search_text", "")}

RAM: {row.get("ram", "")}

Storage: {row.get("storage", "")}

Battery: {row.get("battery", "")}

Processor: {row.get("processor", "")}

Display: {row.get("display", "")}
"""

        texts.append(text)

    logger.info(
        f"Prepared {len(texts)} search texts"
    )

    return texts

# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(model, texts):

    logger.info("=" * 60)
    logger.info("Generating Embeddings")
    logger.info("=" * 60)

    all_embeddings = []

    total_batches = (
        len(texts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    logger.info(
        f"Total Batches : {total_batches}"
    )

    for i in tqdm(

        range(0, len(texts), BATCH_SIZE),

        desc="Embedding Products"

    ):

        batch = texts[i:i + BATCH_SIZE]

        embeddings = model.encode(

            batch,

            convert_to_numpy=True,

            show_progress_bar=False,

            normalize_embeddings=False

        )

        all_embeddings.append(
            embeddings
        )

    embeddings = np.vstack(
        all_embeddings
    )

    logger.info(
        f"Embedding Shape : {embeddings.shape}"
    )

    return embeddings


# ============================================================
# NORMALIZE EMBEDDINGS
# ============================================================

def normalize_embeddings(embeddings):

    logger.info("=" * 60)
    logger.info("Normalizing Embeddings")
    logger.info("=" * 60)

    norms = np.linalg.norm(

        embeddings,

        axis=1,

        keepdims=True

    )

    norms[norms == 0] = 1

    embeddings = embeddings / norms

    logger.info(
        "Embeddings Normalized Successfully"
    )

    return embeddings


# ============================================================
# EMBEDDING SUMMARY
# ============================================================

def embedding_summary(df, embeddings):

    logger.info("=" * 60)
    logger.info("Embedding Summary")
    logger.info("=" * 60)

    print()

    print("Products :", len(df))

    print("Embedding Dimension :", embeddings.shape[1])

    print("Embedding Matrix :", embeddings.shape)

    print()


# ============================================================
# CREATE PRODUCT MAPPING
# ============================================================

# ============================================================
# CREATE PRODUCT MAPPING
# ============================================================

def create_product_mapping(df):

    logger.info("=" * 60)
    logger.info("Creating Product Mapping")
    logger.info("=" * 60)

    mapping = {}

    for idx, row in df.iterrows():

        mapping[idx] = {

            # -----------------------------
            # Product Information
            # -----------------------------
            "product_id": str(row.get("product_id", "")),
            "product_name": str(row.get("product_name", "Unknown")),
            "brand": str(row.get("brand", "Unknown")),

            # -----------------------------
            # Categories
            # -----------------------------
            "main_category": str(row.get("main_category", "Unknown")),
            "sub_category": str(row.get("sub_category", "Unknown")),

            # -----------------------------
            # Pricing
            # -----------------------------
            "price": float(row.get("price", 0.0))
                     if pd.notna(row.get("price")) else 0.0,

            "original_price": float(row.get("original_price", 0.0))
                     if pd.notna(row.get("original_price")) else 0.0,

            "discount_percent": float(row.get("discount_percent", 0.0))
                     if pd.notna(row.get("discount_percent")) else 0.0,

            # -----------------------------
            # Ratings
            # -----------------------------
            "rating": float(row.get("rating", 0.0))
                     if pd.notna(row.get("rating")) else 0.0,

            "rating_count": int(row.get("rating_count", 0))
                     if pd.notna(row.get("rating_count")) else 0,

            # -----------------------------
            # Description
            # -----------------------------
            "description": str(
                    row.get(
                        "description",
                        "No Description Available"
                    )
                ),

            # -----------------------------
            # Specifications
            # -----------------------------
            "ram": str(row.get("ram", "Unknown")),
            "storage": str(row.get("storage", "Unknown")),
            "battery": str(row.get("battery", "Unknown")),
            "display": str(row.get("display", "Unknown")),
            "processor": str(row.get("processor", "Unknown")),
            "warranty": str(row.get("warranty", "Unknown")),

            # -----------------------------
            # URLs
            # -----------------------------
            "image_url": str(row.get("image_url", "Unknown")),
            "product_url": str(row.get("product_url", "Unknown")),

            # -----------------------------
            # Search Text
            # -----------------------------

            "search_text": str(
                row.get("search_text", "")
            ),

            "embedding_text": str(
                row.get("search_text", "")
            )

        }

    logger.info(
        f"Product Mapping Created : {len(mapping)} products"
    )

    return mapping

# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(embeddings):

    logger.info("=" * 60)
    logger.info("Saving Embeddings")
    logger.info("=" * 60)

    output_dir = os.path.dirname(OUTPUT_EMBEDDINGS)

    os.makedirs(output_dir, exist_ok=True)

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings
    )

    logger.info(
        f"Embeddings Saved : {OUTPUT_EMBEDDINGS}"
    )


# ============================================================
# SAVE PRODUCT MAPPING
# ============================================================

def save_product_mapping(mapping):

    logger.info("=" * 60)
    logger.info("Saving Product Mapping")
    logger.info("=" * 60)

    output_dir = os.path.dirname(OUTPUT_MAPPING)

    os.makedirs(output_dir, exist_ok=True)

    with open(
        OUTPUT_MAPPING,
        "wb"
    ) as file:

        pickle.dump(
            mapping,
            file
        )

    logger.info(
        f"Mapping Saved : {OUTPUT_MAPPING}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("EMBEDDING PIPELINE STARTED")
    logger.info("=" * 60)

    # ------------------------------------
    # Load Dataset
    # ------------------------------------

    df = load_dataset(INPUT_FILE)

    validate_dataset(df)

    # ------------------------------------
    # Prepare Search Text
    # ------------------------------------

    texts = prepare_search_text(df)

    # ------------------------------------
    # Load Model
    # ------------------------------------

    model = load_embedding_model()

    # ------------------------------------
    # Generate Embeddings
    # ------------------------------------

    embeddings = generate_embeddings(
        model,
        texts
    )

    # ------------------------------------
    # Normalize Embeddings
    # ------------------------------------

    embeddings = normalize_embeddings(
        embeddings
    )

    # ------------------------------------
    # Summary
    # ------------------------------------

    embedding_summary(
        df,
        embeddings
    )

    # ------------------------------------
    # Product Mapping
    # ------------------------------------

    mapping = create_product_mapping(
        df
    )

    # ------------------------------------
    # Save Outputs
    # ------------------------------------

    save_embeddings(
        embeddings
    )

    save_product_mapping(
        mapping
    )

    logger.info("=" * 60)
    logger.info("EMBEDDING PIPELINE COMPLETED")
    logger.info("=" * 60)

    print("\nEmbeddings Generated Successfully!")

    print(f"\nEmbeddings : {OUTPUT_EMBEDDINGS}")

    print(f"Product Mapping : {OUTPUT_MAPPING}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()