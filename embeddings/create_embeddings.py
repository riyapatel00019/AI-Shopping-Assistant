"""
============================================================
AI SHOPPING ASSISTANT
Production Embedding Generation Pipeline
============================================================

This script performs:

1. Load shopping dataset
2. Validate dataset
3. Clean dataset
4. Generate rich search text
5. Generate embeddings
6. Normalize embeddings
7. Create FAISS Index
8. Save Embeddings (.npy)
9. Save FAISS Index (.index)
10. Save Product Mapping (.pkl)
11. Save Metadata (.json)

Author : Riya Patel
============================================================
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import os
import json
import time
import pickle
import logging
import warnings

import faiss
import numpy as np
import pandas as pd

from tqdm import tqdm
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

# Dataset Path
DATASET_PATH = "data/shopping_products.csv"

# Output Directory
OUTPUT_DIR = "embeddings"

# Create Output Directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output Files
EMBEDDINGS_FILE = os.path.join(
    OUTPUT_DIR,
    "embeddings.npy"
)

FAISS_INDEX_FILE = os.path.join(
    OUTPUT_DIR,
    "faiss.index"
)

PRODUCT_MAPPING_FILE = os.path.join(
    OUTPUT_DIR,
    "product_mapping.pkl"
)

METADATA_FILE = os.path.join(
    OUTPUT_DIR,
    "embedding_metadata.json"
)

# Embedding Model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Batch Size
BATCH_SIZE = 64

# Embedding Dimension
EMBEDDING_DIMENSION = 384

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

# ============================================================
# START TIMER
# ============================================================

START_TIME = time.time()

logger.info("=" * 70)
logger.info("AI SHOPPING ASSISTANT")
logger.info("Production Embedding Pipeline Started")
logger.info("=" * 70)

# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load shopping products dataset.
    """

    logger.info("=" * 70)
    logger.info("Loading Shopping Dataset")
    logger.info("=" * 70)

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Dataset not found : {filepath}"
        )

    df = pd.read_csv(filepath)

    logger.info(f"Dataset Loaded Successfully : {len(df)} products")

    return df


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [

    "product_id",

    "product_name",

    "category",

    "sub_category",

    "brand",

    "price",

    "rating",

    "description",

    "features",

    "purpose",

    "image_url",

    "product_url"

]


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(df: pd.DataFrame):

    logger.info("=" * 70)
    logger.info("Validating Dataset")
    logger.info("=" * 70)

    missing_columns = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns

    ]

    if missing_columns:

        raise Exception(

            f"Missing Columns : {missing_columns}"

        )

    logger.info("Dataset Validation Successful")


# ============================================================
# CLEAN DATASET
# ============================================================

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("=" * 70)
    logger.info("Cleaning Dataset")
    logger.info("=" * 70)

    # ---------------------------------------
    # Remove Duplicate Products
    # ---------------------------------------

    before = len(df)

    df = df.drop_duplicates(

        subset=[

            "product_name",

            "brand"

        ]

    )

    after = len(df)

    logger.info(

        f"Duplicate Products Removed : {before-after}"

    )

    # ---------------------------------------
    # Fill Missing Values
    # ---------------------------------------

    df = df.fillna("")

    # ---------------------------------------
    # Numeric Columns
    # ---------------------------------------

    numeric_columns = [

        "price",

        "rating"

    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(

            df[column],

            errors="coerce"

        )

        df[column] = df[column].fillna(0)

    logger.info("Missing Values Filled")

    logger.info("Dataset Cleaned Successfully")

    return df


# ============================================================
# DATASET SUMMARY
# ============================================================

def dataset_summary(df: pd.DataFrame):

    logger.info("=" * 70)
    logger.info("Dataset Summary")
    logger.info("=" * 70)

    print()

    print("-" * 50)

    print(f"Total Products      : {len(df)}")

    print(f"Categories          : {df['category'].nunique()}")

    print(f"Brands              : {df['brand'].nunique()}")

    print(f"Average Price       : ₹{df['price'].mean():,.2f}")

    print(f"Average Rating      : {df['rating'].mean():.2f}")

    print(f"Minimum Price       : ₹{df['price'].min():,.2f}")

    print(f"Maximum Price       : ₹{df['price'].max():,.2f}")

    print("-" * 50)

    print()

# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize text before generating embeddings.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.lower()

    text = text.replace("\n", " ")

    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text


# ============================================================
# CREATE SEARCH DOCUMENT
# ============================================================

def create_search_document(row: pd.Series) -> str:
    """
    Build a rich semantic search document for each product.
    """

    document = f"""
    Product Name : {row.get('product_name','')}

    Category : {row.get('category','')}

    Sub Category : {row.get('sub_category','')}

    Brand : {row.get('brand','')}

    Price : ₹{row.get('price','')}

    Rating : {row.get('rating','')}

    Description : {row.get('description','')}

    Features : {row.get('features','')}

    Purpose : {row.get('purpose','')}

    Color : {row.get('color','')}

    RAM : {row.get('ram','')}

    Storage : {row.get('storage','')}

    Processor : {row.get('processor','')}

    Display : {row.get('display','')}

    Camera : {row.get('camera','')}

    Battery : {row.get('battery','')}

    Warranty : {row.get('warranty','')}

    Seller : {row.get('seller','')}
    """

    return normalize_text(document)


# ============================================================
# PREPARE SEARCH DOCUMENTS
# ============================================================

def prepare_documents(df: pd.DataFrame):

    logger.info("=" * 70)
    logger.info("Preparing Search Documents")
    logger.info("=" * 70)

    documents = []

    for _, row in tqdm(

        df.iterrows(),

        total=len(df),

        desc="Preparing Documents"

    ):

        documents.append(

            create_search_document(row)

        )

    logger.info(

        f"Total Search Documents : {len(documents)}"

    )

    logger.info("=" * 70)

    logger.info("Sample Search Document")

    logger.info("=" * 70)

    logger.info(documents[0])

    return documents

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():
    """
    Load SentenceTransformer model.
    """

    logger.info("=" * 70)
    logger.info("Loading Embedding Model")
    logger.info("=" * 70)

    try:

        model = SentenceTransformer(
            MODEL_NAME
        )

        logger.info(f"Embedding Model Loaded Successfully")

        return model

    except Exception as e:

        logger.error(f"Error Loading Model : {e}")

        raise e


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(
    model,
    documents
):
    """
    Generate embeddings using batch processing.
    """

    logger.info("=" * 70)
    logger.info("Generating Embeddings")
    logger.info("=" * 70)

    all_embeddings = []

    total_batches = (

        len(documents)

        + BATCH_SIZE

        - 1

    ) // BATCH_SIZE

    logger.info(f"Total Documents : {len(documents)}")
    logger.info(f"Batch Size : {BATCH_SIZE}")
    logger.info(f"Total Batches : {total_batches}")

    for start in tqdm(

        range(

            0,

            len(documents),

            BATCH_SIZE

        ),

        desc="Generating Embeddings"

    ):

        batch = documents[
            start:
            start + BATCH_SIZE
        ]

        batch_embeddings = model.encode(

            batch,

            batch_size=BATCH_SIZE,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=False

        )

        all_embeddings.append(
            batch_embeddings
        )

    embeddings = np.vstack(
        all_embeddings
    )

    logger.info(
        f"Embedding Shape : {embeddings.shape}"
    )

    return embeddings.astype(np.float32)


# ============================================================
# VALIDATE EMBEDDINGS
# ============================================================

def validate_embeddings(
    embeddings
):
    """
    Validate generated embeddings.
    """

    logger.info("=" * 70)
    logger.info("Validating Embeddings")
    logger.info("=" * 70)

    if embeddings is None:

        raise Exception(
            "Embeddings are None."
        )

    if len(embeddings) == 0:

        raise Exception(
            "No embeddings generated."
        )

    if embeddings.shape[1] != EMBEDDING_DIMENSION:

        raise Exception(

            f"""
Expected Dimension : {EMBEDDING_DIMENSION}

Found : {embeddings.shape[1]}
"""

        )

    logger.info("Embedding Validation Successful")


# ============================================================
# EMBEDDING SUMMARY
# ============================================================

def embedding_summary(df, embeddings):
    """
    Print embedding summary.
    """

    logger.info("=" * 70)
    logger.info("Embedding Summary")
    logger.info("=" * 70)

    print()

    print("-" * 60)
    print(f"Products Indexed     : {len(df)}")
    print(f"Embedding Shape      : {embeddings.shape}")
    print(f"Embedding Dimension  : {embeddings.shape[1]}")
    print(f"Embedding Data Type  : {embeddings.dtype}")
    print("-" * 60)

    print()

# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create FAISS Index using cosine similarity.
    """

    logger.info("=" * 70)
    logger.info("Creating FAISS Index")
    logger.info("=" * 70)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    logger.info(f"Indexed Products : {index.ntotal}")

    return index


# ============================================================
# SAVE FILES
# ============================================================

def save_embeddings(embeddings):

    np.save(EMBEDDINGS_FILE, embeddings)

    logger.info(f"Embeddings Saved : {EMBEDDINGS_FILE}")


def save_faiss_index(index):

    faiss.write_index(index, FAISS_INDEX_FILE)

    logger.info(f"FAISS Index Saved : {FAISS_INDEX_FILE}")


def save_product_mapping(df):

    mapping = df.to_dict("records")

    with open(PRODUCT_MAPPING_FILE, "wb") as f:

        pickle.dump(mapping, f)

    logger.info(
        f"Product Mapping Saved : {PRODUCT_MAPPING_FILE}"
    )


def save_metadata(df, embeddings):

    metadata = {

        "total_products": len(df),

        "embedding_dimension": embeddings.shape[1],

        "embedding_model": MODEL_NAME,

        "batch_size": BATCH_SIZE,

        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")

    }

    with open(METADATA_FILE, "w") as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    logger.info(f"Metadata Saved : {METADATA_FILE}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    try:

        logger.info("=" * 70)
        logger.info("STARTING EMBEDDING PIPELINE")
        logger.info("=" * 70)

        # Load Dataset
        df = load_dataset(DATASET_PATH)

        validate_dataset(df)

        df = clean_dataset(df)

        dataset_summary(df)

        # Create Search Documents
        documents = prepare_documents(df)

        # Load Model
        model = load_embedding_model()

        # Generate Embeddings
        embeddings = generate_embeddings(
            model,
            documents
        )

        validate_embeddings(
            embeddings
        )

        embedding_summary(
            df,
            embeddings
        )

        # Create FAISS
        index = create_faiss_index(
            embeddings
        )

        # Save Files
        save_embeddings(
            embeddings
        )

        save_faiss_index(
            index
        )

        save_product_mapping(
            df
        )

        save_metadata(
            df,
            embeddings
        )

        elapsed = time.time() - START_TIME

        logger.info("=" * 70)
        logger.info("EMBEDDING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

        print()

        print("=" * 60)

        print("Embedding Generation Completed Successfully!")

        print("=" * 60)

        print(f"Products Indexed     : {len(df)}")

        print(f"Embeddings File      : {EMBEDDINGS_FILE}")

        print(f"FAISS Index          : {FAISS_INDEX_FILE}")

        print(f"Product Mapping      : {PRODUCT_MAPPING_FILE}")

        print(f"Metadata             : {METADATA_FILE}")

        print(f"Execution Time       : {elapsed:.2f} sec")

        print("=" * 60)

        print()

    except Exception as e:

        logger.exception("Embedding Pipeline Failed")

        print(f"\nError : {e}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()