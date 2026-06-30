"""
============================================================
AI SHOPPING ASSISTANT
Vector Database Manager
============================================================

Responsible for:

1. Load Embedding Model
2. Load FAISS Index
3. Load Product Mapping
4. Convert Query → Embedding
5. Semantic Search

Author : Riya Patel
============================================================
"""

import os
import pickle
import logging

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

FAISS_INDEX_PATH = "embeddings/faiss.index"

PRODUCT_MAPPING_PATH = "embeddings/product_mapping.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================
# VECTOR DATABASE
# ============================================================

class VectorDB:

    def __init__(self):

        logger.info("=" * 70)
        logger.info("Initializing Vector Database")
        logger.info("=" * 70)

        self.model = self.load_model()

        self.index = self.load_index()

        self.products = self.load_mapping()

        logger.info("Vector Database Ready")


    # ============================================================
    # LOAD MODEL
    # ============================================================

    def load_model(self):

        logger.info("Loading Embedding Model...")

        model = SentenceTransformer(MODEL_NAME)

        logger.info("Embedding Model Loaded")

        return model


    # ============================================================
    # LOAD FAISS
    # ============================================================

    def load_index(self):

        logger.info("Loading FAISS Index...")

        if not os.path.exists(FAISS_INDEX_PATH):

            raise FileNotFoundError(
                f"FAISS index not found : {FAISS_INDEX_PATH}"
            )

        index = faiss.read_index(FAISS_INDEX_PATH)

        logger.info(f"Indexed Products : {index.ntotal}")

        return index


    # ============================================================
    # LOAD PRODUCT MAPPING
    # ============================================================

    def load_mapping(self):

        logger.info("Loading Product Mapping...")

        if not os.path.exists(PRODUCT_MAPPING_PATH):

            raise FileNotFoundError(
                f"Mapping file not found : {PRODUCT_MAPPING_PATH}"
            )

        with open(PRODUCT_MAPPING_PATH, "rb") as f:

            mapping = pickle.load(f)

        logger.info(f"Products Loaded : {len(mapping)}")

        return mapping


    # ============================================================
    # QUERY EMBEDDING
    # ============================================================

    def embed_query(self, query: str):

        embedding = self.model.encode(

            query,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        return embedding.astype(np.float32)


    # ============================================================
    # SEMANTIC SEARCH
    # ============================================================

    def search(

        self,

        query,

        top_k=10

    ):

        query_embedding = self.embed_query(query)

        query_embedding = np.expand_dims(

            query_embedding,

            axis=0

        )

        scores, indices = self.index.search(

            query_embedding,

            top_k

        )

        results = []

        for score, idx in zip(

            scores[0],

            indices[0]

        ):

            if idx == -1:

                continue

            product = self.products[idx].copy()

            product["similarity_score"] = float(score)

            results.append(product)

        return results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    db = VectorDB()

    results = db.search(

        "Best gaming laptop under 70000",

        top_k=5

    )

    print()

    print("=" * 70)

    print("SEARCH RESULTS")

    print("=" * 70)

    for product in results:

        print()

        print(product["product_name"])

        print(product["brand"])

        print(product["price"])

        print(product["similarity_score"])