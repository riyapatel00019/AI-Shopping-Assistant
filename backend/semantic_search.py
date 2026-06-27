"""
============================================================
AI SHOPPING ASSISTANT
Semantic Search Engine
============================================================

Purpose:
--------
This module performs ONLY semantic retrieval.

Workflow:
User Query
    ↓
Sentence Transformer
    ↓
Query Embedding
    ↓
FAISS Search
    ↓
Top K Similar Products

This module should NOT:
-----------------------
❌ Detect Intent
❌ Filter Budget
❌ Generate LLM Response
❌ Store Conversation Memory
❌ Compare Products
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import pickle
import logging
import warnings

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

FAISS_INDEX_PATH = "vector_db/faiss_index.bin"

PRODUCT_MAPPING_PATH = "embeddings/product_mapping.pkl"

TOP_K = 10


# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# SEMANTIC SEARCH ENGINE
# ============================================================

class SemanticSearchEngine:

    def __init__(self):

        self.model = None
        self.index = None
        self.mapping = None

        self.load_resources()


# ============================================================
# LOAD ALL RESOURCES
# ============================================================

    def load_resources(self):

        logger.info("=" * 60)
        logger.info("Loading Semantic Search Resources")
        logger.info("=" * 60)

        self.load_model()

        self.load_faiss()

        self.load_mapping()

        self.validate()


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

    def load_model(self):

        logger.info("Loading Embedding Model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        logger.info("Embedding Model Loaded Successfully")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

    def load_faiss(self):

        logger.info("Loading FAISS Index...")

        if not os.path.exists(
            FAISS_INDEX_PATH
        ):
            raise FileNotFoundError(
                FAISS_INDEX_PATH
            )

        self.index = faiss.read_index(
            FAISS_INDEX_PATH
        )

        logger.info(
            f"Indexed Products : {self.index.ntotal}"
        )


# ============================================================
# LOAD PRODUCT MAPPING
# ============================================================

    def load_mapping(self):

        logger.info("Loading Product Mapping...")

        if not os.path.exists(
            PRODUCT_MAPPING_PATH
        ):
            raise FileNotFoundError(
                PRODUCT_MAPPING_PATH
            )

        with open(
            PRODUCT_MAPPING_PATH,
            "rb"
        ) as file:

            self.mapping = pickle.load(file)

        logger.info(
            f"Mapping Loaded : {len(self.mapping)}"
        )


# ============================================================
# VALIDATE RESOURCES
# ============================================================

    def validate(self):

        if self.index.ntotal != len(self.mapping):

            raise Exception(
                "Index and Mapping Size Mismatch."
            )

        logger.info("Semantic Search Ready")


# ============================================================
# CREATE QUERY EMBEDDING
# ============================================================

    def create_embedding(self, query):

        embedding = self.model.encode(

            [query],

            convert_to_numpy=True,

            normalize_embeddings=True

        )

        return embedding.astype("float32")
    
# ============================================================
# SEARCH FAISS
# ============================================================

    def search(self, query, top_k=TOP_K):

        logger.info("=" * 60)
        logger.info("Semantic Search Started")
        logger.info("=" * 60)

        logger.info(f"Query : {query}")

        # ---------------------------------------------
        # Create Query Embedding
        # ---------------------------------------------

        query_embedding = self.create_embedding(query)

        # ---------------------------------------------
        # Search FAISS
        # ---------------------------------------------

        scores, indices = self.index.search(

            query_embedding,

            top_k

        )

        logger.info(

            f"Retrieved {len(indices[0])} products"

        )

        # ---------------------------------------------
        # Convert Search Results
        # ---------------------------------------------

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            product = self.mapping[idx].copy()

            product["similarity_score"] = round(
                float(score),
                4
            )

            results.append(product)

        logger.info(

            f"Returned {len(results)} products"

        )

        return results


# ============================================================
# SEARCH BY EMBEDDING
# ============================================================

    def search_embedding(self, embedding, top_k=TOP_K):

        scores, indices = self.index.search(

            embedding,

            top_k

        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            product = self.mapping[idx].copy()

            product["similarity_score"] = round(
                float(score),
                4
            )

            results.append(product)

        return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

    def display_results(self, results):

        logger.info("=" * 60)
        logger.info("Semantic Search Results")
        logger.info("=" * 60)

        print()

        for i, product in enumerate(results, start=1):

            print("=" * 70)

            print(f"Rank              : {i}")

            print(f"Product           : {product['product_name']}")

            print(f"Brand             : {product['brand']}")

            print(
                f"Main Category     : "
                f"{product.get('main_category', 'N/A')}"
            )

            print(
                f"Sub Category      : "
                f"{product.get('sub_category', 'N/A')}"
            )

            print(f"Price             : ₹{product['price']}")

            print(f"Rating            : {product['rating']}")

            print(f"Similarity Score  : {product['similarity_score']}")

        print()


# ============================================================
# GET PRODUCT BY INDEX
# ============================================================

    def get_product(self, index):

        if index not in self.mapping:

            return None

        return self.mapping[index]


# ============================================================
# TOTAL PRODUCTS
# ============================================================

    def total_products(self):

        return self.index.ntotal
# ============================================================
# INTERACTIVE SEARCH
# ============================================================

    def interactive_search(self):

        logger.info("=" * 60)
        logger.info("Interactive Semantic Search")
        logger.info("=" * 60)

        print("\nType 'exit' to quit.\n")

        while True:

            query = input("Search Query : ").strip()

            if query.lower() == "exit":
                break

            if query == "":
                print("Please enter a valid query.\n")
                continue

            results = self.search(query)

            self.display_results(results)


# ============================================================
# QUICK SEARCH FUNCTION
# ============================================================

_engine = None


def semantic_search(query, top_k=TOP_K):

    """
    Reusable semantic search function.

    This is the function that other backend modules
    (Recommendation Engine, Chatbot, FastAPI, LangGraph)
    will call.
    """

    global _engine

    if _engine is None:

        _engine = SemanticSearchEngine()

    return _engine.search(

        query=query,

        top_k=top_k

    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("SEMANTIC SEARCH ENGINE STARTED")
    logger.info("=" * 60)

    engine = SemanticSearchEngine()

    print("\nSemantic Search Engine Ready!")

    print(f"Indexed Products : {engine.total_products()}")

    print()

    engine.interactive_search()

    logger.info("=" * 60)
    logger.info("SEMANTIC SEARCH ENGINE CLOSED")
    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()