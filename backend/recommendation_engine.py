"""
============================================================
Recommendation Engine
============================================================

Generates intelligent product recommendations using

    • Query Parser
    • Conversation Memory
    • Semantic Search (FAISS)

This module is responsible for

✓ Context-aware recommendations
✓ Budget filtering
✓ Brand filtering
✓ Category filtering
✓ Product ranking

Used By

    - Chatbot
    - FastAPI
    - Streamlit
    - LangGraph

"""

import logging
import pandas as pd


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:

    from backend.semantic_search import SemanticSearchEngine

    from backend.query_parser import QueryParser

    from backend.conversation_memory import ConversationMemory

except ImportError:

    from semantic_search import SemanticSearchEngine

    from query_parser import QueryParser

    from conversation_memory import ConversationMemory


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

class RecommendationEngine:

    """
    Intelligent Recommendation Engine
    """

    def __init__(self):

        logging.info("=" * 60)
        logging.info("Initializing Recommendation Engine")
        logging.info("=" * 60)

        # --------------------------------------------
        # Load Modules
        # --------------------------------------------

        self.parser = QueryParser()

        self.memory = ConversationMemory()

        self.semantic_search = SemanticSearchEngine()

        # --------------------------------------------
        # Load Dataset
        # --------------------------------------------

        logging.info("Loading Products Dataset")

        self.products = pd.read_csv("data/products.csv")

        logging.info(
            f"Products Loaded : {len(self.products)}"
        )

        logging.info("Recommendation Engine Ready")
    
        # ============================================================
    # BUILD USER CONTEXT
    # ============================================================

    def build_context(self, query):

        """
        Merge parsed query with conversation memory.
        Current query has higher priority than memory.
        """

        parsed = self.parser.parse_query(query)

        self.memory.update_memory(parsed)

        context = self.memory.build_context()

        logging.info("=" * 60)
        logging.info("Current User Context")
        logging.info("=" * 60)

        logging.info(context)

        return context


    # ============================================================
    # RETRIEVE CANDIDATES
    # ============================================================

    def retrieve_candidates(self, query, top_k=50):

        """
        Retrieve semantically similar products.
        """

        logging.info("=" * 60)
        logging.info("Retrieving Candidate Products")
        logging.info("=" * 60)

        results = self.semantic_search.search(

            query=query,

            top_k=top_k

        )

        logging.info(

            f"Candidates Retrieved : {len(results)}"

        )

        return results


    # ============================================================
    # FILTER PRODUCTS
    # ============================================================

    def filter_products(self, products, context):

        """
        Apply business filters.
        """

        logging.info("=" * 60)
        logging.info("Filtering Products")
        logging.info("=" * 60)

        filtered = []

        for product in products:

            # ----------------------------
            # Budget Filter
            # ----------------------------

            if context.get("budget"):

                if product["price"] > context["budget"]:

                    continue

            # ----------------------------
            # Brand Filter
            # ----------------------------

            if context.get("brand"):

                if str(product["brand"]).lower() != \
                   str(context["brand"]).lower():

                    continue

            # ----------------------------
            # Category Filter
            # ----------------------------

            if context.get("category"):

                main_category = str(
                    product.get("main_category", "")
                ).lower()

                sub_category = str(
                    product.get("sub_category", "")
                ).lower()

                user_category = context["category"].lower()

                if (
                    user_category not in main_category and
                    user_category not in sub_category
                ):
                    continue

            filtered.append(product)

        logging.info(

            f"Products After Filtering : {len(filtered)}"

        )

        return filtered


    # ============================================================
    # GENERATE RECOMMENDATIONS
    # ============================================================

    def recommend(self, query, top_k=10):

        """
        Main Recommendation Pipeline
        """

        logging.info("=" * 60)
        logging.info("Recommendation Started")
        logging.info("=" * 60)

        # Step 1
        context = self.build_context(query)

        # Step 2
        candidates = self.retrieve_candidates(
            query,
            top_k=50
        )

        # Step 3
        filtered = self.filter_products(
            candidates,
            context
        )

        if len(filtered) == 0:

            logging.warning(
                "No Products After Filtering."
            )

            logging.info(
                "Returning Semantic Search Results."
            )

            ranked = self.rank_products(candidates)

            return ranked[:top_k]
        # --------------------------------------------
        # Rank Filtered Products
        # --------------------------------------------

        ranked = self.rank_products(filtered)

        return ranked[:top_k]
    # ============================================================
    # RANK PRODUCTS
    # ============================================================

    def rank_products(self, products):

        """
        Rank products using similarity, rating and price.
        """

        logging.info("=" * 60)
        logging.info("Ranking Products")
        logging.info("=" * 60)

        if len(products) == 0:

            return []

        for product in products:

            similarity = float(
                product.get("similarity_score", 0)
            )

            rating = float(
                product.get("rating", 0)
            ) / 5

            price_bonus = 0

            if product.get("price", 0) > 0:

                price_bonus = 1 / product["price"]

            score = (

                similarity * 0.70 +

                rating * 0.25 +

                price_bonus * 0.05

            )

            product["final_score"] = score

        products = sorted(

            products,

            key=lambda x: x["final_score"],

            reverse=True

        )

        logging.info(

            f"Ranked Products : {len(products)}"

        )

        return products


    # ============================================================
    # DISPLAY RECOMMENDATIONS
    # ============================================================

    def display_recommendations(self, products):

        """
        Display recommended products.
        """

        if len(products) == 0:

            print("\nNo Products Found.\n")

            return

        print("\n" + "=" * 70)
        print("RECOMMENDED PRODUCTS")
        print("=" * 70)

        for index, product in enumerate(products, start=1):

            print(f"\nRank : {index}")

            print(f"Product : {product['product_name']}")

            print(f"Brand : {product['brand']}")

            print(
                f"Main Category : "
                f"{product.get('main_category', 'N/A')}"
            )

            print(
                f"Sub Category  : "
                f"{product.get('sub_category', 'N/A')}")
            
            print(f"Price : ₹{product['price']}")

            print(f"Rating : {product['rating']}")

            print(

                f"Similarity : "

                f"{product.get('similarity_score',0):.4f}"

            )

            print(

                f"Final Score : "

                f"{product['final_score']:.4f}"

            )

            print("-" * 70)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("RECOMMENDATION ENGINE STARTED")
    logging.info("=" * 60)

    engine = RecommendationEngine()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if not query:

            print("Please enter a query.\n")

            continue

        if query.lower() == "exit":

            print("\nRecommendation Engine Closed.")

            break

        try:

            products = engine.recommend(query)

            engine.display_recommendations(products)

        except Exception as e:

            logging.error(f"Recommendation Error : {e}")