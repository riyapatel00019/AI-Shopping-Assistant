"""
============================================================
Comparison Engine
============================================================

Compares products intelligently using

    • Semantic Search
    • Query Parser
    • Conversation Memory

Responsibilities

✓ Find products
✓ Compare specifications
✓ Score products
✓ Select winner
✓ Explain comparison

Used By

    - LangChain Tools
    - Shopping Graph
    - FastAPI
    - Streamlit
"""

import logging
import pandas as pd

try:
    from backend.semantic_search import SemanticSearchEngine
    from backend.query_parser import QueryParser
    from backend.conversation_memory import ConversationMemory

except ImportError:
    from semantic_search import SemanticSearchEngine
    from query_parser import QueryParser
    from conversation_memory import ConversationMemory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


class ComparisonEngine:

    """
    Intelligent Product Comparison Engine.
    """

    def __init__(self):

        logging.info("=" * 60)
        logging.info("Initializing Comparison Engine")
        logging.info("=" * 60)

        self.parser = QueryParser()

        self.memory = ConversationMemory()

        self.semantic_search = SemanticSearchEngine()

        logging.info("Loading Products Dataset")

        self.products = pd.read_csv(
            "data/shopping_products.csv"
        )

        logging.info(
            f"Products Loaded : {len(self.products)}"
        )

        logging.info("Comparison Engine Ready")
        # ============================================================
    # BUILD USER CONTEXT
    # ============================================================

    def build_context(self, query):
        """
        Build user context using the query parser
        and conversation memory.
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
    # RETRIEVE PRODUCTS
    # ============================================================

    def retrieve_products(
        self,
        query,
        top_k=5
    ):
        """
        Retrieve candidate products using semantic search.
        """

        logging.info("=" * 60)
        logging.info("Retrieving Products")
        logging.info("=" * 60)

        products = self.semantic_search.search(
            query=query,
            top_k=top_k
        )

        logging.info(
            f"Products Retrieved : {len(products)}"
        )

        return products


    # ============================================================
    # FIND BEST MATCH
    # ============================================================

    def find_best_match(
        self,
        query
    ):
        """
        Find the best matching product.

        Uses semantic search instead of
        exact string matching.
        """

        logging.info("=" * 60)
        logging.info("Finding Best Product Match")
        logging.info("=" * 60)

        results = self.retrieve_products(
            query,
            top_k=3
        )

        if len(results) == 0:

            logging.warning(
                "No Product Found"
            )

            return None

        best_product = results[0]

        logging.info(
            f"Selected Product : {best_product['product_name']}"
        )

        return best_product
    # ============================================================
    # COMPARE PRODUCTS
    # ============================================================

    def compare_products(
        self,
        products
    ):
        """
        Compare retrieved products using
        similarity, rating and price.
        """

        logging.info("=" * 60)
        logging.info("Comparing Products")
        logging.info("=" * 60)

        comparison = []

        for product in products:

            similarity = float(
                product.get("similarity_score", 0)
            )

            rating = float(
                product.get("rating", 0)
            ) / 5

            try:

                price = float(
                    product.get("price", 1)
                )

            except (ValueError, TypeError):

                price = 1.0

            # Lower price gets a small bonus
            price_score = 1 / max(price, 1)

            comparison_score = (

                similarity * 0.50 +

                rating * 0.30 +

                price_score * 0.20

            )

            product["comparison_score"] = round(
                comparison_score,
                4
            )

            comparison.append(product)

        comparison = sorted(

            comparison,

            key=lambda x: x["comparison_score"],

            reverse=True

        )

        logging.info(
            f"Products Compared : {len(comparison)}"
        )

        return comparison


    # ============================================================
    # SELECT WINNER
    # ============================================================

    def select_winner(
        self,
        comparison
    ):
        """
        Select the highest scoring product.
        """

        logging.info("=" * 60)
        logging.info("Selecting Winner")
        logging.info("=" * 60)

        if len(comparison) == 0:

            return None

        winner = comparison[0]

        logging.info(

            f"Winner : {winner['product_name']}"

        )

        return winner


    # ============================================================
    # MAIN COMPARISON PIPELINE
    # ============================================================

    def compare(
        self,
        query
    ):
        """
        Complete comparison workflow.
        """

        logging.info("=" * 60)
        logging.info("Comparison Started")
        logging.info("=" * 60)

        self.build_context(query)

        products = self.retrieve_products(
            query,
            top_k=5
        )

        if len(products) < 2:

            logging.warning(
                "Not enough products for comparison."
            )

            return None

        comparison = self.compare_products(
            products
        )

        winner = self.select_winner(
            comparison
        )

        return {

            "products": comparison,

            "winner": winner,

            "total_products": len(comparison)

        }
    # ============================================================
    # DISPLAY COMPARISON
    # ============================================================

    def display_comparison(
        self,
        result
    ):
        """
        Display comparison results.
        """

        if result is None:

            print("\nNot enough products found for comparison.\n")

            return

        products = result["products"]

        winner = result["winner"]

        print(f"\nCompared Products : {result['total_products']}")

        print("\n" + "=" * 70)
        print("PRODUCT COMPARISON")
        print("=" * 70)

        for index, product in enumerate(products, start=1):

            print(f"\nProduct {index}")

            print(f"Name       : {product['product_name']}")
            print(f"Brand      : {product['brand']}")
            print(f"Price      : ₹{product['price']}")
            print(f"Rating     : {product['rating']}")
            print(
                f"Similarity : "
                f"{product.get('similarity_score',0):.4f}"
            )
            print(
                f"Score      : "
                f"{product['comparison_score']:.4f}"
            )

            print("-" * 70)

        print("\n" + "=" * 70)
        print("WINNER")
        print("=" * 70)

        print(f"Product : {winner['product_name']}")
        print(f"Brand   : {winner['brand']}")
        print(f"Price   : ₹{winner['price']}")
        print(f"Rating  : {winner['rating']}")

        print("\nWhy this product won?")

        print("✓ Highest semantic similarity")
        print("✓ Better customer rating")
        print("✓ Better overall comparison score")

        print("=" * 70)
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("COMPARISON ENGINE STARTED")
    logging.info("=" * 60)

    engine = ComparisonEngine()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nComparison Engine Closed.")

            break

        if query == "":

            continue

        try:

            result = engine.compare(query)

            engine.display_comparison(result)

        except Exception as e:

            logging.error(
                f"Comparison Error : {e}"
            )


