"""
============================================================
Bundle Engine
============================================================

Generates intelligent bundle recommendations using

    • Query Parser
    • Conversation Memory
    • Semantic Search (FAISS)

This module is responsible for

✓ Bundle Detection
✓ Accessory Recommendation
✓ Complementary Products
✓ Context-aware Bundling

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
# BUNDLE ENGINE
# ============================================================

class BundleEngine:

    """
    Intelligent Bundle Recommendation Engine
    """

    def __init__(self):

        logging.info("=" * 60)
        logging.info("Initializing Bundle Engine")
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

        self.products = pd.read_csv(
            "data/products.csv"
        )

        logging.info(
            f"Products Loaded : {len(self.products)}"
        )

        logging.info("Bundle Engine Ready")


    # ============================================================
    # BUILD USER CONTEXT
    # ============================================================

    def build_context(self, query):

        """
        Merge parsed query with conversation memory.
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
    # RETRIEVE MAIN PRODUCT
    # ============================================================

    def retrieve_main_product(
        self,
        query
    ):

        """
        Retrieve the main product using semantic search.
        """

        logging.info("=" * 60)
        logging.info("Searching Main Product")
        logging.info("=" * 60)

        results = self.semantic_search.search(

            query=query,

            top_k=20

        )

        if len(results) == 0:

            logging.warning(
                "No Main Product Found"
            )

            return None

        context = self.memory.build_context()

        user_category = str(
            context.get("category", "")
        ).lower()

        for product in results:

            category = str(
                product.get("main_category", "")
            ).lower()

            if user_category in category:

                logging.info(
                    f"Main Product : {product['product_name']}"
                )

                return product

        logging.info(
            f"Main Product : {results[0]['product_name']}"
        )

        return results[0]
    
    # ============================================================
    # FIND BUNDLE PRODUCTS
    # ============================================================

    def find_bundle_products(
        self,
        main_product,
        top_k=10
    ):

        """
        Retrieve complementary products for the bundle.
        """

        logging.info("=" * 60)
        logging.info("Finding Bundle Products")
        logging.info("=" * 60)

        category = str(
            main_product.get(
                "main_category",
                ""
            )
        )

        brand = str(
            main_product.get(
                "brand",
                ""
            )
        )

        if brand.lower() == "unknown" or brand.strip() == "":

            query = category

        else:

            query = f"{brand} {category}"

        results = self.semantic_search.search(

            query=query,

            top_k=top_k

        )

        bundle = []

        for product in results:

            # Skip the main product
            if product["product_name"] == main_product["product_name"]:
                continue

            bundle.append(product)

        logging.info(
            f"Bundle Products Found : {len(bundle)}"
        )

        return bundle


    # ============================================================
    # CREATE BUNDLE
    # ============================================================

    def create_bundle(
        self,
        query
    ):

        """
        Generate a complete bundle.
        """

        logging.info("=" * 60)
        logging.info("Creating Bundle")
        logging.info("=" * 60)

        # Step 1
        context = self.build_context(query)

        # Step 2
        main_product = self.retrieve_main_product(query)

        if main_product is None:

            logging.warning(
                "Unable to create bundle."
            )

            return None

        # Step 3
        bundle_products = self.find_bundle_products(
            main_product
        )

        bundle = {

            "context": context,

            "main_product": main_product,

            "bundle_products": bundle_products

        }

        logging.info("Bundle Created Successfully")

        return bundle
    
        # ============================================================
    # DISPLAY BUNDLE
    # ============================================================

    def display_bundle(self, bundle):

        """
        Display the generated bundle.
        """

        if bundle is None:

            print("\nNo Bundle Found.\n")

            return

        main_product = bundle["main_product"]

        bundle_products = bundle["bundle_products"]

        print("\n" + "=" * 70)
        print("RECOMMENDED BUNDLE")
        print("=" * 70)

        print("\nMAIN PRODUCT\n")

        print(f"Product : {main_product['product_name']}")
        print(f"Brand   : {main_product['brand']}")
        print(f"Price   : ₹{main_product['price']}")
        print(f"Rating  : {main_product['rating']}")

        print("\n" + "-" * 70)
        print("BUNDLE PRODUCTS")
        print("-" * 70)

        total_price = float(main_product["price"])

        for index, product in enumerate(bundle_products, start=1):

            print(f"\n{index}.")

            print(f"Product : {product['product_name']}")
            print(f"Brand   : {product['brand']}")
            print(f"Price   : ₹{product['price']}")
            print(f"Rating  : {product['rating']}")

            total_price += float(product["price"])

        print("\n" + "=" * 70)
        print(f"Total Bundle Price : ₹{total_price:.2f}")
        print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("BUNDLE ENGINE STARTED")
    logging.info("=" * 60)

    engine = BundleEngine()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if not query:

            print("Please enter a query.\n")

            continue

        if query.lower() == "exit":

            print("\nBundle Engine Closed.")

            break

        try:

            bundle = engine.create_bundle(query)

            engine.display_bundle(bundle)

        except Exception as e:

            logging.error(f"Bundle Error : {e}")