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



# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from backend.semantic_search import SemanticSearchEngine
from backend.query_understanding_engine import QueryUnderstandingEngine
from backend.ranking_engine import ContextAwareRankingEngine
from backend.conversation_memory import ConversationMemory
from agent.llm_manager import LLMManager

    


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

        # LLM
        self.llm = LLMManager().get_llm()

        # Query Understanding
        self.query_engine = QueryUnderstandingEngine(
            self.llm
        )

        # Conversation Memory
        self.memory = ConversationMemory()

        # Semantic Search
        self.semantic_search = SemanticSearchEngine()

        # Context Ranking
        self.ranking_engine = ContextAwareRankingEngine()


        logging.info("Recommendation Engine Ready")
    
        # ============================================================
    # BUILD USER CONTEXT
    # ============================================================

    def build_context(self, query):

        """
        Merge parsed query with conversation memory.
        Current query has higher priority than memory.
        """

        shopping_context = self.query_engine.understand(
            query
        )

        self.memory.update_memory(
            shopping_context
        )

        context = self.memory.build_context()

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

                brands = context["brand"]

                if isinstance(brands, str):
                    brands = [brands]

                brands = [b.lower() for b in brands]

                if str(product["brand"]).lower() not in brands:
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

    def recommend(self, query, top_k=5):

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
            top_k=30
        )

        # Step 3
        filtered = self.filter_products(
            candidates,
            context
        )

        if len(filtered) == 0:

            logging.warning(
                "No products after filtering. Using semantic candidates."
            )

            filtered = candidates[:10]


        ranked = self.ranking_engine.rank_products(

            query=query,

            products=filtered,

            shopping_context=context,

            top_k=top_k

        )

        return ranked
        

    # ============================================================
    # DISPLAY RECOMMENDATIONS
    # ============================================================

    def display_recommendations(self, products):

        """
        Display Beautiful Recommendations
        """

        if len(products) == 0:

            print("\n❌ No Products Found.\n")

            return

        print("\n")
        print("=" * 90)
        print("🛍 AI SHOPPING ASSISTANT RECOMMENDATIONS")
        print("=" * 90)

        for index, product in enumerate(products, start=1):

            print(f"\n🏆 Recommendation #{index}")
            print("-" * 90)

            print(f"📦 Product        : {product.product_name}")

            print(f"🏷 Brand          : {product.brand}")

            print(f"📂 Category       : {product.category}")

            print(f"📁 Sub Category   : {product.sub_category}")

            print(f"💰 Price          : ₹{product.price:,.0f}")

            print(f"⭐ Rating         : {product.rating}/5")

            print(f"🎯 Match Score    : {product.final_score:.3f}")

            print()

            print("✅ Why Recommended")

            print(f"   {product.explanation}")

            print()

            print(f"🖼 Image URL")

            print(product.image_url)

            print()

            print("🔗 Product URL")

            print(product.product_url)

            print()

            print("=" * 90)

            

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