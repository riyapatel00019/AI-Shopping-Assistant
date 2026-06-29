"""
============================================================
AI SHOPPING ASSISTANT
Shopping Agent
============================================================

Responsibilities
----------------
✓ Execute Shopping Graph
✓ Intent-based Routing
✓ Recommendation
✓ Comparison
✓ Bundle Creation
✓ Return Structured Response
"""

# ============================================================
# IMPORTS
# ============================================================

import logging

# ============================================================
# PROJECT MODULES
# ============================================================

from agent.shopping_graph import ShoppingGraph

from backend.recommendation_engine import RecommendationEngine

from backend.comparison_engine import ComparisonEngine

from backend.bundle_engine import BundleEngine

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)

# ============================================================
# SHOPPING AGENT
# ============================================================

class ShoppingAgent:

    """
    Main AI Shopping Assistant.
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Shopping Agent")
        logger.info("=" * 60)

        # --------------------------------------------
        # Shopping Graph
        # --------------------------------------------

        self.shopping_graph = ShoppingGraph()

        # --------------------------------------------
        # Recommendation Engine
        # --------------------------------------------

        self.recommendation_engine = RecommendationEngine()

        # --------------------------------------------
        # Comparison Engine
        # --------------------------------------------

        self.comparison_engine = ComparisonEngine()

        # --------------------------------------------
        # Bundle Engine
        # --------------------------------------------

        self.bundle_engine = BundleEngine()

        logger.info("Shopping Agent Ready")

        # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        query: str
    ):
        """
        Process a shopping query.

        Workflow

        User
            ↓
        Shopping Graph
            ↓
        Query Understanding
            ↓
        Intent Routing
            ↓
        Recommendation /
        Comparison /
        Bundle Engine
        """

        logger.info("=" * 60)
        logger.info("SHOPPING AGENT STARTED")
        logger.info("=" * 60)

        logger.info(f"User Query : {query}")

        # --------------------------------------------------------
        # Shopping Graph
        # --------------------------------------------------------

        graph_result = self.shopping_graph.chat(query)

        shopping_context = graph_result["shopping_context"]

        intent = graph_result["intent"]

        # --------------------------------------------------------
        # If follow-up is required
        # --------------------------------------------------------

        if graph_result["follow_up_required"]:

            return {

                "messages": graph_result["messages"],

                "shopping_context": shopping_context,

                "conversation_complete": graph_result["conversation_complete"],

                "response": graph_result["follow_up_question"],

                "provider": self.get_provider(),

                "products": [],

                "comparison": None,

                "bundle": None

            }

        # --------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------

        if intent == "recommendation":

            logger.info("Routing → Recommendation Engine")

            try:

                products = self.recommendation_engine.recommend(query)

                return {

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": graph_result["conversation_complete"],

                    "response": f"I found {len(products)} products matching your requirements.",

                    "provider": self.get_provider(),

                    "products": products,

                    "comparison": None,

                    "bundle": None

                }

            except Exception as e:

                logger.error(f"Recommendation Error : {e}")

                return {

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": False,

                    "response": "Sorry, I couldn't retrieve recommendations at the moment.",

                    "provider": self.get_provider(),

                    "products": [],

                    "comparison": None,

                    "bundle": None

                }

        # --------------------------------------------------------
        # Comparison
        # --------------------------------------------------------

        elif intent == "comparison":

            logger.info("Routing → Comparison Engine")

            try:

                comparison = self.comparison_engine.compare(query)

                return {

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": False,

                    "response": "Sorry, I couldn't compare those products.",

                    "provider": self.get_provider(),

                    "products": [],

                    "comparison": None,

                    "bundle": None

                }


            except Exception as e:

                logger.error(f"Comparison Error : {e}")


                return {

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": graph_result["conversation_complete"],

                    "response": "I compared the selected products.",

                    "provider": self.get_provider(),

                    "products": [],

                    "comparison": comparison,

                    "bundle": None

                }

        # --------------------------------------------------------
        # Bundle
        # --------------------------------------------------------

        elif intent == "bundle":

            logger.info("Routing → Bundle Engine")

            try:

                bundle = self.bundle_engine.create_bundle(query)

                return {

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": False,

                    "response": "Sorry, I couldn't create a shopping bundle.",

                    "provider": self.get_provider(),

                    "products": [],

                    "comparison": None,

                    "bundle": None

                }

            except Exception as e:

                logger.error(f"Bundle Error : {e}")

                return {    

                    "messages": graph_result["messages"],

                    "shopping_context": shopping_context,

                    "conversation_complete": graph_result["conversation_complete"],

                    "response": "I created a shopping bundle for you.",

                    "provider": self.get_provider(),

                    "products": [],

                    "comparison": None,

                    "bundle": bundle

                }

        # --------------------------------------------------------
        # Default
        # --------------------------------------------------------

        else:

            logger.info("Unknown intent → Recommendation Engine")

            products = self.recommendation_engine.recommend(query)

            return {

                "messages": graph_result["messages"],

                "shopping_context": shopping_context,

                "conversation_complete": graph_result["conversation_complete"],

                "response": f"I found {len(products)} products matching your requirements.",

                "provider": self.get_provider(),

                "products": products,

                "comparison": None,

                "bundle": None

            }
    # ============================================================
    # GET PROVIDER
    # ============================================================

    def get_provider(self):
        """
        Return the active LLM provider.
        """

        return self.shopping_graph.llm_manager.get_provider()


    # ============================================================
    # DISPLAY RESPONSE
    # ============================================================

    def display_response(
        self,
        result
    ):
        """
        Display chatbot response in terminal.
        Useful for CLI testing.
        """

        print("\n" + "=" * 70)
        print("AI SHOPPING ASSISTANT")
        print("=" * 70)

        print()

        print(result["response"])

        print()

        # --------------------------------------------------------
        # Recommended Products
        # --------------------------------------------------------

        products = result.get("products", [])

        if products:

            print("=" * 70)
            print("RECOMMENDED PRODUCTS")
            print("=" * 70)

            for i, product in enumerate(products, start=1):

                print(f"\nRank : {i}")

                print(f"Product : {product.product_name}")

                print(f"Brand : {product.brand}")

                print(f"Price : ₹{product.price}")

                print(f"Rating : {product.rating}")

                print(f"Reason : {product.explanation}")

                print("-" * 60)

        # --------------------------------------------------------
        # Comparison
        # --------------------------------------------------------

        comparison = result.get("comparison")

        if comparison:

            print("=" * 70)
            print("PRODUCT COMPARISON")
            print("=" * 70)

            winner = comparison.get("winner")

            if winner:

                print(
                    f"\n🏆 Winner : {winner['product_name']}"
                )

                print(
                    f"Brand : {winner['brand']}"
                )

                print(
                    f"Price : ₹{winner['price']}"
                )

        # --------------------------------------------------------
        # Bundle
        # --------------------------------------------------------

        bundle = result.get("bundle")

        if bundle:

            print("=" * 70)
            print("SHOPPING BUNDLE")
            print("=" * 70)

            main = bundle["main_product"]

            print(f"\nMain Product : {main['product_name']}")

            print(f"Price : ₹{main['price']}")

            print("\nBundle Products")

            for item in bundle["bundle_products"]:

                print(f"• {item['product_name']}")

        print("=" * 70)


    # ============================================================
    # PROVIDER INFORMATION
    # ============================================================

    def provider_info(self):
        """
        Display active LLM provider.
        """

        print("\n" + "=" * 60)

        print("ACTIVE LLM")

        print("=" * 60)

        print(

            f"Provider : {self.get_provider()}"

        )

        print("=" * 60)

    # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        query: str
    ):
        """
        Complete shopping assistant workflow.

        Used for CLI testing.
        """

        result = self.chat(query)

        self.display_response(result)

        return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("SHOPPING AGENT STARTED")
    logger.info("=" * 60)

    agent = ShoppingAgent()

    print("\n" + "=" * 70)
    print("🛒 CARTGENIUS AI")
    print("=" * 70)

    print("\nYour Intelligent Shopping Assistant")

    print("\nCapabilities")

    print("• Product Recommendation")

    print("• Product Comparison")

    print("• Shopping Bundles")

    print("• Budget Shopping")

    print("• Conversation Memory")

    print("• Intent Detection")

    print()

    agent.provider_info()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nThank you for using CartGenius AI!")

            break

        if not query:

            print("Please enter a query.\n")

            continue

        try:

            agent.ask(query)

        except Exception as e:

            logger.error(

                f"Shopping Agent Error : {e}"

            )

            print("\nSomething went wrong.")

    logger.info("=" * 60)
    logger.info("SHOPPING AGENT CLOSED")
    logger.info("=" * 60)