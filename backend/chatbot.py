"""
============================================================
AI SHOPPING ASSISTANT
Chatbot Controller
============================================================

Central controller for the AI Shopping Assistant.

Responsibilities

✓ Parse user query
✓ Detect intent
✓ Route request to the correct engine
✓ Maintain conversation memory
✓ Generate chatbot response

Uses

    • Query Parser
    • Conversation Memory
    • Recommendation Engine
    • Comparison Engine
    • Bundle Engine

Future Integration

    • LangChain
    • LangGraph
    • FastAPI
    • Streamlit

"""

# ============================================================
# IMPORTS
# ============================================================

import logging


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:

    from backend.query_parser import QueryParser

    from backend.conversation_memory import ConversationMemory

    from backend.recommendation_engine import RecommendationEngine

    from backend.comparison_engine import ComparisonEngine

    from backend.bundle_engine import BundleEngine

except ImportError:

    from query_parser import QueryParser

    from conversation_memory import ConversationMemory

    from recommendation_engine import RecommendationEngine

    from comparison_engine import ComparisonEngine

    from bundle_engine import BundleEngine


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)


# ============================================================
# CHATBOT
# ============================================================

class ShoppingChatbot:

    """
    Central AI Shopping Assistant
    """

    def __init__(self):

        logging.info("=" * 60)
        logging.info("Initializing Shopping Chatbot")
        logging.info("=" * 60)

        # --------------------------------------------
        # Core Components
        # --------------------------------------------

        self.parser = QueryParser()

        self.memory = ConversationMemory()

        self.recommendation_engine = RecommendationEngine()

        self.comparison_engine = ComparisonEngine()

        self.bundle_engine = BundleEngine()

        logging.info("Shopping Chatbot Ready")


    # ============================================================
    # PARSE USER QUERY
    # ============================================================

    def parse_query(self, query):

        """
        Parse the user query.
        """

        logging.info("=" * 60)
        logging.info("Parsing User Query")
        logging.info("=" * 60)

        parsed_query = self.parser.parse_query(query)

        logging.info(parsed_query)

        return parsed_query


    # ============================================================
    # UPDATE CONVERSATION MEMORY
    # ============================================================

    def update_memory(
        self,
        parsed_query
    ):

        """
        Update conversation memory.
        """

        self.memory.update_memory(parsed_query)

        logging.info("Conversation Memory Updated")

        # ============================================================
    # ROUTE USER REQUEST
    # ============================================================

    def route_query(
        self,
        query
    ):

        """
        Route the user query to the appropriate engine.
        """

        logging.info("=" * 60)
        logging.info("Routing User Query")
        logging.info("=" * 60)

        # --------------------------------------------
        # Parse Query
        # --------------------------------------------

        parsed_query = self.parse_query(query)

        # --------------------------------------------
        # Update Memory
        # --------------------------------------------

        self.update_memory(parsed_query)

        # --------------------------------------------
        # Detect Intent
        # --------------------------------------------

        intent = parsed_query.get(
            "intent",
            "general"
        )

        logging.info(
            f"Detected Intent : {intent}"
        )

        # --------------------------------------------
        # Recommendation
        # --------------------------------------------

        if intent == "recommendation":

            logging.info(
                "Calling Recommendation Engine"
            )

            products = self.recommendation_engine.recommend(
                query
            )

            return {

                "intent": intent,

                "response": products

            }

        # --------------------------------------------
        # Comparison
        # --------------------------------------------

        elif intent == "comparison":

            logging.info(
                "Calling Comparison Engine"
            )

            comparison = self.comparison_engine.compare(
                query
            )

            return {

                "intent": intent,

                "response": comparison

            }

        # --------------------------------------------
        # Bundle
        # --------------------------------------------

        elif intent == "bundle":

            logging.info(
                "Calling Bundle Engine"
            )

            bundle = self.bundle_engine.create_bundle(
                query
            )

            return {

                "intent": intent,

                "response": bundle

            }

        # --------------------------------------------
        # General Chat
        # --------------------------------------------

        logging.info(
            "General Conversation"
        )

        return {

            "intent": "general",

            "response": (
                "I'm your AI Shopping Assistant. "
                "I can recommend products, compare items, "
                "and suggest bundles."
            )

        }
    
        # ============================================================
    # DISPLAY RESPONSE
    # ============================================================

    def display_response(self, result):

        """
        Display chatbot response.
        """

        intent = result["intent"]

        response = result["response"]

        print("\n" + "=" * 70)
        print(f"INTENT : {intent.upper()}")
        print("=" * 70)

        # --------------------------------------------
        # Recommendation
        # --------------------------------------------

        if intent == "recommendation":

            self.recommendation_engine.display_recommendations(
                response
            )

        # --------------------------------------------
        # Comparison
        # --------------------------------------------

        elif intent == "comparison":

            self.comparison_engine.display_comparison(
                response
            )

        # --------------------------------------------
        # Bundle
        # --------------------------------------------

        elif intent == "bundle":

            self.bundle_engine.display_bundle(
                response
            )

        # --------------------------------------------
        # General Chat
        # --------------------------------------------

        else:

            print(response)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("SHOPPING CHATBOT STARTED")
    logging.info("=" * 60)

    chatbot = ShoppingChatbot()

    print("\nWelcome to AI Shopping Assistant!")
    print("Type 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if not query:

            print("Please enter a query.\n")

            continue

        if query.lower() == "exit":

            print("\nGoodbye!")

            break

        try:

            result = chatbot.route_query(query)

            chatbot.display_response(result)

        except Exception as e:

            logging.error(
                f"Chatbot Error : {e}"
            )