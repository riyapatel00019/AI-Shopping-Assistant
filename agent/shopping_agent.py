"""
============================================================
AI SHOPPING ASSISTANT
Shopping Agent
============================================================

Purpose
-------

Main interface for the AI Shopping Assistant.

Responsibilities

✓ Accept user queries
✓ Execute Shopping Graph
✓ Display AI responses

Uses

    • Shopping Graph
    • LLM Manager
    • LangChain Tools

Future

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

    from agent.shopping_graph import ShoppingGraph

except ImportError:

    from shopping_graph import ShoppingGraph


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

        logger.info(
            "Shopping Graph Loaded Successfully"
        )

        logger.info("=" * 60)
        logger.info("Shopping Agent Ready")
        logger.info("=" * 60)
        # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        query: str
    ):
        """
        Process a user query using the Shopping Graph.
        """

        logger.info("=" * 60)
        logger.info("SHOPPING AGENT STARTED")
        logger.info("=" * 60)

        logger.info(f"User Query : {query}")

        try:

            result = self.shopping_graph.chat(query)

            logger.info(
                "Shopping Graph Executed Successfully"
            )

            return result

        except Exception as e:

            logger.error(
                f"Shopping Agent Error : {e}"
            )

            raise
        # ============================================================
    # GET PROVIDER
    # ============================================================

    def get_provider(self):
        """
        Return the currently active LLM provider.
        """

        return self.shopping_graph.llm_manager.get_provider()
    
        # ============================================================
    # PROVIDER INFORMATION
    # ============================================================

    def provider_info(self):
        """
        Display the active LLM provider.
        """

        print("\n" + "=" * 60)

        print("ACTIVE LLM")

        print("=" * 60)

        print(

            f"Provider : {self.get_provider()}"

        )

        print("=" * 60)
        # ============================================================
    # DISPLAY RESPONSE
    # ============================================================

    def display_response(
        self,
        result
    ):
        """
        Display the AI response.
        """

        print("\n" + "=" * 70)
        print("AI SHOPPING ASSISTANT")
        print("=" * 70)

        messages = result["messages"]

        for message in reversed(messages):

            if hasattr(message, "content"):

                print()

                print(message.content)

                print()

                break

        print("=" * 70)
        # ============================================================
    # ASK
    # ============================================================

    def ask(
        self,
        query: str
    ):
        """
        Complete chatbot workflow.
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
    print("WELCOME TO AI SHOPPING ASSISTANT")
    print("=" * 70)

    print("\nPowered By")

    print("• LangGraph")
    print("• LangChain")
    print("• Groq")
    print("• Gemini")
    print("• Recommendation Engine")
    print("• Comparison Engine")
    print("• Bundle Engine")
    print("• Semantic Search")
    print("• FAISS")
    print("• Conversation Memory")

    print()

    agent.provider_info()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nThank you for using AI Shopping Assistant!")

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

            print(
                "\nSorry, something went wrong."
            )

    logger.info("=" * 60)
    logger.info("SHOPPING AGENT CLOSED")
    logger.info("=" * 60)