"""
============================================================
AI SHOPPING ASSISTANT
Shopping Graph
============================================================

Responsibilities
----------------
✓ Conversation Memory
✓ Query Understanding
✓ Follow-up Questions
✓ Intent Detection

This graph DOES NOT perform recommendations.

RecommendationEngine,
ComparisonEngine and BundleEngine
are called later by ShoppingAgent.

This graph is responsible only for
understanding the user's shopping request.
"""

# ============================================================
# IMPORTS
# ============================================================

import logging

from typing import TypedDict
from typing import List
from typing import Optional

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)

# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from agent.llm_manager import LLMManager

from backend.query_understanding_engine import QueryUnderstandingEngine


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# ============================================================
# SHOPPING STATE
# ============================================================

class ShoppingState(TypedDict):

    messages: List[BaseMessage]

    query: str

    shopping_context: dict

    intent: Optional[str]

    follow_up_required: bool

    follow_up_question: Optional[str]


# ============================================================
# SHOPPING GRAPH
# ============================================================

class ShoppingGraph:

    """
    Handles

    ✓ Query Understanding

    ✓ Conversation Memory

    ✓ Follow-up Questions

    ✓ Intent Detection
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Shopping Graph")
        logger.info("=" * 60)

        # ----------------------------------------
        # LLM
        # ----------------------------------------

        self.llm_manager = LLMManager()

        # Get the active LLM
        llm = self.llm_manager.get_llm()

        # ----------------------------------------
        # Query Understanding
        # ----------------------------------------

        self.query_engine = QueryUnderstandingEngine(llm)

        # ----------------------------------------
        # Conversation Memory
        # ----------------------------------------

        self.messages = []

        self.shopping_context = {}


        logger.info("Shopping Graph Ready")
    
    # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        query: str
    ):
        """
        Process a shopping query.

        This graph is responsible only for:

        ✓ Query understanding
        ✓ Intent detection
        ✓ Conversation memory
        ✓ Follow-up questions

        It DOES NOT perform recommendations.
        """

        logger.info("=" * 60)
        logger.info("SHOPPING GRAPH STARTED")
        logger.info("=" * 60)

        logger.info(f"User Query : {query}")

        # --------------------------------------------------------
        # Store User Message
        # --------------------------------------------------------

        MAX_HISTORY = 20

        self.messages.append(
            HumanMessage(content=query)
        )

        if len(self.messages) > MAX_HISTORY:

            self.messages = self.messages[-MAX_HISTORY:]

        # --------------------------------------------------------
        # Understand Query
        # --------------------------------------------------------

        shopping_context = self.query_engine.understand(query)

        # --------------------------------------------------------
        # Merge Previous Shopping Context
        # --------------------------------------------------------

        for key, value in self.shopping_context.items():

            if shopping_context.get(key) in (None, "", []):

                shopping_context[key] = value

        # --------------------------------------------------------
        # Normalize Category
        # --------------------------------------------------------

        category = shopping_context.get("category")

        if category:

            shopping_context["category"] = (
                category
                .lower()
                .rstrip("s")
            )

        # Save normalized shopping context
        self.shopping_context = shopping_context

        logger.info("Shopping Context Generated")

        intent = shopping_context.get(
            "intent",
            "recommendation"
        )

        logger.info(f"Intent : {intent}")

        # --------------------------------------------------------
        # Missing Information Detection
        # --------------------------------------------------------

        follow_up_required = False

        follow_up_question = None

        missing_fields = []

        # --------------------------------------------------------
        # Recommendation
        # --------------------------------------------------------

        if intent == "recommendation":

            if shopping_context.get("category") is None:

                missing_fields.append("product category")

            if shopping_context.get("budget") is None:

                missing_fields.append("budget")

        # --------------------------------------------------------
        # Comparison
        # --------------------------------------------------------

        elif intent == "comparison":

            if len(
                shopping_context.get(
                    "product_names",
                    []
                )
            ) < 2:

                missing_fields.append("second product")

        # --------------------------------------------------------
        # Bundle
        # --------------------------------------------------------

        elif intent == "bundle":

            if shopping_context.get("bundle_type") is None:

                missing_fields.append("bundle type")

        # --------------------------------------------------------
        # Generate Follow-up Question
        # --------------------------------------------------------

        if missing_fields:

            follow_up_required = True

            prompt = f"""
            You are an experienced e-commerce sales associate.

            Shopping Context:

            {shopping_context}

            Detected Intent:

            {intent}

            Missing Information:

            {', '.join(missing_fields)}

            Ask ONE friendly follow-up question.

            Don't ask for information that is already known.

            Return ONLY the question.
            """

            follow_up_question = self.llm_manager.invoke(
                prompt
            ).content.strip()

            self.messages.append(

                AIMessage(
                    content=follow_up_question
                )

            )

        logger.info("=" * 60)
        logger.info("SHOPPING GRAPH COMPLETED")
        logger.info("=" * 60)

        return {

            "messages": self.messages,

            "query": query,

            "shopping_context": shopping_context,

            "intent": intent,

            "follow_up_required": follow_up_required,

            "follow_up_question": follow_up_question,

            "conversation_complete": not follow_up_required

        }
    # ============================================================
    # ADD USER MESSAGE
    # ============================================================

    def add_user_message(
        self,
        message: str
    ):
        """
        Store a user message in conversation memory.
        """

        self.messages.append(
            HumanMessage(
                content=message
            )
        )


    # ============================================================
    # ADD AI MESSAGE
    # ============================================================

    def add_ai_message(
        self,
        message: str
    ):
        """
        Store an AI response in conversation memory.
        """

        self.messages.append(
            AIMessage(
                content=message
            )
        )


    # ============================================================
    # GET CHAT HISTORY
    # ============================================================

    def get_chat_history(self):
        """
        Return conversation history.
        """

        return self.messages


    # ============================================================
    # GET LAST USER MESSAGE
    # ============================================================

    def get_last_user_message(self):

        """
        Return the latest user message.
        """

        for message in reversed(self.messages):

            if isinstance(message, HumanMessage):

                return message.content

        return None


    # ============================================================
    # GET LAST AI MESSAGE
    # ============================================================

    def get_last_ai_message(self):

        """
        Return the latest assistant message.
        """

        for message in reversed(self.messages):

            if isinstance(message, AIMessage):

                return message.content

        return None


    # ============================================================
    # CLEAR MEMORY
    # ============================================================

    def clear_memory(self):

        """
        Reset the conversation.
        """

        logger.info("=" * 60)
        logger.info("Conversation Memory Cleared")
        logger.info("=" * 60)

        self.messages = []
        self.shopping_context = {}


    # ============================================================
    # MEMORY SIZE
    # ============================================================

    def conversation_length(self):

        """
        Number of stored messages.
        """

        return len(self.messages)


    # ============================================================
    # DISPLAY MEMORY
    # ============================================================

    def display_memory(self):

        """
        Print conversation history.
        """

        print("\n" + "=" * 70)
        print("CONVERSATION MEMORY")
        print("=" * 70)

        for message in self.messages:

            role = "User"

            if isinstance(message, AIMessage):

                role = "Assistant"

            print(f"\n{role}")

            print("-" * 40)

            print(message.content)

        print("=" * 70)

        # ============================================================
    # GRAPH INFORMATION
    # ============================================================

    def graph_info(self):
        """
        Display graph information.
        """

        print("\n" + "=" * 70)
        print("SHOPPING GRAPH INFORMATION")
        print("=" * 70)

        print(f"Conversation Messages : {len(self.messages)}")

        print(
            f"LLM Provider : {self.llm_manager.get_provider()}"
        )

        print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("SHOPPING GRAPH TEST STARTED")
    logger.info("=" * 60)

    graph = ShoppingGraph()

    print("\n" + "=" * 70)
    print("AI SHOPPING ASSISTANT")
    print("Shopping Graph Testing")
    print("=" * 70)

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nShopping Graph Closed.")

            break

        if not query:

            print("Please enter a query.\n")

            continue

        try:

            result = graph.chat(query)

            print("\n" + "=" * 70)
            print("SHOPPING CONTEXT")
            print("=" * 70)

            context = result["shopping_context"]

            for key, value in context.items():

                print(f"{key:20}: {value}")

            print("=" * 70)

            if result["follow_up_required"]:

                print("\nFollow-up Question:\n")

                print(result["follow_up_question"])

            else:

                print("\nNo follow-up question required.")

            print("=" * 70)

        except Exception as e:

            logger.error(f"Shopping Graph Error : {e}")

            print("\nError:", e)

    logger.info("=" * 60)
    logger.info("SHOPPING GRAPH TEST COMPLETED")
    logger.info("=" * 60)