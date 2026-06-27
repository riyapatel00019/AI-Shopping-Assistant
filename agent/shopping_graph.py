"""
============================================================
AI SHOPPING ASSISTANT
LangGraph Shopping Workflow
============================================================

Purpose
-------

Central workflow powered by LangGraph.

Responsibilities

✓ Maintain conversation state
✓ Invoke the LLM
✓ Execute LangChain tools
✓ Route tool calls
✓ Return final response

Uses

    • LangGraph
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
import time
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, END

from langgraph.graph.message import add_messages

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage
)

from langgraph.prebuilt import ToolNode


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

try:

    from agent.llm_manager import LLMManager

    from agent.tools import get_tools

except ImportError:

    from llm_manager import LLMManager

    from tools import get_tools


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# ============================================================
# GRAPH STATE
# ============================================================

class ShoppingState(TypedDict):
    """
    LangGraph state.
    """

    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ============================================================
# SHOPPING GRAPH
# ============================================================

class ShoppingGraph:

    """
    AI Shopping Workflow using LangGraph.
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Shopping Graph")
        logger.info("=" * 60)

        # --------------------------------------------
        # LLM Manager
        # --------------------------------------------

        self.llm_manager = LLMManager()

        self.llm = self.llm_manager.get_llm()

        logger.info(
            f"Using Provider : {self.llm_manager.get_provider()}"
        )

        # --------------------------------------------
        # LangChain Tools
        # --------------------------------------------

        self.tools = get_tools()

        logger.info(
            f"Loaded {len(self.tools)} Tools"
        )

        # --------------------------------------------
        # Bind Tools to LLM
        # --------------------------------------------

        self.llm_with_tools = self.llm.bind_tools(
            self.tools
        )
        # Compile graph once
        self.graph = self.build_graph()

        logger.info(
            "LLM Successfully Bound With Tools"
        )
        # ============================================================
    # CHATBOT NODE
    # ============================================================

    def chatbot_node(
        self,
        state: ShoppingState
    ):
        """
        Main LLM node.

        The LLM receives the conversation and
        decides whether to answer directly
        or call a tool.
        """

        logger.info("=" * 60)
        logger.info("CHATBOT NODE")
        logger.info("=" * 60)

        # Keep only the most recent messages
        # to avoid large prompts and repeated context.
        messages = state["messages"][-3:]

    

        # Safety check to prevent endless tool-call loops
        tool_message_count = sum(
            1 for msg in messages
            if msg.__class__.__name__ == "ToolMessage"
        )

        if tool_message_count >= 1:
            logger.warning("Too many tool calls detected. Stopping further tool execution.")

            return {
                "messages": [
                    AIMessage(
                        content="I have gathered the required information and cannot perform additional tool calls for this request."
                    )
                ]
            }

        logger.info(
            f"Messages Received : {len(messages)}"
        )


        logger.info("=" * 60)

        for i, msg in enumerate(messages):

            logger.info(f"Message {i+1}: {type(msg).__name__}")

            if hasattr(msg, "content"):

                logger.info(f"Length : {len(str(msg.content))}")

                # Print only the first 100 characters
                # logger.info(str(msg.content)[:100])

        logger.info("=" * 60)
        response = self.llm_with_tools.invoke(
            messages
        )

        logger.info("LLM Response Generated")

        return {

            "messages": [response]

        }
        # ============================================================
    # ROUTER
    # ============================================================

    def should_continue(
        self,
        state: ShoppingState
    ):
        """
        Decide whether to continue calling tools.
        """

        logger.info("=" * 60)
        logger.info("ROUTER")
        logger.info("=" * 60)

        messages = state["messages"]

        # -----------------------------
        # Prevent infinite tool loops
        # -----------------------------
        tool_count = sum(
            1
            for message in messages
            if message.__class__.__name__ == "ToolMessage"
        )

        logger.info(f"Tool Calls So Far : {tool_count}")

        if tool_count >= 1:

            logger.info("Maximum Tool Calls Reached")

            return END

        last_message = messages[-1]

        if hasattr(last_message, "tool_calls"):

            if last_message.tool_calls:

                logger.info("Tool Call Detected")

                return "tools"

        logger.info("Returning Final Response")

        return END
    
        # ============================================================
    # TOOL NODE
    # ============================================================

    def build_tool_node(self):
        """
        Create LangGraph ToolNode.
        """

        logger.info("=" * 60)
        logger.info("Creating Tool Node")
        logger.info("=" * 60)

        return ToolNode(self.tools)
        # ============================================================
    # BUILD GRAPH
    # ============================================================

    def build_graph(self):
        """
        Build the LangGraph workflow.
        """

        logger.info("=" * 60)
        logger.info("BUILDING LANGGRAPH")
        logger.info("=" * 60)

        # --------------------------------------------
        # Create Graph
        # --------------------------------------------

        graph = StateGraph(ShoppingState)

        # --------------------------------------------
        # Create Tool Node
        # --------------------------------------------

        tool_node = self.build_tool_node()

        # --------------------------------------------
        # Add Nodes
        # --------------------------------------------

        graph.add_node(

            "chatbot",

            self.chatbot_node

        )

        graph.add_node(

            "tools",

            tool_node

        )

        logger.info("Nodes Added Successfully")
            # --------------------------------------------
        # Entry Point
        # --------------------------------------------

        graph.set_entry_point(

            "chatbot"

        )

        logger.info("Entry Point Set")
            # --------------------------------------------
        # Conditional Routing
        # --------------------------------------------

        graph.add_conditional_edges(

            "chatbot",

            self.should_continue

        )

        logger.info("Conditional Edge Added")
            # --------------------------------------------
        # Tool → Chatbot
        # --------------------------------------------

        graph.add_edge(

            "tools",

            "chatbot"

        )

        logger.info("Tool Edge Added")
            # --------------------------------------------
        # Compile Graph
        # --------------------------------------------

        self.graph = graph.compile()

        logger.info("=" * 60)
        logger.info("LANGGRAPH COMPILED SUCCESSFULLY")
        logger.info("=" * 60)

        return self.graph
    
        # ============================================================
    # GET GRAPH
    # ============================================================

    def get_graph(self):
        """
        Return the compiled LangGraph workflow.
        """
        return self.graph
    
        # ============================================================
    # CHAT
    # ============================================================

    def chat(
        self,
        query: str
    ):
        """
        Execute the LangGraph workflow.
        """

        logger.info("=" * 60)
        logger.info("SHOPPING GRAPH STARTED")
        logger.info("=" * 60)
        start = time.time()
        graph = self.get_graph()

        state = {

            "messages": [

                HumanMessage(content=query)

            ]

        }

        result = graph.invoke(state)

        end = time.time()

        logger.info(
            f"Execution Time : {end-start:.2f} sec"
        )

        logger.info("Graph Execution Completed")

        return result
        # ============================================================
    # DISPLAY RESPONSE
    # ============================================================

    def display_response(
        self,
        result
    ):
        """
        Display final AI response.
        """

        print("\n" + "=" * 70)
        print("AI SHOPPING ASSISTANT")
        print("=" * 70)

        messages = result["messages"]

        for message in reversed(messages):

            if isinstance(message, AIMessage):

                print(message.content)

                break

        print("=" * 70)
    # ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("SHOPPING GRAPH TEST")
    logger.info("=" * 60)

    shopping_graph = ShoppingGraph()

    # shopping_graph.build_graph()

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

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nShopping Graph Closed.")

            break

        if query == "":

            continue

        try:

            result = shopping_graph.chat(query)

            shopping_graph.display_response(result)

        except Exception as e:

            logger.error(f"Graph Error : {e}")