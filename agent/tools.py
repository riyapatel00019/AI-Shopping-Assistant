"""
============================================================
AI SHOPPING ASSISTANT
LangChain Tools
============================================================

Purpose
-------

Wrap existing backend engines as LangChain Tools.

Uses

    • Recommendation Engine
    • Comparison Engine
    • Bundle Engine

This file DOES NOT contain business logic.

It only exposes backend functionality
to LangChain / LangGraph.

"""

# ============================================================
# IMPORTS
# ============================================================

import logging

# ============================================================
# IMPORT BACKEND ENGINES
# ============================================================
import os
import sys
import logging

from langchain_core.tools import tool

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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


# ============================================================
# LOAD ENGINES
# ============================================================

logging.info("=" * 60)
logging.info("Loading LangChain Tools")
logging.info("=" * 60)

recommendation_engine = RecommendationEngine()

comparison_engine = ComparisonEngine()

bundle_engine = BundleEngine()

logging.info("All Engines Loaded Successfully")


# ============================================================
# RECOMMENDATION TOOL
# ============================================================

@tool
def recommendation_tool(query: str):
    """
    Recommend products based on the user's query.
    """

    products = recommendation_engine.recommend(query)

    if not products:
        return "No products found."

    result = []

    for i, p in enumerate(products[:5], start=1):

        result.append(
            f"""
{i}. {p['product_name']}
Brand : {p['brand']}
Price : ₹{p['price']}
Rating : {p['rating']}
"""
        )

    return "\n".join(result)


# ============================================================
# COMPARISON TOOL
# ============================================================

@tool
def comparison_tool(query: str):
    """
    Compare multiple products.
    """

    comparison = comparison_engine.compare(query)

    if comparison is None:
        return "Unable to compare products."

    winner = comparison.get("winner")

    if winner is None:
        return "No winner found."

    return f"""
Best Product : {winner['product_name']}
Brand : {winner['brand']}
Price : ₹{winner['price']}
Rating : {winner['rating']}
Comparison Score : {winner['comparison_score']}
"""


# ============================================================
# BUNDLE TOOL
# ============================================================

@tool
def bundle_tool(query: str):
    """
    Generate product bundle.
    """

    bundle = bundle_engine.create_bundle(query)

    if bundle is None:
        return "No bundle found."

    main = bundle["main_product"]

    text = f"""
Main Product:
{main['product_name']}

Brand : {main['brand']}
Price : ₹{main['price']}

Recommended Bundle:
"""

    for p in bundle["bundle_products"][:3]:

        text += f"""

• {p['product_name']}
  ₹{p['price']}
"""

    return text
# ============================================================
# ALL LANGCHAIN TOOLS
# ============================================================

TOOLS = [

    recommendation_tool,

    comparison_tool,

    bundle_tool

]


# ============================================================
# GET TOOL NAMES
# ============================================================

def get_tool_names():

    """
    Return all available tool names.
    """

    return [

        tool.name

        for tool in TOOLS

    ]


# ============================================================
# GET ALL TOOLS
# ============================================================

def get_tools():

    """
    Return LangChain tools.
    """

    logging.info("=" * 60)

    logging.info("Returning LangChain Tools")

    logging.info("=" * 60)

    return TOOLS


# ============================================================
# TOOL SUMMARY
# ============================================================

def display_tools():

    """
    Display all available tools.
    """

    print("\n" + "=" * 70)

    print("AVAILABLE LANGCHAIN TOOLS")

    print("=" * 70)

    for index, tool in enumerate(TOOLS, start=1):

        print(f"{index}. {tool.name}")

        print(f"   Description : {tool.description}")

        print("-" * 70)


# ============================================================
# FIND TOOL
# ============================================================

def find_tool(tool_name):

    """
    Return tool by name.
    """

    for tool in TOOLS:

        if tool.name == tool_name:

            return tool

    return None


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(

    tool_name,

    query

):

    """
    Execute a LangChain tool.
    """

    tool = find_tool(tool_name)

    if tool is None:

        raise Exception(

            f"Tool '{tool_name}' not found."

        )

    logging.info("=" * 60)

    logging.info(

        f"Executing Tool : {tool_name}"

    )

    logging.info("=" * 60)

    return tool.invoke(query)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.info("=" * 60)
    logging.info("LANGCHAIN TOOLS TEST STARTED")
    logging.info("=" * 60)

    display_tools()

    print("\nType 'exit' to quit.\n")

    while True:

        print("\nAvailable Tools")

        for index, tool in enumerate(TOOLS, start=1):

            print(f"{index}. {tool.name}")

        tool_name = input("\nTool Name : ").strip()

        if tool_name.lower() == "exit":

            print("\nLangChain Tools Closed.")

            break

        query = input("User Query : ").strip()

        if not query:

            print("Please enter a query.")

            continue

        try:

            result = execute_tool(

                tool_name,

                query

            )

            print("\n" + "=" * 70)
            print("TOOL RESULT")
            print("=" * 70)

            print(result)

            print("=" * 70)

        except Exception as e:

            logging.error(

                f"Tool Execution Error : {e}"

            )