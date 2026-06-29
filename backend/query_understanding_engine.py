"""
============================================================
AI SHOPPING ASSISTANT
Query Understanding Engine
============================================================

Purpose
-------
Uses an LLM with structured output to understand
the user's shopping request.

Responsibilities

✓ Detect Intent
✓ Extract Category
✓ Extract Budget
✓ Extract Brand
✓ Extract Purpose
✓ Extract Occasion
✓ Extract Color
✓ Extract Product Names
✓ Extract Constraints
✓ Return Structured JSON

Used By

    - Shopping Graph
    - Recommendation Engine
    - Comparison Engine
    - Bundle Engine

No hardcoded rules.
No regex.
LLM-driven structured understanding.
"""
 # ============================================================
# IMPORTS
# ============================================================

import logging


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)
# ============================================================
# IMPORTS
# ============================================================

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

# ============================================================
# QUERY UNDERSTANDING SCHEMA
# ============================================================

class QueryUnderstanding(BaseModel):
    """
    Structured understanding of a user's shopping query.
    """

    # --------------------------------------------------------
    # Core Information
    # --------------------------------------------------------



    intent: Literal[
        "recommendation",
        "comparison",
        "bundle",
        "product_information",
        "greeting"
    ] = Field(
        description="Detected shopping intent."
    )

    category: Optional[str] = Field(
        default=None,
        description="Main product category."
    )

    sub_category: Optional[str] = Field(
        default=None,
        description="Specific product type."
    )

    # --------------------------------------------------------
    # Product Information
    # --------------------------------------------------------

    product_names: Optional[List[str]] = Field(
        default_factory=list
    )

    brand: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Preferred brand or brands."
    )

    color: Optional[str] = Field(
        default=None,
        description="Preferred color."
    )

    # --------------------------------------------------------
    # Budget
    # --------------------------------------------------------

    budget: Optional[float] = Field(
        default=None,
        description="Maximum budget."
    )

    currency: Optional[str] = Field(
        default="INR",
        description="Currency."
    )

    # --------------------------------------------------------
    # Shopping Purpose
    # --------------------------------------------------------

    purpose: Optional[str] = Field(
        default=None,
        description="""
        Why the customer needs this product.

        Examples:
        photography
        gaming
        programming
        college
        travel
        fitness
        office
        """
    )

    occasion: Optional[str] = Field(
        default=None,
        description="""
        Occasion if mentioned.

        Example:
        Goa trip
        Wedding
        Gym
        Vacation
        """
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    required_features: Optional[List[str]] = Field(
            default_factory=list
        )
    
    constraints: Optional[List[str]] = Field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Bundle Information
    # --------------------------------------------------------

    bundle_type: Optional[str] = Field(
        default=None,
        description="Bundle type."
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1."
    )
# ============================================================
# QUERY UNDERSTANDING ENGINE
# ============================================================

class QueryUnderstandingEngine:
    """
    AI-powered Query Understanding Engine.

    Uses an LLM with structured output to convert
    natural language into structured shopping context.
    """

    def __init__(self, llm: BaseChatModel):

        self.llm = llm

        # ----------------------------------------------------
        # Prompt
        # ----------------------------------------------------

        self.prompt = ChatPromptTemplate.from_messages(

            [

                (

                    "system",

                    """
You are an expert AI Shopping Assistant.

Your job is NOT to recommend products.
IMPORTANT

Never return null for list fields.

Always return [] for

- product_names
- required_features
- constraints

Never return null.

Example

product_names: []

required_features: []

constraints: []

Your ONLY job is to understand the user's shopping query.

Extract every important shopping detail.

Return ONLY structured data.

Instructions:

- Identify the user's intent.

- Extract category.

- Extract sub-category.

- Extract budget.

- Extract preferred brand.

- Extract preferred color.

- Extract purpose.

- Extract occasion.

- Extract required features.

- Extract shopping constraints.

- Extract all mentioned product names.

- Identify bundle type if applicable.

- Estimate confidence between 0 and 1.

Never recommend products.

Never explain anything.

Return structured output only.

If any information is missing,
leave it as null.

Do not guess brands.

Do not invent products.

Infer technical requirements only
when they logically follow from the
user's purpose.

If the user's purpose implies important product
requirements, infer them.

Examples:

Programming
→ 16GB RAM
→ SSD

Gaming
→ Dedicated GPU
→ High Refresh Rate

Photography
→ OIS
→ High Camera Quality

Gym
→ Breathable Fabric
→ Lightweight

Travel
→ Lightweight
→ Comfortable

College
→ Good Battery Life
→ Portable

"""

                ),

                (

                    "human",

                    "{query}"

                )

            ]

        )

        # ----------------------------------------------------
        # Structured Output
        # ----------------------------------------------------

        self.chain = (

            self.prompt

            |

            self.llm.with_structured_output(

                QueryUnderstanding

            )

        )

   


# ============================================================
# UNDERSTAND QUERY
# ============================================================

    def understand(
        self,
        query: str
    ) -> dict:
        """
        Understand the user's shopping query.

        Returns
        -------
        dict
            Structured shopping information.
        """

        logger.info("=" * 60)
        logger.info("QUERY UNDERSTANDING ENGINE")
        logger.info("=" * 60)

        logger.info(f"User Query : {query}")

        try:

            result = self.chain.invoke(

                {

                    "query": query

                }

            )

            logger.info("Query Successfully Understood")

            logger.info(result)

            return result.model_dump()

        except Exception as e:

            logger.error(

                f"Query Understanding Error : {e}"

            )

            raise


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    from agent.llm_manager import LLMManager

    llm = LLMManager().get_llm()

    engine = QueryUnderstandingEngine(llm)

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nQuery Understanding Engine Closed.")

            break

        if not query:

            continue

        result = engine.understand(query)

        print("\n" + "=" * 70)

        print("UNDERSTOOD QUERY")

        print("=" * 70)

        for key, value in result.items():

            print(f"{key:20}: {value}")

        print("=" * 70)