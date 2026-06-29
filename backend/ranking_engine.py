"""
============================================================
AI SHOPPING ASSISTANT
Context-Aware Ranking Engine
============================================================

Purpose
-------
Ranks candidate products using

✓ Semantic Similarity
✓ Cross Encoder Re-ranking
✓ User Context
✓ Business Rules
✓ Product Quality

Used By

    - Recommendation Engine
    - Shopping Graph
    - FastAPI
    - Streamlit

"""

# ============================================================
# IMPORTS
# ============================================================

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# PRODUCT MODEL
# ============================================================

@dataclass
class RankedProduct:
    """
    Internal product representation used by
    the ranking engine.
    """

    # --------------------------------------------------------
    # Basic Product Information
    # --------------------------------------------------------

    product_id: str = ""

    product_name: str = ""

    brand: str = ""

    category: str = ""

    sub_category: Optional[str] = None

    description: str = ""

    # --------------------------------------------------------
    # Commercial Information
    # --------------------------------------------------------

    price: float = 0.0

    rating: float = 0.0

    popularity: float = 0.0

    # --------------------------------------------------------
    # Media
    # --------------------------------------------------------

    image_url: Optional[str] = None

    product_url: Optional[str] = None

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    features: List[str] = field(default_factory=list)

    metadata: dict = field(default_factory=dict)

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    semantic_score: float = 0.0

    rerank_score: float = 0.0

    context_score: float = 0.0

    business_score: float = 0.0

    final_score: float = 0.0

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation: str = ""

# ============================================================
# CONTEXT AWARE RANKING ENGINE
# ============================================================

class ContextAwareRankingEngine:
    """
    Context-aware ranking engine.

    Pipeline

        Candidate Products
                │
                ▼
        Semantic Similarity
                │
                ▼
        CrossEncoder Re-ranking
                │
                ▼
        Context Scoring
                │
                ▼
        Business Scoring
                │
                ▼
        Final Ranking
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("INITIALIZING CONTEXT RANKING ENGINE")
        logger.info("=" * 60)

        # ----------------------------------------------------
        # Embedding Model
        # ----------------------------------------------------

        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # ----------------------------------------------------
        # Cross Encoder
        # ----------------------------------------------------

        self.cross_encoder = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        logger.info("Embedding Model Loaded")

        logger.info("Cross Encoder Loaded")

    # ============================================================
# CONVERT PRODUCTS
# ============================================================

    def convert_products(
        self,
        products: List
    ) -> List[RankedProduct]:
        """
        Convert Semantic Search results into RankedProduct objects.

        Supports:
        - List[dict]
        - List[LangChain Document]
        - List[RankedProduct]
        """

        if not products:
            return []

        # Already converted
        if isinstance(products[0], RankedProduct):
            return products

        ranked_products = []

        for product in products:

            # ------------------------------------------------
            # LangChain Document
            # ------------------------------------------------

            if hasattr(product, "metadata"):

                metadata = product.metadata

                ranked_products.append(

                    RankedProduct(

                        product_id=str(
                            metadata.get("product_id", "")
                        ),

                        product_name=metadata.get(
                            "product_name", ""
                        ),

                        brand=metadata.get(
                            "brand", ""
                        ),

                        category=metadata.get(
                            "main_category", ""
                        ),

                        sub_category=metadata.get(
                            "sub_category"
                        ),

                        description=product.page_content,

                        price=float(
                            metadata.get("price", 0)
                        ),

                        rating=float(
                            metadata.get("rating", 0)
                        ),

                        popularity=float(
                            metadata.get("popularity", 0)
                        ),

                        image_url=metadata.get(
                            "image_url"
                        ),

                        product_url=metadata.get(
                            "product_url"
                        ),

                        features=metadata.get(
                            "features",
                            []
                        ),

                        metadata=metadata

                    )

                )

            # ------------------------------------------------
            # Dictionary
            # ------------------------------------------------

            elif isinstance(product, dict):

                ranked_products.append(

                    RankedProduct(

                        product_id=str(
                            product.get("product_id", "")
                        ),

                        product_name=product.get(
                            "product_name", ""
                        ),

                        brand=product.get(
                            "brand", ""
                        ),

                        category=product.get(
                            "main_category", ""
                        ),

                        sub_category=product.get(
                            "sub_category"
                        ),

                        description=product.get(
                            "description", ""
                        ),

                        price=float(
                            product.get("price", 0)
                        ),

                        rating=float(
                            product.get("rating", 0)
                        ),

                        popularity=float(
                            product.get("popularity", 0)
                        ),

                        image_url=product.get(
                            "image_url"
                        ),

                        product_url=product.get(
                            "product_url"
                        ),

                        features=product.get(
                            "features",
                            []
                        ),

                        metadata=product,
                        # ADD THIS
                        semantic_score=float(
                            product.get("similarity_score", 0.0)
                        )

                    )

                )

        logger.info(
            f"Converted {len(ranked_products)} products."
        )

        return ranked_products


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

    def semantic_similarity(
        self,
        query: str,
        products: List[RankedProduct]
    ) -> List[RankedProduct]:
        """
        Compute semantic similarity using MiniLM embeddings.
        """

        if not products:
            return products

        query_embedding = self.embedding_model.encode(
            query,
            normalize_embeddings=True
        )

        texts = [

            f"""
            {product.product_name}

            {product.brand}

            {product.category}

            {product.description}

            {' '.join(product.features)}
            """

            for product in products

        ]

        product_embeddings = self.embedding_model.encode(

            texts,

            normalize_embeddings=True

        )

        similarities = np.dot(
            product_embeddings,
            query_embedding
        )

        for product, score in zip(
            products,
            similarities
        ):

            product.semantic_score = float(score)

        logger.info("Semantic similarity computed.")

        return products


# ============================================================
# CROSS ENCODER RERANKING
# ============================================================

    def rerank(
        self,
        query: str,
        products: List[RankedProduct]
    ) -> List[RankedProduct]:
        """
        Refine ranking using CrossEncoder.
        """

        if not products:
            return products

        pairs = [

            (

                query,

                f"""
                {product.product_name}

                {product.description}

                {' '.join(product.features)}
                """

            )

            for product in products

        ]

        scores = self.cross_encoder.predict(
            pairs
        )

        for product, score in zip(
            products,
            scores
        ):

            product.rerank_score = float(score)

        logger.info("CrossEncoder reranking completed.")

        return products


# ============================================================
# NORMALIZE RERANK SCORES
# ============================================================

    def normalize_rerank_scores(
        self,
        products: List[RankedProduct]
    ):
        """
        Normalize rerank scores to 0–1.
        """

        if not products:
            return

        scores = [

            product.rerank_score

            for product in products

        ]

        minimum = min(scores)

        maximum = max(scores)

        if maximum == minimum:

            for product in products:

                product.rerank_score = 1.0

            return

        for product in products:

            product.rerank_score = (

                product.rerank_score - minimum

            ) / (

                maximum - minimum

            )

        logger.info("Rerank scores normalized.")

    # ============================================================
# CONTEXT SCORING
# ============================================================

    def compute_context_scores(
        self,
        products: List[RankedProduct],
        shopping_context: dict
    ) -> List[RankedProduct]:
        """
        Compute context relevance score based on
        the user's shopping intent.
        """

        logger.info("=" * 60)
        logger.info("COMPUTING CONTEXT SCORES")
        logger.info("=" * 60)

        purpose = str(
            shopping_context.get("purpose", "")
        ).lower()

        category = str(
            shopping_context.get("category", "")
        ).lower()

        brand = str(
            shopping_context.get("brand", "")
        ).lower()

        color = str(
            shopping_context.get("color", "")
        ).lower()

        occasion = str(
            shopping_context.get("occasion", "")
        ).lower()

        required_features = [

            feature.lower()

            for feature in shopping_context.get(
                "required_features",
                []
            )

        ]

        constraints = [

            constraint.lower()

            for constraint in shopping_context.get(
                "constraints",
                []
            )

        ]

        for product in products:

            score = 0.0

            # ------------------------------------------------
            # Category Match
            # ------------------------------------------------

            if category:

                if category in product.category.lower():

                    score += 0.25

                elif product.sub_category and \
                        category in product.sub_category.lower():

                    score += 0.20

            # ------------------------------------------------
            # Brand Match
            # ------------------------------------------------

            if brand:

                if isinstance(brand, str):
                    brand = [brand]

                brand = [b.lower() for b in brand]

                if product.brand.lower() in brand:
                    score += 1  

            # ------------------------------------------------
            # Purpose Match
            # ------------------------------------------------

            text = (

                product.description +

                " " +

                " ".join(product.features)

            ).lower()

            if purpose:

                if purpose in text:

                    score += 0.20

            # ------------------------------------------------
            # Occasion Match
            # ------------------------------------------------

            if occasion:

                if occasion in text:

                    score += 0.10

            # ------------------------------------------------
            # Feature Match
            # ------------------------------------------------

            feature_matches = 0

            for feature in required_features:

                if feature in text:

                    feature_matches += 1

            if required_features:

                score += (

                    feature_matches /

                    len(required_features)

                ) * 0.20

            # ------------------------------------------------
            # Constraint Match
            # ------------------------------------------------

            constraint_matches = 0

            for constraint in constraints:

                if constraint in text:

                    constraint_matches += 1

            if constraints:

                score += (

                    constraint_matches /

                    len(constraints)

                ) * 0.10

            # ------------------------------------------------
            # Color Match
            # ------------------------------------------------

            if color:

                if color in text:

                    score += 0.05

            product.context_score = min(score, 1.0)

        logger.info(
            "Context scoring completed."
        )

        return products


# ============================================================
# BUSINESS SCORING
# ============================================================

    def compute_business_scores(
        self,
        products: List[RankedProduct],
        shopping_context: dict
    ) -> List[RankedProduct]:
        """
        Score products using business rules.
        """

        logger.info("=" * 60)
        logger.info("BUSINESS SCORING")
        logger.info("=" * 60)

        budget = shopping_context.get("budget")

        for product in products:

            score = 0.0

            # --------------------------------------------
            # Budget
            # --------------------------------------------

            if budget:

                if product.price <= budget:

                    score += 0.40

                else:

                    difference = product.price - budget

                    penalty = min(
                        difference / budget,
                        1.0
                    )

                    score += max(
                        0.0,
                        0.40 - penalty
                    )

            else:

                score += 0.40

            # --------------------------------------------
            # Rating
            # --------------------------------------------

            score += (

                product.rating / 5

            ) * 0.30

            # --------------------------------------------
            # Popularity
            # --------------------------------------------

            popularity = min(

                product.popularity,

                100

            )

            score += (

                popularity / 100

            ) * 0.30

            product.business_score = min(
                score,
                1.0
            )

        logger.info(
            "Business scoring completed."
        )

        return products
    # ============================================================
# FINAL SCORE
# ============================================================

    def calculate_final_scores(
        self,
        products: List[RankedProduct]
    ) -> List[RankedProduct]:
        """
        Combine all scores into a single ranking score.
        """

        logger.info("=" * 60)
        logger.info("CALCULATING FINAL SCORES")
        logger.info("=" * 60)

        for product in products:

            product.final_score = (

            product.semantic_score * 0.20 +

            product.rerank_score * 0.40 +

            product.context_score * 0.25 +

            product.business_score * 0.15

        )

        return products


# ============================================================
# SORT PRODUCTS
# ============================================================

    def sort_products(
        self,
        products: List[RankedProduct]
    ) -> List[RankedProduct]:

        return sorted(

            products,

            key=lambda x: x.final_score,

            reverse=True

        )


# ============================================================
# TOP K
# ============================================================

    def top_k(
        self,
        products: List[RankedProduct],
        k: int
    ) -> List[RankedProduct]:

        return products[:k]


# ============================================================
# EXPLANATION
# ============================================================

    def generate_explanation(
        self,
        product: RankedProduct,
        shopping_context: dict
    ) -> str:
        """
        Generate a human-readable explanation
        for why a product was recommended.
        """

        reasons = []

        budget = shopping_context.get("budget")
        brand = shopping_context.get("brand")
        purpose = shopping_context.get("purpose")
        features = shopping_context.get(
            "required_features",
            []
        )

        if budget and product.price <= budget:

            reasons.append(
                "Fits your budget"
            )

        if brand:

            if isinstance(brand, str):
                brand = [brand]

            brand = [b.lower() for b in brand]

            if product.brand.lower() in brand:

                reasons.append(
                    f"Matches preferred brand ({product.brand})"
                )

        if product.rating >= 4.5:

            reasons.append(
                "Highly rated by customers"
            )

        if purpose:

            reasons.append(
                f"Suitable for {purpose}"
            )

        text = (

            product.description +

            " " +

            " ".join(product.features)

        ).lower()

        for feature in features:

            if feature.lower() in text:

                reasons.append(
                    f"Includes {feature}"
                )

        if len(reasons) == 0:

            reasons.append(
                "Semantically relevant to your query"
            )

        return " • ".join(reasons)


# ============================================================
# ADD EXPLANATIONS
# ============================================================

    def add_explanations(
        self,
        products: List[RankedProduct],
        shopping_context: dict
    ) -> List[RankedProduct]:

        for product in products:

            product.explanation = self.generate_explanation(

                product,

                shopping_context

            )

        return products


# ============================================================
# DEBUG
# ============================================================

    def debug_scores(
        self,
        products: List[RankedProduct]
    ):

        logger.info("=" * 60)
        logger.info("RANKING RESULTS")
        logger.info("=" * 60)

        for i, product in enumerate(products, start=1):

            logger.info(

                f"""
Rank {i}

Product : {product.product_name}

Semantic : {product.semantic_score:.3f}

Rerank : {product.rerank_score:.3f}

Context : {product.context_score:.3f}

Business : {product.business_score:.3f}

Final : {product.final_score:.3f}

Explanation : {product.explanation}
"""

            )


# ============================================================
# MAIN RANKING PIPELINE
# ============================================================

    def rank_products(
        self,
        query: str,
        products,
        shopping_context: dict,
        top_k: int = 5
    ) -> List[RankedProduct]:
        """
        Complete context-aware ranking pipeline.
        """

        logger.info("=" * 60)
        logger.info("STARTING CONTEXT-AWARE RANKING")
        logger.info("=" * 60)

        # Convert search results
        products = self.convert_products(products)

        if not products:

            return []

        # # Semantic Similarity
        # products = self.semantic_similarity(

        #     query,

        #     products

        # )

        # Cross Encoder Re-ranking
        products = self.rerank(

            query,

            products

        )

        # Normalize Scores
        self.normalize_rerank_scores(

            products

        )

        # Context Score
        products = self.compute_context_scores(

            products,

            shopping_context

        )

        # Business Score
        products = self.compute_business_scores(

            products,

            shopping_context

        )

        # Final Score
        products = self.calculate_final_scores(

            products

        )

        # Sort
        products = self.sort_products(

            products

        )

        # Explanations
        products = self.add_explanations(

            products,

            shopping_context

        )

        # Debug
        self.debug_scores(products)

        logger.info("=" * 60)
        logger.info("RANKING COMPLETED")
        logger.info("=" * 60)

        return self.top_k(

            products,

            top_k

        )