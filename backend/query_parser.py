"""
============================================================
AI SHOPPING ASSISTANT
Query Parser
============================================================

Purpose
-------
Convert natural language into structured information.

Example

Input:
"I need a Dell laptop under ₹60,000 for programming."

Output:

{
    "intent":"recommendation",
    "category":"Laptop",
    "brand":"Dell",
    "budget":60000,
    "purpose":"Programming"
}
"""

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import re
import json
import logging

# ============================================================
# LOGGER
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================
# QUERY PARSER
# ============================================================

class QueryParser:

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing Query Parser")
        logger.info("=" * 60)

        self.load_brands()

        self.load_categories()

        self.load_purposes()

        self.load_occasions()

        logger.info("Query Parser Ready")

# ============================================================
# LOAD BRANDS
# ============================================================

    def load_brands(self):

        self.brands = [

            "Apple",
            "Samsung",
            "OnePlus",
            "Xiaomi",
            "Redmi",
            "Realme",
            "Nothing",
            "Motorola",
            "Google",

            "HP",
            "Dell",
            "Lenovo",
            "Asus",
            "Acer",
            "MSI",

            "Sony",
            "JBL",
            "Boat",
            "Noise",
            "Logitech"

        ]

# ============================================================
# LOAD PRODUCT CATEGORIES
# ============================================================

    def load_categories(self):

        self.categories = {

            "phone":[
                "phone",
                "mobile",
                "smartphone"
            ],

            "laptop":[
                "laptop",
                "notebook"
            ],

            "tablet":[
                "tablet",
                "ipad"
            ],

            "headphones":[
                "headphone",
                "earphone",
                "earbuds"
            ],

            "watch":[
                "watch",
                "smartwatch"
            ],

            "camera":[
                "camera"
            ],

            "keyboard":[
                "keyboard"
            ],

            "mouse":[
                "mouse"
            ],

            "shirt":[
                "shirt",
                "tshirt",
                "t-shirt"
            ],

            "shoes":[
                "shoe",
                "shoes",
                "sneakers",
                "running shoes"
            ]

        }

# ============================================================
# LOAD PURPOSES
# ============================================================

    def load_purposes(self):

        self.purposes = {

            "gaming":[
                "gaming",
                "games"
            ],

            "programming":[
                "programming",
                "coding",
                "developer",
                "computer engineering"
            ],

            "photography":[
                "camera",
                "photography",
                "photo"
            ],

            "office":[
                "office",
                "work",
                "business"
            ],

            "college":[
                "college",
                "student",
                "study"
            ],

            "video_editing":[
                "editing",
                "video editing"
            ],

            "gym":[
                "gym",
                "fitness",
                "workout"
            ],

            "travel":[
                "travel",
                "trip",
                "vacation"
            ]

        }

# ============================================================
# LOAD OCCASIONS
# ============================================================

    def load_occasions(self):

        self.occasions = {

            "goa":[
                "goa"
            ],

            "gym":[
                "gym"
            ],

            "college":[
                "college"
            ],

            "office":[
                "office"
            ],

            "travel":[
                "travel",
                "trip"
            ]

        }

# ============================================================
# EXTRACT BUDGET
# ============================================================

    def extract_budget(self, query):

        query = query.lower()
        query = query.replace(",", "")

        # -------------------------
        # Handle values like 40k
        # -------------------------

        match = re.search(r'(\d+(?:\.\d+)?)\s*k\b', query)

        if match:

            return int(float(match.group(1)) * 1000)

        # -------------------------
        # Handle normal numbers
        # -------------------------

        patterns = [

            r'under\s*₹?\s*(\d+)',

            r'below\s*₹?\s*(\d+)',

            r'less than\s*₹?\s*(\d+)',

            r'within\s*₹?\s*(\d+)',

            r'max\s*₹?\s*(\d+)',

            r'₹\s*(\d+)',

            r'(\d+)\s*rupees'

        ]

        for pattern in patterns:

            match = re.search(pattern, query)

            if match:

                return int(match.group(1))

        return None

# ============================================================
# EXTRACT BRAND
# ============================================================

    def extract_brand(self, query):

        query = query.lower()

        brands = []

        for brand in self.brands:

            if brand.lower() in query:

                brands.append(brand)

        if len(brands) == 0:

            return None

        if len(brands) == 1:

            return brands[0]

        return brands

# ============================================================
# EXTRACT CATEGORY
# ============================================================

    def extract_category(self, query):

        query = query.lower()

        for category, keywords in self.categories.items():

            for keyword in keywords:

                if keyword in query:

                    return category

        return None
    
    # ============================================================
# EXTRACT PURPOSE
# ============================================================

    def extract_purpose(self, query):

        query = query.lower()

        for purpose, keywords in self.purposes.items():

            for keyword in keywords:

                if keyword in query:

                    return purpose

        return None


# ============================================================
# EXTRACT OCCASION
# ============================================================

    def extract_occasion(self, query):

        query = query.lower()

        for occasion, keywords in self.occasions.items():

            for keyword in keywords:

                if keyword in query:

                    return occasion

        return None


# ============================================================
# DETECT COMPARISON
# ============================================================

    def is_comparison(self, query):

        query = query.lower()

        comparison_words = [

            "compare",
            "comparison",
            "vs",
            "versus",
            "better than",
            "difference"

        ]

        for word in comparison_words:

            if word in query:

                return True

        return False


# ============================================================
# DETECT BUNDLE REQUEST
# ============================================================

    def is_bundle_request(self, query):

        query = query.lower()

        bundle_words = [

            "complete setup",
            "complete kit",
            "everything",
            "all products",
            "what should i buy",
            "shopping list",
            "bundle",
            "full setup"

        ]

        for word in bundle_words:

            if word in query:

                return True

        return False


# ============================================================
# EXTRACT PRIORITY
# ============================================================

    def extract_priority(self, query):

        query = query.lower()

        priorities = {

            "camera":[
                "camera",
                "photography",
                "selfie"
            ],

            "battery":[
                "battery",
                "backup"
            ],

            "performance":[
                "gaming",
                "performance",
                "processor",
                "speed"
            ],

            "display":[
                "display",
                "amoled",
                "oled",
                "screen"
            ],

            "storage":[
                "storage",
                "memory"
            ],

            "ram":[
                "ram"
            ],

            "lightweight":[
                "lightweight",
                "light weight"
            ]

        }

        for priority, keywords in priorities.items():

            for keyword in keywords:

                if re.search(

                    rf"\b{re.escape(keyword)}\b",

                    query ):

                    return priority

        return None


# ============================================================
# EXTRACT PRODUCT NAMES
# ============================================================

    def extract_products(self, query):

        """
        Used mainly for product comparison.

        Example:

        Compare iPhone 16 Pro and Samsung S25 Ultra

        Returns:

        [
            "iPhone 16 Pro",
            "Samsung S25 Ultra"
        ]
        """

        query = query.strip()

        separators = [

            " vs ",

            " versus ",

            " and "

        ]

        for sep in separators:

            if sep in query.lower():

                parts = re.split(

                    sep,

                    query,

                    flags=re.IGNORECASE

                )

                products = []

                for product in parts:

                    product = product.strip()

                    product = re.sub(

                        r'^(compare|compare the)\s+',

                        '',

                        product,

                        flags=re.IGNORECASE

                    )

                    products.append(product)

                return products

        return []


# ============================================================
# DETECT USER INTENT
# ============================================================

    def detect_intent(self, query):

        query = query.lower()

        if self.is_comparison(query):

            return "comparison"

        if self.is_bundle_request(query):

            return "bundle"

        recommend_words = [

            "recommend",

            "suggest",

            "best",

            "buy",

            "need",

            "looking for",

            "find"

        ]

        for word in recommend_words:

            if word in query:

                return "recommendation"

        return "general"
    
    # ============================================================
# PARSE COMPLETE QUERY
# ============================================================

    def parse_query(self, query):

        logger.info("=" * 60)
        logger.info("Parsing User Query")
        logger.info("=" * 60)

        parsed = {

            "original_query": query,

            "intent": self.detect_intent(query),

            "category": self.extract_category(query),

            "brand": self.extract_brand(query),

            "budget": self.extract_budget(query),

            "purpose": self.extract_purpose(query),

            "occasion": self.extract_occasion(query),

            "priority": self.extract_priority(query),

            "comparison": self.is_comparison(query),

            "bundle": self.is_bundle_request(query),

            "products": self.extract_products(query)

        }

        logger.info("Query Parsed Successfully")

        return parsed


# ============================================================
# DISPLAY PARSED QUERY
# ============================================================

    def display(self, parsed):

        print("\n" + "=" * 70)

        print("PARSED QUERY")

        print("=" * 70)

        print(json.dumps(

            parsed,

            indent=4,

            ensure_ascii=False

        ))

        print("=" * 70 + "\n")


# ============================================================
# INTERACTIVE TESTING
# ============================================================

    def interactive(self):

        print("\nType 'exit' to quit.\n")

        while True:

            query = input("User : ").strip()

            if query.lower() == "exit":

                break

            if query == "":

                print("Please enter a query.\n")

                continue

            parsed = self.parse_query(query)

            self.display(parsed)


# ============================================================
# QUICK PARSER FUNCTION
# ============================================================

_parser = None


def parse_query(query):

    """
    Reusable parser function.

    Other backend modules can simply do:

    from backend.query_parser import parse_query

    parsed = parse_query(user_query)
    """

    global _parser

    if _parser is None:

        _parser = QueryParser()

    return _parser.parse_query(query)


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("QUERY PARSER STARTED")
    logger.info("=" * 60)

    parser = QueryParser()

    parser.interactive()

    logger.info("=" * 60)
    logger.info("QUERY PARSER CLOSED")
    logger.info("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()