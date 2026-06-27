"""
============================================================
Conversation Memory
============================================================

Stores user conversation context for contextual shopping.

Used By:
    - Query Parser
    - Recommendation Engine
    - Comparison Engine
    - Bundle Engine
    - Chatbot
    - LangGraph
    - FastAPI


"""

import copy
import json
import logging
from datetime import datetime


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)


# ============================================================
# CONVERSATION MEMORY
# ============================================================

class ConversationMemory:

    """
    Stores user context across multiple conversations.
    """

    def __init__(self):

        logging.info("=" * 60)
        logging.info("Initializing Conversation Memory")
        logging.info("=" * 60)

        self.sessions = {}

        logging.info("Conversation Memory Ready")


    # ========================================================
    # DEFAULT MEMORY STRUCTURE
    # ========================================================

    def default_memory(self):

        """
        Creates an empty memory dictionary.
        """

        return {

            "user_id": None,

            "category": None,

            "brand": None,

            "budget": None,

            "purpose": None,

            "priority": None,

            "occasion": None,

            "color": None,

            "ram": None,

            "storage": None,

            "os": None,

            "processor": None,

            "comparison": False,

            "bundle": False,

            "products": [],

            "history": [],

            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        }


    # ========================================================
    # CREATE SESSION
    # ========================================================

    def create_session(self, user_id="user_1"):

        """
        Creates a new user session.
        """

        if user_id not in self.sessions:

            memory = self.default_memory()

            memory["user_id"] = user_id

            self.sessions[user_id] = memory

            logging.info(f"New Session Created : {user_id}")

        else:

            logging.info(f"Existing Session Loaded : {user_id}")

        return self.sessions[user_id]


    # ========================================================
    # GET MEMORY
    # ========================================================

    def get_memory(self, user_id="user_1"):

        """
        Returns current memory.
        """

        if user_id not in self.sessions:

            self.create_session(user_id)

        return self.sessions[user_id]


    # ========================================================
    # SHOW MEMORY
    # ========================================================

    def show_memory(self, user_id="user_1"):

        """
        Pretty print memory.
        """

        memory = self.get_memory(user_id)

        print("\n" + "=" * 70)
        print("CURRENT CONVERSATION MEMORY")
        print("=" * 70)

        print(
            json.dumps(
                memory,
                indent=4
            )
        )

        print("=" * 70 + "\n")

        # ========================================================
    # UPDATE MEMORY
    # ========================================================

    def update_memory(self, parsed_query, user_id="user_1"):

        """
        Updates memory using parsed query.
        Only non-empty values overwrite existing memory.
        """

        memory = self.get_memory(user_id)

        ignore_fields = [
            "original_query",
            "history",
            "intent",
            "created_at",
            "updated_at"
        ]

        for key, value in parsed_query.items():

            if key in ignore_fields:
                continue

            if value is None:
                continue

            if isinstance(value, str) and value.strip() == "":
                continue

            if isinstance(value, list) and len(value) == 0:
                continue

            memory[key] = value

        memory["updated_at"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        self.add_history(
            parsed_query.get("original_query", ""),
            user_id
        )

        logging.info("Conversation Memory Updated")


    # ========================================================
    # ADD HISTORY
    # ========================================================

    def add_history(self, message, user_id="user_1"):

        """
        Stores user conversation history.
        """

        if message is None:
            return

        memory = self.get_memory(user_id)

        if len(memory["history"]) == 0 or \
        memory["history"][-1]["message"] != message:

            memory["history"].append({

                "message": message,

                "time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            })
        MAX_HISTORY = 20

        if len(memory["history"]) > MAX_HISTORY:

            memory["history"].pop(0)


    # ========================================================
    # GET HISTORY
    # ========================================================

    def get_history(self, user_id="user_1"):

        """
        Returns conversation history.
        """

        memory = self.get_memory(user_id)

        return memory["history"]


    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(self, user_id="user_1"):

        """
        Returns memory without metadata.
        Used by Recommendation Engine and LLM.
        """

        memory = copy.deepcopy(
            self.get_memory(user_id)
        )

        context = {}

        for key, value in memory.items():

            if key in [

                "history",

                "created_at",

                "updated_at"

            ]:

                continue

            if value is None:

                continue

            if value == []:

                continue

            context[key] = value

        return context


    # ========================================================
    # REMOVE FIELD
    # ========================================================

    def remove_field(self, field, user_id="user_1"):

        """
        Clears one field from memory.
        """

        memory = self.get_memory(user_id)

        if field in memory:

            if isinstance(memory[field], list):

                memory[field] = []

            elif isinstance(memory[field], bool):

                memory[field] = False

            else:

                memory[field] = None

            logging.info(f"{field} removed from memory.")


    # ========================================================
    # CLEAR MEMORY
    # ========================================================

    def clear_memory(self, user_id="user_1"):

        """
        Clears entire user memory.
        """

        self.sessions[user_id] = self.default_memory()

        self.sessions[user_id]["user_id"] = user_id

        logging.info("Conversation Memory Cleared")
    
    # ============================================================
# INTERACTIVE TEST
# ============================================================

if __name__ == "__main__":

    try:
        from backend.query_parser import QueryParser
    except ImportError:
        from query_parser import QueryParser

    logging.info("=" * 60)
    logging.info("CONVERSATION MEMORY STARTED")
    logging.info("=" * 60)

    parser = QueryParser()
    memory = ConversationMemory()

    user_id = "user_1"

    memory.create_session(user_id)

    print("\nType 'exit' to quit.")
    print("Type 'clear' to clear memory.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nConversation Ended.")
            break

        if query.lower() == "clear":

            memory.clear_memory(user_id)

            print("\nMemory Cleared Successfully.\n")

            continue
        if query.lower() == "show":

            memory.show_memory(user_id)

            continue

        if query.lower() == "history":

            history = memory.get_history(user_id)

            print()

            for i, item in enumerate(history, 1):

                print(f"{i}. {item['message']}")

            print()

            continue

        # ---------------------------------------------
        # Parse Query
        # ---------------------------------------------

        parsed_query = parser.parse_query(query)

        # ---------------------------------------------
        # Update Memory
        # ---------------------------------------------

        memory.update_memory(
            parsed_query,
            user_id
        )

        # ---------------------------------------------
        # Build Context
        # ---------------------------------------------

        context = memory.build_context(user_id)

        # ---------------------------------------------
        # Display Current Context
        # ---------------------------------------------

        print("\n" + "=" * 70)
        print("CURRENT SHOPPING CONTEXT")
        print("=" * 70)

        print(
            json.dumps(
                context,
                indent=4
            )
        )

        print("=" * 70)

        # ---------------------------------------------
        # Display History
        # ---------------------------------------------

        print("\nConversation History")

        history = memory.get_history(user_id)

        for index, item in enumerate(history, start=1):

            print(f"{index}. {item['message']}")

        print()