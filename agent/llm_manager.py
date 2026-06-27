"""
============================================================
AI SHOPPING ASSISTANT
LLM Manager
============================================================

Purpose
-------

Centralized LLM Manager.

Responsibilities

✓ Load API Keys
✓ Initialize Groq
✓ Initialize Gemini
✓ Provide LLM Instances
✓ Support Future LLM Fallback

Supported Providers

    • Groq
    • Gemini

Future

    • OpenRouter
    • Ollama

"""

# ============================================================
# IMPORTS
# ============================================================

import os
import logging

from dotenv import load_dotenv

from langchain_groq import ChatGroq

from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# API KEYS
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


# ============================================================
# LLM MANAGER
# ============================================================

class LLMManager:

    """
    Centralized LLM Manager.
    """

    def __init__(self):

        logger.info("=" * 60)
        logger.info("Initializing LLM Manager")
        logger.info("=" * 60)

        self.groq_llm = None

        self.gemini_llm = None

        self.current_provider = None

        self.initialize_llms()
        logger.info(
                f"Available Providers : {self.get_available_providers()}"
            )
        # ============================================================
    # INITIALIZE LLMs
    # ============================================================

    def initialize_llms(self):

        logger.info("Loading Available LLMs")

        # ---------------------------------------------
        # Groq
        # ---------------------------------------------

        if GROQ_API_KEY:

            try:

                self.groq_llm = ChatGroq(

                    model="llama-3.3-70b-versatile",

                    temperature=0,

                    api_key=GROQ_API_KEY

                )

                logger.info("Groq Loaded Successfully")

            except Exception as e:

                logger.warning(

                    f"Groq Initialization Failed : {e}"

                )

        else:

            logger.warning("Groq API Key Not Found")


        # ---------------------------------------------
        # Gemini
        # ---------------------------------------------

        if GOOGLE_API_KEY:

            try:

                self.gemini_llm = ChatGoogleGenerativeAI(

                    model="gemini-2.5-flash",

                    temperature=0,

                    google_api_key=GOOGLE_API_KEY

                )

                logger.info("Gemini Loaded Successfully")

            except Exception as e:

                logger.warning(

                    f"Gemini Initialization Failed : {e}"

                )

        else:

            logger.warning("Google API Key Not Found")
    
    #     # ============================================================
    # # CHECK GROQ
    # # ============================================================

    # def check_groq(self):

    #     """
    #     Check whether Groq is available.
    #     """

    #     if self.groq_llm is None:

    #         return False

    #     try:

    #         self.groq_llm.invoke("Hello")

    #         logger.info("Groq Health Check Passed")

    #         return True

    #     except Exception as e:

    #         logger.warning(

    #             f"Groq Health Check Failed : {e}"

    #         )

    #         return False


    # # ============================================================
    # # CHECK GEMINI
    # # ============================================================

    # def check_gemini(self):

    #     """
    #     Check whether Gemini is available.
    #     """

    #     if self.gemini_llm is None:

    #         return False

    #     try:

    #         self.gemini_llm.invoke("Hello")

    #         logger.info("Gemini Health Check Passed")

    #         return True

    #     except Exception as e:

    #         logger.warning(

    #             f"Gemini Health Check Failed : {e}"

    #         )

    #         return False
    # ============================================================
    # GET ACTIVE LLM
    # ============================================================

    def get_llm(self):

        """
        Return the primary available LLM.
        """

        logger.info("=" * 60)
        logger.info("Selecting LLM")
        logger.info("=" * 60)

        if self.groq_llm:

            self.current_provider = "Groq"

            logger.info("Using Groq")

            return self.groq_llm

        if self.gemini_llm:

            self.current_provider = "Gemini"

            logger.info("Using Gemini")

            return self.gemini_llm

        raise RuntimeError(
            "No LLM Available."
        )
        # ============================================================
    # CURRENT PROVIDER
    # ============================================================

    def get_provider(self):

        """
        Return the currently selected provider.
        """

        return self.current_provider
    def invoke(self, query):

        """
        Invoke the active LLM with automatic fallback.
        """

        llm = self.get_llm()

        try:

            return llm.invoke(query)

        except Exception as e:

            logger.warning(
                f"{self.current_provider} Failed : {e}"
            )

            if self.current_provider == "Groq":

                if self.gemini_llm:

                    self.current_provider = "Gemini"

                    logger.info(
                        "Switching to Gemini"
                    )

                    return self.gemini_llm.invoke(query)

            elif self.current_provider == "Gemini":

                if self.groq_llm:

                    self.current_provider = "Groq"

                    logger.info(
                        "Switching to Groq"
                    )

                    return self.groq_llm.invoke(query)

            raise RuntimeError(
                "No LLM Provider Available."
            )
    
        # ============================================================
    # AVAILABLE PROVIDERS
    # ============================================================

    def get_available_providers(self):

        """
        Return all configured LLM providers.
        """

        providers = []

        if self.groq_llm is not None:

            providers.append("Groq")

        if self.gemini_llm is not None:

            providers.append("Gemini")

        return providers
    
        # ============================================================
    # SWITCH PROVIDER
    # ============================================================

    def switch_provider(self, provider):

        """
        Manually switch the active LLM.
        """

        provider = provider.lower()

        if provider == "groq":

            if self.groq_llm is None:

                raise ValueError(
                    "Groq is not available."
                )

            self.current_provider = "Groq"

            logger.info("Switched to Groq")

            return self.groq_llm


        elif provider == "gemini":

            if self.gemini_llm is None:

                raise ValueError(
                    "Gemini is not available."
                )

            self.current_provider = "Gemini"

            logger.info("Switched to Gemini")

            return self.gemini_llm


        else:

            raise ValueError(

                f"Unknown provider : {provider}"

            )
        
        # ============================================================
    # PROVIDER INFORMATION
    # ============================================================

    def provider_info(self):

        """
        Display provider information.
        """

        print("\n" + "=" * 60)

        print("LLM PROVIDERS")

        print("=" * 60)

        print(
            f"Current Provider : {self.current_provider}"
        )

        print()

        print("Available Providers")

        for provider in self.get_available_providers():

            print(f"• {provider}")

        print("=" * 60)

        # ============================================================
    # GET PROVIDER STATUS
    # ============================================================

    def provider_status(self):

        """
        Return provider status.
        """

        return {

            "current_provider": self.current_provider,

            "available_providers":
                self.get_available_providers()

        }
    
# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info("=" * 60)
    logger.info("LLM MANAGER TEST")
    logger.info("=" * 60)

    manager = LLMManager()

    print("\n" + "=" * 70)
    print("AI SHOPPING ASSISTANT")
    print("LLM MANAGER")
    print("=" * 70)

    manager.provider_info()

    try:

        manager.get_llm()

        print(
            f"\nActive Provider : {manager.get_provider()}"
        )

    except Exception as e:

        logger.error(e)

        exit()

    print("\nType 'exit' to quit.\n")

    while True:

        query = input("User : ").strip()

        if query.lower() == "exit":

            print("\nLLM Manager Closed.")

            break

        if query == "":

            continue

        try:

            response = manager.invoke(query)

            print("\n" + "=" * 70)
            print(f"Provider : {manager.get_provider()}")
            print("=" * 70)

            print(response.content)

            print("=" * 70)

        except Exception as e:

            logger.error(e)

            print("\nNo available LLM providers.")