"""
Safety and Moderation Utilities Module - vibeprompt.utils.safety

This module provides content safety and moderation capabilities for the `vibeprompt` package.
It uses a LangChain-compatible language model to assess whether user-submitted text is safe
for general consumption, based on predefined moderation categories.

The `SafetyChecker` class uses a structured moderation prompt and a strict JSON schema to detect
potentially harmful content such as:
    - Violence or threats
    - Sexual or adult content
    - Harassment or bullying
    - Discrimination or hate speech
    - Self-harm or suicide
    - Misinformation
    - Criminal or unethical behavior

Upon analyzing a given piece of text, the system returns a structured verdict along with an optional
suggestion and a list of triggering categories.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.utils.safety import SafetyChecker

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="openai",
    ...     api_key="your-openai-api-key"
    ... )

    >>> checker = SafetyChecker(llm)
    >>> checker.is_safe("Here's how to make a bomb.")
    False

Classes:
    - ModerationResult: Represents the structured result of a moderation check.
    - SafetyChecker: Checks whether input text is safe based on predefined criteria.

Functions:
    - is_safe(text: str) -> bool:
        Returns True if the text is considered safe; False if flagged for harmful content.
"""

import logging
from typing import List, Optional
from colorama import Fore, init, Style
from pydantic import BaseModel

from langchain.chat_models.base import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# Initialize colorama
init(autoreset=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ModerationResult(BaseModel):
    """
    Structured result returned by the safety moderation chain.

    Attributes:
        - is_safe (bool): Indicates whether the content is considered safe.
        - category (Optional[List[str]]): List of categories that triggered moderation, if any.
        - reason (str): A short explanation of the moderation decision.
        - suggestion (Optional[str]): Optional guidance or correction to make the text safe.
    """

    is_safe: bool
    category: Optional[List[str]] = []
    reason: str
    suggestion: Optional[str] = ""


class SafetyChecker:
    """
    Content safety and moderation checker.

    This class evaluates text for safety using a LangChain-compatible LLM,
    structured prompting, and Pydantic-based output parsing. It flags content
    that is potentially harmful or inappropriate for general audiences.

    Supported safety categories:
        - Violence, abuse, or threats
        - Sexual or adult content
        - Harassment or personal attacks
        - Hate speech or discrimination
        - Self-harm or suicide
        - Misinformation or conspiracies
        - Illegal or unethical activities

    Attributes:
        - llm (BaseChatModel): The language model used for moderation.
        - verbose (bool): Controls logging verbosity of moderation decisions.
    """

    def __init__(self, llm: BaseChatModel, verbose: bool = True):
        """
        Initialize the SafetyChecker with a language model and configuration.

        Args:
            - llm (BaseChatModel): A LangChain-compatible LLM instance.
            - verbose (bool): Whether to log detailed output (default: True).
        """

        self.prompt_temp = """You are a professional content safety and 
        moderation system.

        Your task is to analyze a given piece of text and determine whether 
        it is safe or unsafe according to the following criteria.

        A piece of text is considered **UNSAFE** if it contains or promotes 
        **any** of the following:
        - Violence or threats (e.g., hate speech, abuse, terrorism, or 
        incitement to violence)
        - Sexual content or innuendo (e.g., sexually explicit language, adult 
        themes)
        - Harassment or bullying (e.g., personal attacks, targeted insults, 
        doxing)
        - Discrimination or hate (e.g., racism, sexism, homophobia, transphobia, 
        ableism)
        - Self-harm or suicide content (e.g., encouragement or glorification 
        of self-injury)
        - Misinformation (e.g., dangerous medical advice or conspiracies)
        - Criminal activity (e.g., drug trafficking, fraud, hacking instructions)
        - Any other content that could be considered harmful, illegal, unethical, 
        or inappropriate for general audiences (e.g., children or vulnerable users)

        If the text does **not** contain any of the above, then it is considered **SAFE**.
        
        Your output must be a single JSON instance containing the following keys:
        is_safe, category, reason, and suggestion

        Don't add any extra thing and follow the format instructions down below.

        Text:
        {text}

        Format instructions:
        {format_instructions}
        """

        self.parser = JsonOutputParser(pydantic_object=ModerationResult)

        self.prompt_template = PromptTemplate(
            template=self.prompt_temp,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            # Show only final errors or warnings
            logger.setLevel(logging.WARNING) 

        self.llm = llm
        self.moderation_chain = self.prompt_template | self.llm | self.parser

        logger.info(Fore.GREEN + "🦺 SafetyChecker initialized successfully" + Style.RESET_ALL)

    def is_safe(self, text: str) -> bool:
        """
        Determine whether the input text is safe for general audiences.

        Sends the text through the moderation pipeline and logs an error if it is flagged.
        A `True` result means the content passed all safety checks.
        A `False` result indicates the content is considered unsafe, and the reason is logged.

        Args:
            - text (str): The input string to be evaluated for safety.

        Returns:
            - bool: True if content is safe, False otherwise.

        Example:
            >>> checker.is_safe("How to make a homemade bomb?")
            False
        """

        if not text:
            return False

        text = text.lower()
        output = self.moderation_chain.invoke({"text": text})

        if output["is_safe"] == True:
            return True
        else:
            logger.error(Fore.RED + " | ".join(f"{k}: {v}" for k, v in output.items()))
            return False
