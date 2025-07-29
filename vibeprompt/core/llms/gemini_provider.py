"""
Google Gemini LLM Provider - vibeprompt.core.llms.gemini_provider

This module implements the `GeminiProvider` class, a concrete subclass of
`BaseLLMProvider`, to integrate Google’s Gemini models into the `vibeprompt` framework.

It supports multiple Gemini versions (e.g., `gemini-2.5-pro`, `gemini-2.0-flash-lite`)
via LangChain’s `ChatGoogleGenerativeAI` interface. This provider handles
model validation, default fallback, and runtime instantiation with configurable
parameters like temperature and max output tokens.

Usage:
    >>> from vibeprompt.core.llms.gemini_provider import GeminiProvider
    >>> provider = GeminiProvider(api_key="sk-google-...")
    >>> llm = provider.get_llm()
    >>> response = llm.invoke("Describe how neural networks learn.")

Classes:
    - GeminiProvider: Concrete LLM provider for Google's Gemini models.
"""
import warnings

warnings.filterwarnings("ignore")

from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models.base import BaseChatModel

# Local
from .base import BaseLLMProvider


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider class for Gemini models.

    This class integrates with Google’s Gemini API via LangChain, enabling support
    for multiple Gemini model variants across v2.0 and v2.5 series with optional
    lightweight alternatives.

    Inherits from:
        - BaseLLMProvider

    Methods Implemented:
        - get_default_model()
        - get_provider_name()
        - get_valid_models()
        - create_llm_instance()
    """

    def get_default_model(self) -> str:
        """
        Return the default Gemini model name.

        Returns:
            - str: Default model name, e.g., "gemini-2.0-flash".
        """
        return "gemini-2.0-flash"

    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            - str: "Gemini"
        """
        return "Gemini"

    def get_valid_models(self) -> List[str]:
        """
        Return the list of valid Gemini model names.

        Returns:
            - List[str]: Supported Gemini models.
        """
        return [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
        ]

    def create_llm_instance(self) -> BaseChatModel:
        """
        Instantiate and return the Gemini LangChain LLM.

        This method constructs a `ChatGoogleGenerativeAI` object using the configured model,
        API key, and additional parameters such as temperature and output token limit.

        Returns:
            - BaseChatModel: LangChain-compatible Gemini LLM instance.
        """
        return ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            **{
                k: v
                for k, v in self.config.items()
            }
        )
