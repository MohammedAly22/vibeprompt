"""
Cohere LLM Provider - vibeprompt.core.llms.cohere_provider

This module implements the `CohereProvider` class, a concrete subclass of
`BaseLLMProvider`, to integrate Cohere’s Command models into the `vibeprompt` framework.

It supports multiple Cohere models (e.g., `command-a-03-2025`, `command-r-plus-04-2024`)
via LangChain’s `ChatCohere` interface. This provider manages model validation,
default selection, and instantiation with configurable parameters such as temperature
and max token limits.

Usage:
    >>> from vibeprompt.core.llms.cohere_provider import CohereProvider
    >>> provider = CohereProvider(api_key="sk-cohere-...")
    >>> llm = provider.get_llm()
    >>> response = llm.invoke("Write a short poem about spring.")

Classes:
    - CohereProvider: Concrete LLM provider for Cohere's Command models.
"""

from typing import List
from langchain_cohere import ChatCohere
from langchain.chat_models.base import BaseChatModel

# Local
from .base import BaseLLMProvider


class CohereProvider(BaseLLMProvider):
    """
    Cohere LLM provider class for Command models.

    This class integrates with Cohere’s LLM API via LangChain, supporting
    a range of models including Command-A, Command-R, and lightweight variants.

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
        Return the default Cohere model name.

        Returns:
            - str: Default model name, e.g., "command-a-03-2025".
        """
        return "command-a-03-2025"

    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            - str: "Cohere"
        """
        return "Cohere"

    def get_valid_models(self) -> List[str]:
        """
        Return the list of valid Command model names.

        Returns:
            - List[str]: Supported Cohere models.
        """
        return [
            "command-a-03-2025",
            "command-r-plus-04-2024",
            "command-r",
            "command-light",
            "command-xlarge",
        ]

    def create_llm_instance(self) -> BaseChatModel:
        """
        Instantiate and return the Cohere LangChain LLM.

        This method constructs a `ChatCohere` object using the selected model,
        API key, and optional parameters such as temperature and max tokens.

        Returns:
            - BaseChatModel: LangChain-compatible Cohere LLM instance.
        """

        return ChatCohere(
            model=self.model_name,
            cohere_api_key=self.api_key,
            **{k: v for k, v in self.config.items()}
        )
