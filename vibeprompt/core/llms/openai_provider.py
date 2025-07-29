"""
OpenAI LLM Provider - vibeprompt.core.llms.openai_provider

This module implements the `OpenAIProvider` class, a concrete subclass of
`BaseLLMProvider`, to integrate OpenAI’s GPT models into the `vibeprompt` framework.

It supports multiple GPT model versions (e.g., `gpt-4`, `gpt-4o`, `gpt-3.5-turbo`)
via LangChain’s `ChatOpenAI` interface. This provider manages model validation,
default selection, and LLM instantiation with configurable parameters such as
temperature and max token limits.

Usage:
    >>> from vibeprompt.core.llms.openai_provider import OpenAIProvider
    >>> provider = OpenAIProvider(api_key="sk-openai-...")
    >>> llm = provider.get_llm()
    >>> response = llm.invoke("List 5 benefits of daily exercise.")

Classes:
    - OpenAIProvider: Concrete LLM provider for OpenAI's GPT models.
"""

from typing import List
from langchain_openai import ChatOpenAI
from langchain.chat_models.base import BaseChatModel

# Local
from .base import BaseLLMProvider


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM provider class for GPT models.

    This class integrates with OpenAI’s GPT APIs via LangChain, supporting
    popular models including GPT-3.5, GPT-4, GPT-4-turbo, and GPT-4o.

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
        Return the default OpenAI model name.

        Returns:
            - str: Default model name, e.g., "gpt-3.5-turbo".
        """
        return "gpt-3.5-turbo"

    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            - str: "OpenAI"
        """
        return "OpenAI"

    def get_valid_models(self) -> List[str]:
        """
        Return the list of valid OpenAI GPT model names.

        Returns:
            - List[str]: Supported GPT models.
        """
        return [
            "gpt-4",
            "gpt-4-turbo",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]

    def create_llm_instance(self) -> BaseChatModel:
        """
        Instantiate and return the OpenAI LangChain LLM.

        This method constructs a `ChatOpenAI` object using the selected model,
        API key, and configurable parameters such as temperature and max tokens.

        Returns:
            - BaseChatModel: LangChain-compatible OpenAI LLM instance.
        """
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.api_key,
            **{k: v for k, v in self.config.items()}
        )
