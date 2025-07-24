"""
Anthropic LLM Provider - vibeprompt.core.llms.anthropic_provider

This module implements the `AnthropicProvider` class, a concrete subclass of
`BaseLLMProvider`, to integrate Anthropic's Claude models into the `vibeprompt` framework.

It supports multiple Claude models (e.g., Claude 3 Opus, Claude 2.1) via LangChain's
`ChatAnthropic` interface. The provider handles model validation, default selection,
and instantiation with configurable parameters like temperature and token limits.

Usage:
    >>> from vibeprompt.core.llms.anthropic import AnthropicProvider
    >>> provider = AnthropicProvider(api_key="sk-ant-...")
    >>> llm = provider.get_llm()
    >>> response = llm.invoke("Summarize quantum computing for high schoolers.")

Classes:
    - AnthropicProvider: Concrete LLM provider for Anthropic Claude models.
"""

from typing import List
from langchain_anthropic import ChatAnthropic
from langchain.chat_models.base import BaseChatModel

# Local
from .base import BaseLLMProvider


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic LLM provider class for Claude models.

    This provider integrates with Anthropic's Claude API via LangChain, enabling
    dynamic model selection and lazy instantiation based on user configuration.

    Supported models include Claude 3 (Opus, Sonnet, Haiku) and Claude 2.x series.

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
        Return the default Anthropic model name.

        Returns:
            - str: Default model name, e.g., "claude-3-sonnet-20240229".
        """
        return "claude-3-sonnet-20240229"

    def get_provider_name(self) -> str:
        """
        Return the provider name.

        Returns:
            - str: "Anthropic"
        """
        return "Anthropic"

    def get_valid_models(self) -> List[str]:
        """
        Return the list of valid Claude model names.

        Returns:
            - List[str]: Supported Anthropic models.
        """
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
        ]

    def create_llm_instance(self) -> BaseChatModel:
        """
        Instantiate and return the Anthropic LangChain LLM.

        This method constructs a `ChatAnthropic` object using the configured model,
        API key, and additional parameters such as temperature and max_tokens.

        Returns:
            - BaseChatModel: LangChain-compatible Claude instance.
        """

        return ChatAnthropic(
            model=self.model_name,
            anthropic_api_key=self.api_key,
            temperature=self.config.get('temperature', 0.7),
            max_tokens=self.config.get('max_tokens', 1000),
            **{k: v for k, v in self.config.items() 
               if k not in ['temperature', 'max_tokens']}
        )
