"""
LLM Provider Factory - vibeprompt.core.llms.provider_factory

This module implements the `LLMProviderFactory` class, a centralized factory
used to instantiate LLM providers and create configured LangChain LLM instances
based on the user's selected provider (e.g., OpenAI, Cohere, Gemini, Anthropic).

It supports validation, dynamic provider lookup, logging, and runtime
configuration including custom model selection and parameters like temperature
and token limits.

Usage:
    >>> from vibeprompt.core.llms.provider_factory import LLMProviderFactory
    >>> llm = LLMProviderFactory.create_provider("openai", api_key="sk-...", model_name="gpt-4")
    >>> response = llm.invoke("Summarize this paragraph...")

Classes:
    - LLMProviderFactory: Factory for building and configuring LLM provider instances.
"""

import logging
from typing import Optional, Dict, Type
from langchain.chat_models.base import BaseChatModel
from colorama import Fore, Style, init

# Local
from .base import BaseLLMProvider
from ..llms import CohereProvider, OpenAIProvider, AnthropicProvider, GeminiProvider


# Initialize colorama
init(autoreset=True)


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class LLMProviderFactory:
    """
    Factory for creating LLM provider instances.

    This class serves as a dynamic factory that registers and instantiates
    LLM providers from supported vendors (e.g., OpenAI, Cohere, Gemini, Anthropic),
    returning a fully configured LangChain-compatible LLM instance.

    Class Attributes:
        - _providers (dict): Internal registry mapping provider names to their classes.

    Class Methods:
        - create_provider: Instantiate and configure a provider instance.
        - get_available_providers: Return list of supported providers.
        - get_provider_models: Return valid model names for a given provider.
    """
    
    _providers: Dict[str, Type[BaseLLMProvider]] = {
        'cohere': CohereProvider,
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
        'gemini': GeminiProvider,
    }
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        api_key: str,
        model_name: Optional[str] = None,
        verbose: bool = False,
        **config
    ) -> BaseChatModel:
        """
        Create and return a configured LLM instance from the specified provider.

        This method looks up the appropriate provider class, initializes it with the
        given API key, model name, and additional configuration options, and returns
        the associated LangChain-compatible LLM.

        Args:
            - provider_name (str): Name of the provider (e.g., "openai", "cohere").
            - api_key (str): The API key for authentication.
            - model_name (Optional[str]): Optional model to use. Defaults to provider's default.
            - verbose (bool): Whether to log detailed progress output. Defaults to False.
            - **config: Additional config options (e.g., temperature, max_tokens).

        Returns:
            - BaseChatModel: A configured LangChain LLM instance ready for use.

        Raises:
            - ValueError: If an unsupported provider is specified.
        """

        cls.verbose = verbose
        if cls.verbose:
            logger.setLevel(logging.INFO)
        else:
            # Show only final errors or warnings
            logger.setLevel(logging.WARNING)

        logger.info(f"{Fore.CYAN}🏭 LLM Factory: Creating provider '{provider_name}'...{Style.RESET_ALL}")
        
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower not in cls._providers:
            available_providers = list(cls._providers.keys())
            error_msg = (
                f"Unsupported provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )
            logger.error(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            raise ValueError(error_msg)
        
        logger.info(f"{Fore.GREEN}✅ Provider '{provider_name}' found in registry{Style.RESET_ALL}")
        provider_class = cls._providers[provider_name_lower]
        
        # Log configuration details
        if model_name:
            logger.info(f"{Fore.BLUE}🎯 Custom model specified: {model_name}{Style.RESET_ALL}")
        
        if config:
            logger.info(f"{Fore.MAGENTA}⚙️ Additional config: {config}{Style.RESET_ALL}")
        
        provider_instance = provider_class(
            model_name=model_name,
            api_key=api_key,
            verbose=verbose,
            **config
        )
        
        return provider_instance.get_llm()
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Return a list of available provider names.

        Returns:
            - list: Supported LLM providers (e.g., ["openai", "cohere"]).
        """
        providers = list(cls._providers.keys())
        logger.info(f"{Fore.CYAN}📋 Available providers: {', '.join(providers)}{Style.RESET_ALL}")
        return providers
    
    @classmethod
    def get_provider_models(cls, provider_name: str) -> list:
        """
        Return list of valid model names for the given provider.

        Args:
            - provider_name (str): The name of the provider (e.g., "openai").

        Returns:
            - list: Supported model names for the given provider.

        Raises:
            - ValueError: If the provider is not recognized.
        """
        logger.info(f"{Fore.CYAN}🔍 Fetching available models for '{provider_name}'...{Style.RESET_ALL}")
        
        provider_name_lower = provider_name.lower()
        
        if provider_name_lower not in cls._providers:
            available_providers = ', '.join(cls._providers.keys())
            error_msg = (
                f"Unsupported provider '{provider_name}'. "
                f"Available providers: {available_providers}"
            )
            logger.error(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            raise ValueError(error_msg)
        
        provider_class = cls._providers[provider_name_lower]
        # Create temporary instance to get valid models
        temp_instance = provider_class(api_key="temp")
        models = temp_instance.get_valid_models()
        
        logger.info(f"{Fore.GREEN}📋 Found {len(models)} models for {provider_name}: {models}{Style.RESET_ALL}")
        return models