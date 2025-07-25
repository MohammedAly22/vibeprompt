"""
LLM Provider Base Class - vibeprompt.core.llms.base

This module defines the abstract base class `BaseLLMProvider` for integrating
various large language model (LLM) providers into the `vibeprompt` package.

It provides a consistent interface for initializing and validating LLM providers,
such as OpenAI, Anthropic, or others compatible with LangChain. Each concrete
provider subclass must implement methods for model selection, provider identification,
and instance creation using LangChain's `BaseChatModel`.

This system ensures that all provider classes:
    - Declare supported model names
    - Provide default model fallbacks
    - Validate configuration before runtime
    - Delay LLM instantiation until required

Usage:
    >>> from vibeprompt.core.llms.openai import OpenAIProvider
    >>> provider = OpenAIProvider(api_key="sk-...")
    >>> llm = provider.get_llm()
    >>> llm.invoke("Explain relativity to a child.")

Classes:
    - BaseLLMProvider: Abstract base class for all LLM providers.

Methods to implement:
    - get_default_model()
    - get_provider_name()
    - get_valid_models()
    - create_llm_instance()
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from langchain.chat_models.base import BaseChatModel
from colorama import Fore, Style, init


# Initialize colorama
init(autoreset=True)

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers in the `vibeprompt` framework.

    This class defines the interface that all provider-specific subclasses must implement,
    including methods for model selection, validation, and instantiation of a LangChain-compatible
    chat model. It also supports deferred loading of the LLM instance.

    Attributes:
        - model_name (Optional[str]): The selected or default model name.
        - api_key (Optional[str]): The API key for the provider service.
        - config (dict): Additional keyword arguments for provider configuration.
        - verbose (bool): Enables detailed logging if True.
        - _llm_instance (Optional[BaseChatModel]): The cached LLM instance (created lazily).
    """
    
    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None, verbose: bool = False, **kwargs):
        """
        Initialize the provider with optional model name and API key.

        Args:
            - model_name (Optional[str]): Name of the LLM to use. If None, uses default model.
            - api_key (Optional[str]): API key for authenticating with the provider.
            - verbose (bool): Whether to enable detailed logging (default: False).
            - **kwargs: Additional configuration options passed to the provider.

        Logs:
            - Model selection and provider initialization details.
        """

        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            # Show only final errors or warnings
            logger.setLevel(logging.WARNING)

        logger.info(f"{Fore.CYAN}🏗️ Initializing `{self.get_provider_name()}` provider...{Style.RESET_ALL}")
        
        self.model_name = model_name or self.get_default_model()
        self.api_key = api_key
        self.config = kwargs
        self._llm_instance = None
        
        if not model_name:
            logger.info(f"{Fore.YELLOW}⚙️ Using default model: `{self.model_name}`{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}🎯 Using specified model: `{self.model_name}`{Style.RESET_ALL}")
        
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Return the default model name for this provider.

        Returns:
            - str: Default model name (e.g., "gpt-4").
        """

        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Return the unique provider name.

        Returns:
            - str: Provider identifier (e.g., "openai", "anthropic").
        """

        pass
    
    @abstractmethod
    def get_valid_models(self) -> list:
        """
        Return a list of supported model names for this provider.

        Returns:
            - list[str]: A list of valid model names.
        """

        pass
    
    @abstractmethod
    def create_llm_instance(self) -> BaseChatModel:
        """
        Instantiate and return a LangChain-compatible LLM instance.

        This should include model name, API credentials, and other config parameters.

        Returns:
            - BaseChatModel: The instantiated LangChain chat model.
        """

        pass
    
    def get_llm(self) -> BaseChatModel:
        """
        Retrieve or create the LangChain LLM instance.

        If not already created, this method:
            - Runs provider validation
            - Creates the LLM instance
            - Caches and returns it

        Returns:
            - BaseChatModel: The initialized LLM instance.

        Raises:
            - Exception: If validation or instantiation fails.

        Logs:
            - LLM creation progress and success message.
        """
        
        if self._llm_instance is None:
            logger.info(f"{Fore.LIGHTMAGENTA_EX}🔧 Creating LLM instance for `{self.get_provider_name()}`...{Style.RESET_ALL}")
            from ...utils.validation import Validator
            validator = Validator(self.verbose)
            validator.validate_provider(self)
            self._llm_instance = self.create_llm_instance()
            logger.info(f"{Fore.GREEN}✨ LLM instance created successfully and ready to run!{Style.RESET_ALL}")
            logger.info("=" * 61)
            
        return self._llm_instance