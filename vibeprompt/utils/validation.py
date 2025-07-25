"""
LLM Provider Validation Module - vibeprompt.utils.validation

This module offers validation utilities for LLM provider configurations
in the `vibeprompt` package. It ensures that user-specified model names
and API keys are correct, supported, and functional by making test calls
and comparing environment-provided credentials.

The `Validator` class is designed to help detect and prevent misconfiguration
issues early in the lifecycle of an LLM-based application. It performs:
    - Model name validation against provider-supported models
    - API key presence and correctness checks
    - Detection of changes to credentials via hashing
    - Optional reuse of previously validated keys via caching

Validation errors are raised as `ValidationError` and provide detailed
logging information (controlled by the `verbose` flag).

Usage:
    >>> from vibeprompt.utils.validation import Validator
    >>> from vibeprompt.core.llms.openai_provider import OpenAIProvider

    >>> provider = OpenAIProvider(api_key="sk-...", model_name="gpt-3.5-turbo")
    >>> validator = Validator()
    >>> validator.validate_provider(provider)  # Raises if misconfigured

Classes:
    - ValidationError: Custom exception raised during validation failures.
    - Validator: Performs checks on model names and API key configuration.

Methods:
    - validate_model_name(provider): Checks that the model name is valid.
    - validate_api_key(provider): Validates the provided API key via test call.
    - validate_provider(provider): Performs full validation of model and key.
"""

import hashlib
import os
import logging
from colorama import Fore, Style, init

# Local
from ..core.llms.base import BaseLLMProvider


# Initialize colorama
init(autoreset=True)


# Configure logger
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """
    Custom exception raised when validation of a provider fails.

    Typically used to indicate:
        - Unsupported or invalid model name
        - Missing or incorrect API key
        - Inability to connect or authenticate with the provider's API
    """
    pass


class Validator:
    """
    Validator class for LLM provider configuration.

    This utility checks whether a provider instance is properly configured
    by validating its model name and API key. It supports caching of validated
    API keys using environment variables and detects changes via secure hashing.

    Attributes:
        - verbose (bool): If True, logs detailed validation steps to stdout.
    """

    def __init__(self, verbose: bool=True):
        """
        Initialize a Validator instance.

        Args:
            - verbose (bool): Whether to print detailed logs during validation (default: True).
        """

        self.verbose = verbose
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            # Show only final errors or warnings
            logger.setLevel(logging.WARNING)

    def validate_model_name(self, provider: BaseLLMProvider) -> None:
        """
        Validate that the model name specified in the provider is supported.

        Looks up valid models from the provider and checks if the current one
        matches. Logs success or failure depending on the result.

        Args:
            - provider (BaseLLMProvider): An instance of an LLM provider.

        Raises:
            - ValidationError: If the model name is not supported by the provider.

        Example:
            >>> validator.validate_model_name(provider)
        """

        logger.info(f"{Fore.CYAN}🔍 Validating model name `{provider.model_name}` for `{provider.get_provider_name()}`...{Style.RESET_ALL}")
        
        valid_models = provider.get_valid_models()
        if provider.model_name not in valid_models:
            if self.verbose:
                error_msg = (
                    f"Model '{provider.model_name}' is not valid for `{provider.get_provider_name()}`. "
                    f"Valid models: {valid_models}"
                )
                logger.error(f"{Fore.RED}❌ Model validation failed: {error_msg}{Style.RESET_ALL}")
            raise ValidationError(error_msg)
        
        logger.info(f"{Fore.GREEN}✅ Model `{provider.model_name}` is valid for `{provider.get_provider_name()}`{Style.RESET_ALL}")

    def validate_api_key(self, provider: BaseLLMProvider) -> None:
        """
        Validate the API key of the provider by making a test LLM call.

        Supports reading the API key from:
            1. The provider's `api_key` attribute
            2. An environment variable named `<PROVIDER>_API_KEY`

        If the key has been validated previously and hasn't changed, the cached
        validation is reused unless a change is detected via hashing.

        Args:
            - provider (BaseLLMProvider): The LLM provider to validate.

        Raises:
            - ValidationError: If no API key is found or a test call fails.

        Example:
            >>> validator.validate_api_key(provider)
        """

        provider_name = provider.get_provider_name()
        provider_name_upper = provider_name.upper()
        
        # Get API key from argument or environment variable
        api_key_from_arg = provider.api_key
        api_key_from_env = os.environ.get(f"{provider_name_upper}_API_KEY")
        
        # Determine which API key to use (argument takes precedence)
        current_api_key = api_key_from_arg or api_key_from_env
        
        # Check if API key exists
        if not current_api_key:
            error_msg = (
                f"API key is required for `{provider_name}`. You can either:\n"
                f"  1. Pass it as argument: api_key='sk-***'\n"
                f"  2. Set environment variable: {provider_name_upper}_API_KEY='sk-***'"
            )
            logger.error(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            raise ValidationError(error_msg)
        
        # Create hash of current API key for change detection
        current_key_hash = hashlib.sha256(current_api_key.encode()).hexdigest()
        
        # Get stored hash and validation status
        stored_key_hash = os.environ.get(f"{provider_name_upper}_API_KEY_HASH")
        is_validated = os.environ.get(f"{provider_name_upper}_API_KEY_VALIDATED") == "true"
        
        # Log which method is being used
        if api_key_from_arg:
            if api_key_from_env and api_key_from_arg != api_key_from_env:
                logger.info(f"{Fore.YELLOW}⚠️ API key from argument differs from environment variable. Using argument value.{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}🔑 Using API key from function argument for `{provider_name}`{Style.RESET_ALL}")
        else:
            logger.info(f"{Fore.CYAN}🌍 Using API key from environment variable for `{provider_name}`{Style.RESET_ALL}")
        
        # Check if validation is needed
        needs_validation = (
            not is_validated or 
            stored_key_hash != current_key_hash or 
            stored_key_hash is None
        )
        
        if not needs_validation:
            logger.info(f"{Fore.GREEN}📀 Using cached validated API key for `{provider_name}`!{Style.RESET_ALL}")
            # Update provider's api_key to ensure consistency
            provider.api_key = current_api_key
            return
        
        # Log why validation is needed
        if not is_validated:
            logger.info(f"{Fore.YELLOW}🔍 API key for `{provider_name}` not validated yet{Style.RESET_ALL}")
        elif stored_key_hash != current_key_hash:
            logger.info(f"{Fore.YELLOW}🔄 API key for `{provider_name}` has changed, re-validation required{Style.RESET_ALL}")
        
        # Perform validation
        try:
            logger.info(f"{Fore.YELLOW}🔑 Validating API key for `{provider_name}`...{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}🧪 Making test call to `{provider_name}` API...{Style.RESET_ALL}")
            
            # Temporarily set the API key in provider for testing
            original_api_key = provider.api_key
            provider.api_key = current_api_key
            
            # Create a temporary LLM instance for testing
            test_llm = provider.create_llm_instance()
            
            # Make a simple test call
            test_response = test_llm.invoke("Hello")
            
            if test_response:
                # Save API key, hash, and validation flag to environment variables
                os.environ[f"{provider_name_upper}_API_KEY"] = current_api_key
                os.environ[f"{provider_name_upper}_API_KEY_HASH"] = current_key_hash
                os.environ[f"{provider_name_upper}_API_KEY_VALIDATED"] = "true"
                
                logger.info(f"{Fore.LIGHTMAGENTA_EX}💾 API key and validation status saved to environment{Style.RESET_ALL}")
                
                # Ensure provider has the validated API key
                provider.api_key = current_api_key
                
            else:
                # Restore original API key if validation failed
                provider.api_key = original_api_key
                raise Exception("API test call returned empty response")
                
        except Exception as e:
            # Restore original API key if validation failed
            provider.api_key = original_api_key
            
            # Clear validation status on failure
            if f"{provider_name_upper}_API_KEY_VALIDATED" in os.environ:
                del os.environ[f"{provider_name_upper}_API_KEY_VALIDATED"]
            if f"{provider_name_upper}_API_KEY_HASH" in os.environ:
                del os.environ[f"{provider_name_upper}_API_KEY_HASH"]
            
            error_msg = f"API key validation failed for `{provider_name}`: {str(e)}"
            logger.error(f"{Fore.RED}❌ {error_msg}{Style.RESET_ALL}")
            raise ValidationError(error_msg)

    def validate_provider(self, provider: BaseLLMProvider) -> None:
        """
        Run all validation checks for the given provider.

        This is a convenience method that sequentially calls:
            - validate_model_name()
            - validate_api_key()

        Args:
            - provider (BaseLLMProvider): The LLM provider to fully validate.

        Raises:
            - ValidationError: If any validation check fails.

        Example:
            >>> validator.validate_provider(provider)
        """

        logger.info(f"{Fore.CYAN}🚀 Starting validation for {provider.get_provider_name()} provider...{Style.RESET_ALL}")
        
        try:
            self.validate_model_name(provider)
            self.validate_api_key(provider)
            logger.info(f"{Fore.LIGHTGREEN_EX}🎉 All validations passed for {provider.get_provider_name()}!{Style.RESET_ALL}")
        except ValidationError:
            logger.error(f"{Fore.RED}💥 Validation failed for {provider.get_provider_name()}{Style.RESET_ALL}")
            raise
