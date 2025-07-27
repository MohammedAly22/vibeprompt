"""
Base Styler - vibeprompt.core.base_styler

This module implements the `PromptStyler` class, a user-facing utility that applies
style, audience targeting, and safety validation to prompts before sending them to
a language model.

It supports pluggable style transformations (e.g., "playful", "technical"), audience
adjustments (e.g., "children", "experts"), and optional safety checking for harmful
or unsafe content. The transformed prompt is then ready to be passed to an LLM
provider via LangChain.

Usage:
    >>> styler = PromptStyler(style="playful", audience="children", provider="openai", api_key="sk-...")
    >>> transformed_prompt = styler.transform("Explain gravity.")
    >>> print(transformed_prompt)

Classes:
    - PromptStyler: Main interface for transforming and validating prompts.
"""


import logging
from colorama import Fore, Style
from typing import Literal, Optional, Union

# Local
from .style_registry import load_style, load_audience
from .llms.factory import LLMProviderFactory
from ..utils.safety import SafetyChecker

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Create static Enums
StyleType = Literal[
    "assertive",
    "formal",
    "humorous",
    "playful",
    "poetic",
    "sequential",
    "simple",
    "technical"
]

AudienceType = Literal[
    "business",
    "children",
    "developers",
    "experts",
    "general",
    "healthcare",
    "students",
]

ProviderType = Literal[
    "anthropic",
    "cohere",
    "gemini",
    "openai",
]

AnthropicModelType = Literal[
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-2.1",
    "claude-2.0",
]

CohereModelType = Literal[
    "command-a-03-2025",
    "command-r-plus-04-2024",
    "command-r",
    "command-light",
    "command-xlarge",
]

GeminiModelType = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
]

OpenAIModelType = Literal[
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4o",
    "gpt-3.5-turbo",
]

ModelType = Union[AnthropicModelType, CohereModelType, GeminiModelType, OpenAIModelType]


class PromptStyler:
    """
    Interface for transforming prompts with custom style, audience, and safety validation.

    This class applies transformations to user-provided prompts to match a selected
    communication style and/or target audience. It also performs optional safety checks
    on the input and output using a large language model.

    Attributes:
        - style_name (str): The name of the selected style (e.g., "technical").
        - audience_name (Optional[str]): The intended audience (e.g., "students").
        - model_name (Optional[str]): The model to use from the selected provider.
        - provider_name (str): The LLM provider (e.g., "openai").
        - api_key (Optional[str]): API key for authenticating with the provider.
        - enable_safety (bool): Whether safety checking is enabled.
        - verbose (bool): Whether to log progress and transformation steps.

    Methods:
        - transform(prompt: str) -> str: Applies the full transformation pipeline to a prompt.
    """

    def __init__(
        self,
        provider: Optional[ProviderType] = "cohere", 
        model: Optional[ModelType] = None, 
        api_key: Optional[str] = None, 
        enable_safety: bool = True, 
        verbose: bool = False,
        **config
    ) -> None:
        """
        Initialize a PromptStyler instance with selected options.

        Args:
            - provider (Optional[ProviderType]): LLM provider to use (default: "cohere").
            - model (Optional[ModelType]): Optional specific model name.
            - api_key (Optional[str]): API key for provider authentication.
            - enable_safety (bool): Enable prompt/content safety checks. Default: True.
            - verbose (bool): Enable logging and CLI display. Default: False.
        """

        self.verbose = verbose

        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            # Show only final errors or warnings
            logger.setLevel(logging.WARNING)  

        logger.info(
            f"🎨 Initializing PromptStyler with provider=`{Fore.CYAN + provider}`" + Style.RESET_ALL
        )

        self.model_name = model
        self.provider_name = provider
        self.api_key = api_key
        self.enable_safety = enable_safety

        # Initialize LLM
        self.provider = LLMProviderFactory.create_provider(
            provider_name=provider,
            model_name=self.model_name,
            api_key=self.api_key,
            verbose=self.verbose,
            **config
        )
        
        self.llm = self.provider.get_llm()

        # Initialize safety checker
        self.safety_checker = SafetyChecker(llm=self.llm, verbose=self.verbose) if self.enable_safety else None

        # Check if it's none, notify the user
        if not self.safety_checker:
            logger.info(Fore.LIGHTYELLOW_EX + "⚠️ Warning: The SafetyChecker is currently disabled. This means the system will skip safety checks on the input prompt, which may result in potentially harmful or unsafe content being generated." + Style.RESET_ALL)
            logger.info(Fore.CYAN + "💡 Tip: Enable the `enable_safety=True` to ensure prompt safety validation is applied." + Style.RESET_ALL)

        logger.info(Fore.GREEN +"🧙🏼‍♂️ PromptStyler initialized successfully!" + Style.RESET_ALL)

    def transform(self, prompt: str, style: StyleType = "simple", audience: Optional[AudienceType] = None) -> str:
        """
        Transform a prompt with the selected style, audience, and safety constraints.

        This method applies the style and audience transformations in sequence
        and validates the input/output using a safety checker (if enabled).

        Args:
            - prompt (str): The raw input prompt to transform.
            - style (StyleType): The transformation style to apply (default: "simple").
            - audience (Optional[AudienceType]): Optional audience target.

        Returns:
            - str: The fully transformed and validated prompt string.

        Raises:
            - ValueError: If the prompt is empty or fails safety checks.
            - Exception: For any unexpected transformation errors.
        """

        if not prompt or not prompt.strip():
            raise ValueError(Fore.RED +"❌ Prompt cannot be empty" + Style.RESET_ALL)
        
        self.style_name = style
        self.audience_name = audience

        # Load transformation modules
        self.style = load_style(style)
        self.audience = load_audience(audience) if audience else None
        
        if self.audience:
            logger.info(
                f"🎨 Configured PromptStyler with style=`{Fore.CYAN + style}` {Style.RESET_ALL}, audience=`{Fore.CYAN + audience}`" + Style.RESET_ALL
            )
        else:
            logger.info(
                f"🎨 Configured PromptStyler with style={Fore.CYAN + style} {Style.RESET_ALL}"
            )

        prompt = prompt.strip()
        logger.info(Fore.CYAN + f"✨ Transforming prompt: {prompt[:30]}..." + Style.RESET_ALL)

        # Safety check input
        if self.safety_checker and not self.safety_checker.is_safe(prompt):
            raise ValueError(Fore.RED + "❌ Input prompt failed safety checks" + Style.RESET_ALL)
        
        # Apply transformation in order
        try:
            # Apply style transformation
            styled_prompt = self.style.apply(text=prompt, llm=self.llm)
            logger.info(Fore.GREEN + "🖌️ Style transformation completed" + Style.RESET_ALL)
            logger.info(Fore.LIGHTBLUE_EX + styled_prompt + Style.RESET_ALL)

            # Apply audience transformation
            if self.audience:
                styled_prompt = self.audience.apply(text=styled_prompt, llm=self.llm)
                logger.info(Fore.GREEN + "🧑🏼‍🦰 Audience transformation completed" + Style.RESET_ALL)
                logger.info(Fore.LIGHTBLUE_EX + styled_prompt + Style.RESET_ALL)
            
            # Final safety check
            if self.safety_checker and not self.safety_checker.is_safe(styled_prompt):
                raise ValueError(Fore.RED + "❌ Generated content failed safety checks"+ Style.RESET_ALL)
            
            logger.info(f"\n{Fore.LIGHTWHITE_EX}{'=' * 61}")
            logger.info(f"{Fore.LIGHTWHITE_EX}📝 Original:" + Style.RESET_ALL)
            logger.info(f"{Fore.WHITE}{prompt.strip()}" + Style.RESET_ALL)
            logger.info(f"\n{Fore.LIGHTWHITE_EX}✨ Transformed (style: {self.style_name}" + 
            (f" ➡️ audience: {self.audience_name}" if self.audience_name else "") + "):" + Style.RESET_ALL)
            logger.info(f"{Fore.CYAN}{styled_prompt.strip()}\n" + Style.RESET_ALL)
            logger.info(Fore.LIGHTGREEN_EX + f"🎉 Transformation completed successfully!" + Style.RESET_ALL)

            return styled_prompt.strip()
        
        except Exception as e:
            logger.error(Fore.RED + f"❌ Transformation failed: {str(e)}" + Style.RESET_ALL)
            raise
