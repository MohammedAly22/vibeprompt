"""
Registry System Module - vibeprompt.core.style_registry

This module provides a centralized and extensible registration system
for writing styles and target audiences used in the `vibeprompt` package.

It maintains internal registries for:
    - Prompt transformation styles (e.g., simple, technical, poetic)
    - Target audiences (e.g., developers, children, experts)

These registries enable dynamic discovery and instantiation of transformation
adapters by name. It uses decorators (`@register_style` and `@register_audience`)
to associate unique identifiers with transformation classes and provides lazy loading
to ensure modules are only imported when needed.

Usage:
    >>> from vibeprompt.core.style_registry import load_style, get_styles
    >>> style = load_style("simple")
    >>> style.apply("What is gravity?", llm)

Functions:
    - register_style(name: str) -> Callable:
        Decorator to register a style transformation class.

    - register_audience(name: str) -> Callable:
        Decorator to register an audience transformation class.

    - get_styles() -> List[str]:
        Returns a list of all registered style names.

    - get_audiences() -> List[str]:
        Returns a list of all registered audience names.

    - load_style(name: str) -> Any:
        Loads and returns an instance of a registered style class.

    - load_audience(name: Optional[str]) -> Optional[Any]:
        Loads and returns an instance of a registered audience class.
"""

import logging
from colorama import Fore, Style
from typing import Type, Dict, Optional, Any


# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Global registries
_STYLE_REGISTRY: Dict[str, Type] = {}
_AUDIENCE_REGISTRY: Dict[str, Type] = {}


def register_style(name: str):
    """
    Decorator to register a style transformation class.

    This function associates a unique name with a prompt style class, allowing it
    to be later discovered and instantiated using `load_style()`.

    Args:
        - name (str): Unique identifier for the style (e.g., "simple", "technical").

    Returns:
        - Callable: A decorator that registers the class in the global style registry.

    Example:
        >>> @register_style("simple")
        >>> class SimpleStyle:
        >>>     ...
    """

    def decorator(cls):
        _STYLE_REGISTRY[name] = cls
        logger.debug(Fore.CYAN + f"💾 Registered style: {name}" + Style.RESET_ALL)
        return cls

    return decorator


def register_audience(name: str):
    """
    Decorator to register an audience transformation class.

    Associates a unique name with an audience adapter class that adjusts
    content for a specific target group (e.g., children, healthcare, experts).

    Args:
        - name (str): Unique identifier for the audience type.

    Returns:
        - Callable: A decorator that registers the class in the global audience registry.

    Example:
        >>> @register_audience("students")
        >>> class StudentAudience:
        >>>     ...
    """

    def decorator(cls):
        _AUDIENCE_REGISTRY[name] = cls
        logger.debug(Fore.CYAN + f"💾 Registered audience: {name}" + Style.RESET_ALL)
        return cls

    return decorator


def get_styles() -> list[str]:
    """
    Return a list of all available registered style names.

    Triggers lazy loading of modules if the registry is empty.

    Returns:
        - List[str]: A list of style names such as ["simple", "technical", "humorous"].
    """

    _ensure_modules_loaded()
    return list(_STYLE_REGISTRY.keys())


def get_audiences() -> list[str]:
    """
    Return a list of all available registered audience names.

    Triggers lazy loading of modules if the registry is empty.

    Returns:
        - List[str]: A list of audience identifiers such as ["children", "experts", "general"].
    """

    _ensure_modules_loaded()
    return list(_AUDIENCE_REGISTRY.keys())


def load_style(name: str) -> Any:
    """
    Load and instantiate a registered style class by name.

    Args:
        - name (str): Name of the style class to load.

    Returns:
        - Any: An instance of the registered style class.

    Raises:
        - ValueError: If the given name does not exist in the style registry.

    Example:
        >>> style = load_style("playful")
        >>> styled_prompt = style.apply("Explain black holes", llm)
    """

    _ensure_modules_loaded()
    if name not in _STYLE_REGISTRY:
        raise ValueError(
            Fore.RED + f"❌ Unknown style: {name}. Available: {list(_STYLE_REGISTRY.keys())}" + Style.RESET_ALL
        )
    return _STYLE_REGISTRY[name]()


def load_audience(name: Optional[str]) -> Optional[Any]:
    """
    Load and instantiate a registered audience class by name.

    Args:
        - name (Optional[str]): Name of the audience to load. If None, returns None.

    Returns:
        - Optional[Any]: An instance of the audience class, or None if no name provided.

    Raises:
        - ValueError: If the audience name is not registered.

    Example:
        >>> audience = load_audience("healthcare")
        >>> modified_prompt = audience.apply(prompt, llm)
    """

    if not name:
        return None
    _ensure_modules_loaded()
    if name not in _AUDIENCE_REGISTRY:
        raise ValueError(
            Fore.RED + f"❌ Unknown audience: {name}. Available: {list(_AUDIENCE_REGISTRY.keys())}" + Style.RESET_ALL
        )
    return _AUDIENCE_REGISTRY[name]()

def _ensure_modules_loaded():
    """
    Lazily import all style and audience modules to trigger registration.

    This ensures that all adapters are automatically registered without requiring
    explicit imports at runtime. It checks whether the internal registries are
    populated and performs a module-level import if necessary.
    """
    
    if not _STYLE_REGISTRY:
        # Import all modules to trigger registration
        import vibeprompt.styles.assertive
        import vibeprompt.styles.formal
        import vibeprompt.styles.humorous
        import vibeprompt.styles.playful
        import vibeprompt.styles.poetic
        import vibeprompt.styles.sequential
        import vibeprompt.styles.simple
        import vibeprompt.styles.technical

        import vibeprompt.audiences.business
        import vibeprompt.audiences.children
        import vibeprompt.audiences.developers
        import vibeprompt.audiences.experts
        import vibeprompt.audiences.general
        import vibeprompt.audiences.healthcare
        import vibeprompt.audiences.students
