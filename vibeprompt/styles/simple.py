"""
Simple Style Transformation Module - vibeprompt.styles.simple

This module provides a simple writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with plain, easy-to-understand language suited for beginners or non-experts.

The `SimpleStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage short, clear sentences, eliminate jargon, and promote educational accessibility—
all without changing the original question or topic.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.simple import SimpleStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = SimpleStyle()
    >>> prompt = "Explain how photosynthesis works"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain how photosynthesis works in a simple and easy way. Use basic words and short sentences
    so that someone without science background can understand.'

Classes:
    - SimpleStyle: Adapts prompts to guide the LLM toward clear, beginner-friendly, and jargon-free explanations.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in a simple, plain, and easy-to-understand tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("simple")
class SimpleStyle:
    """
    Transform text to a simple and easy-to-understand tone.

    This style adapter rewrites a prompt so that it encourages the language model
    to respond using basic, beginner-friendly language, clear structure, and minimal complexity—
    ideal for educational or non-technical audiences.

    Adaptation techniques include:
        - Using plain, everyday vocabulary
        - Short, declarative sentence structure
        - Avoiding technical terms, acronyms, or jargon
        - Explaining concepts in intuitive, accessible ways

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to guide the LLM
            to respond in a simple, clear, and easily digestible tone.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply simple style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a beginner-friendly tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit clear, plain, and easily understood responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.simple import SimpleStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = SimpleStyle()
            >>> style.apply("What is machine learning?", llm)
            'Explain what machine learning is in a simple and clear way. Use short sentences and basic words
            so someone without a tech background can easily understand.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to respond 
        in a simple, easy-to-understand tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - clear and plain language,
        - short and simple sentences,
        - no complex words or jargon,
        - beginner-friendly explanations,
        - and an educational tone that makes things easy to grasp.

        For example:
        Original text:
        Explain how photosynthesis works

        Rewritten prompt:
        Explain how photosynthesis works in a simple and easy way. Use 
        basic words and short sentences so that someone without science 
        background can understand.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
