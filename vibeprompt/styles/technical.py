"""
Technical Style Transformation Module - vibeprompt.styles.technical

This module provides a technical writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with a formal, expert-level, and domain-specific tone suited for technical or professional audiences.

The `TechnicalStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to elicit responses that include precise terminology, structured reasoning, and scientific or technical clarity—
all without changing the original question or intent.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.technical import TechnicalStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = TechnicalStyle()
    >>> prompt = "How does a neural network learn?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Provide a detailed technical explanation of how a neural network learns. Use accurate
    machine learning terminology, describe the backpropagation process, and explain the role
    of gradient descent in updating weights.'

Classes:
    - TechnicalStyle: Adapts prompts to guide the LLM toward formal, domain-specific, and technically accurate explanations.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses that demonstrate technical expertise and precision.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("technical")
class TechnicalStyle:
    """
    Transform text to a technical, expert-level tone.

    This style adapter rewrites a prompt so that it encourages the language model
    to respond using domain-accurate terminology, formal tone, and structured explanations—
    ideal for engineers, scientists, analysts, or professional readers.

    Adaptation techniques include:
        - Using precise technical vocabulary
        - Maintaining a formal and objective tone
        - Describing processes and methods in detail
        - Including formulas, units, or system-specific references

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to guide the LLM
            to respond in a precise, structured, and domain-informed manner.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply technical style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a technical, expert tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit responses that are formal, structured, and domain-specific.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.technical import TechnicalStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = TechnicalStyle()
            >>> style.apply("How does a neural network learn?", llm)
            'Provide a detailed technical explanation of how a neural network learns. Use accurate
            machine learning terminology, describe the backpropagation process, and explain the role
            of gradient descent in updating weights.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to respond 
        in a technical tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - accurate, precise, and domain-specific terminology,
        - a formal and objective tone,
        - structured explanation of methods or calculations,
        - appropriate use of symbols, units, or formulas when relevant,
        - and a tone suited for professionals or technical readers.

        For example:
        Original text:
        How does a neural network learn?

        Rewritten prompt:
        Provide a detailed technical explanation of how a neural network 
        learns. Use accurate machine learning terminology, describe the 
        backpropagation process, and explain the role of gradient descent 
        in updating weights.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
