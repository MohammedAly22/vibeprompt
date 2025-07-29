"""
Minimalist Style Transformation Module - vibeprompt.styles.minimalist

This module provides a minimalist writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with conciseness, clarity, and an emphasis on essential information.

The MinimalistStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that are direct, free of jargon, and
focus purely on conveying core ideas efficiently, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.minimalist import MinimalistStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = MinimalistStyle()
    >>> prompt = "Explain the process of photosynthesis in detail."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Concisely explain the core process of photosynthesis. Use minimal words,
    focusing only on essential steps and components, ensuring maximum clarity
    with no extraneous details.'

Classes:
- MinimalistStyle: Adapts prompts to guide the LLM toward a concise and essential writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are brief, direct,
and focused on core information.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("minimalist")
class MinimalistStyle:
    """
    Transform text to a minimalist tone.

    This style adapter rewrites the prompt to encourage concise, direct,
    and essential responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Removing all superfluous words and phrases.
        - Focusing only on core facts and concepts.
        - Using clear, simple, and unambiguous language.
        - Prioritizing brevity and directness.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit minimalist,
            concise, and essential LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply minimalist style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a minimalist tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit brief, direct,
              and core-information-focused responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.minimalist import MinimalistStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = MinimalistStyle()
            >>> style.apply("What is the capital of France and why is it important?", llm)
            'State the capital of France. Provide only essential information,
            excluding all non-critical details or elaborations on its importance.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in a minimalist, concise, and direct tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - extreme conciseness and brevity,
        - only essential information, removing all superfluous words,
        - clear, simple, and unambiguous language,
        - a focus on core facts without elaboration or unnecessary context.

        For example:
        Original text:
        Tell me everything about the history of the internet.

        Rewritten prompt:
        Briefly outline the key milestones in the history of the internet. Focus
        exclusively on essential developments, using minimal words and no decorative
        language or extensive background.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
