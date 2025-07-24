"""
Formal Style Transformation Module - vibeprompt.styles.formal

This module provides a formal writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with professionalism, clarity, and politeness.

The `FormalStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage respectful, polished, and structured responses without altering the original
intent or conceptual content.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.formal import FormalStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = FormalStyle()
    >>> prompt = "Tell me how to fix a bug in my code"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Please explain how to resolve a software error using a formal and professional tone.
    Avoid informal expressions and ensure clarity in explanation.'

Classes:
    - FormalStyle: Adapts prompts to guide the LLM toward a professional and respectful writing style.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in a formal, structured, and courteous tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("formal")
class FormalStyle:
    """
    Transform text to a formal tone.

    This style adapter rewrites a prompt so that it guides the language model
    to respond in a polished, professional, and respectful manner, while retaining
    the original subject and purpose.

    Adaptation techniques include:
        - Replacing casual expressions with formal phrasing
        - Avoiding contractions, slang, and colloquial speech
        - Encouraging polite and structured language
        - Maintaining conceptual clarity and respectful tone

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt that prompts the LLM
            to reply with formality, professionalism, and decorum.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply formal style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a formal tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit structured, polite, and professional responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.formal import FormalStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = FormalStyle()
            >>> style.apply("How do I start a tech blog?", llm)
            'Please provide guidance on how to establish a technology-focused blog using formal and professional language.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a formal tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - formal and professional language,
        - polite and respectful tone,
        - no slang or colloquialisms,
        - clear and structured phrasing,
        - and avoidance of contractions.

        For example:
        Original text:
        Tell me how to fix a bug in my code

        Rewritten prompt:
        Please explain how to resolve an error in a software program using a formal 
        and professional tone. Avoid informal expressions and use clear, respectful language.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
