"""
Humorous Style Transformation Module - vibeprompt.styles.humorous

This module provides a humorous writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with creativity, wit, and a light-hearted tone.

The `HumorousStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage playful, funny, and clever responses without changing the original
intent or core information.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.humorous import HumorousStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = HumorousStyle()
    >>> prompt = "Describe how gravity works"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain how gravity works in a humorous tone. Use playful language, witty analogies, and inject some scientific comedy to make it entertaining and engaging.'

Classes:
    - HumorousStyle: Adapts prompts to guide the LLM toward a witty, light-hearted, and entertaining writing style.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in a humorous, clever, and engaging tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("humorous")
class HumorousStyle:
    """
    Transform text to a humorous tone.

    This style adapter rewrites a prompt so that it instructs the language model
    to respond with wit, charm, and an amusing spin on the subject—while preserving
    the original focus of the request.

    Adaptation techniques include:
        - Adding jokes, puns, or clever exaggerations
        - Using light, playful language and metaphors
        - Maintaining the subject while making it engaging and funny
        - Injecting a dose of comedic flair without distorting core meaning

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to encourage a
            witty, comedic, and entertaining response from the LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply humorous style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a humorous tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit playful, funny, and engaging responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.humorous import HumorousStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = HumorousStyle()
            >>> style.apply("What is quantum computing?", llm)
            'Explain quantum computing in a humorous tone. Use playful analogies, puns, and light-hearted metaphors to make it engaging and funny.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a humorous tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - light-hearted and playful language,
        - witty or clever phrasing,
        - jokes, puns, or exaggerated metaphors (when appropriate),
        - and an overall entertaining tone that makes the reader smile.

        For example:
        Original text:
        Describe how gravity works

        Rewritten prompt:
        Explain how gravity works in a humorous tone. Use playful language, 
        witty analogies, and inject a bit of scientific comedy to make the 
        explanation entertaining and engaging.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
