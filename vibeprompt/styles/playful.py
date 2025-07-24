"""
Playful Style Transformation Module - vibeprompt.styles.playful

This module provides a playful writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with whimsy, curiosity, and light-hearted flair.

The `PlayfulStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage imaginative, cheerful, and childlike responses without altering the original
topic or informative content.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.playful import PlayfulStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = PlayfulStyle()
    >>> prompt = "How does the internet work?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain how the internet works in a playful tone. Use cheerful language, creative comparisons,
    and a fun, bouncy style that makes it feel like a magical adventure.'

Classes:
    - PlayfulStyle: Adapts prompts to guide the LLM toward a whimsical, creative, and delight-inducing writing style.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in a cheerful, curious, and playful tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("playful")
class PlayfulStyle:
    """
    Transform text to a playful tone.

    This style adapter rewrites a prompt so that it encourages the language model
    to respond with light-hearted imagination, wonder, and a fun-loving tone—without
    changing the original request's topic or learning objective.

    Adaptation techniques include:
        - Adding cheerful, whimsical phrasing and playful exaggerations
        - Using imaginative metaphors or curious questions
        - Presenting information in a bouncy, spirited voice
        - Making the explanation feel like a fun, engaging exploration

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to guide the LLM
            to respond with creativity, wonder, and a touch of childlike joy.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply playful style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a playful tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit curious, cheerful, and delightfully imaginative responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.playful import PlayfulStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = PlayfulStyle()
            >>> style.apply("What does a black hole do?", llm)
            'Explain what a black hole does in a playful tone. Use vivid, curious metaphors and fun language
            that makes it feel like a cosmic story full of imagination and mystery.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a playful tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - whimsical and lighthearted language,
        - imaginative or childlike phrasing,
        - a curious and cheerful voice,
        - occasional fun metaphors, silly exaggerations, or quirky analogies,
        - and a tone that sparks delight and wonder.

        For example:
        Original text:
        How does the internet work?

        Rewritten prompt:
        Explain how the internet works in a playful tone. Use cheerful language, 
        creative comparisons, and a fun, bouncy style that makes it feel like a 
        magical adventure.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
