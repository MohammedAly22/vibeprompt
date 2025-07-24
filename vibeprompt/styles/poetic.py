"""
Poetic Style Transformation Module - vibeprompt.styles.poetic

This module provides a poetic writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with expressive, lyrical, and artistic language.

The `PoeticStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage elegant phrasing, rich imagery, and emotionally resonant responses without altering
the original topic or core information.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.poetic import PoeticStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = PoeticStyle()
    >>> prompt = "Describe how the moon affects the tides"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Describe how the moon influences the tides using a poetic tone. Embrace vivid imagery,
    flowing rhythm, and a lyrical voice that captures the elegance and mystery of nature.'

Classes:
    - PoeticStyle: Adapts prompts to guide the LLM toward a lyrical, artistic, and emotionally evocative writing style.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in a poetic, expressive, and beautifully phrased tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("poetic")
class PoeticStyle:
    """
    Transform text to a poetic tone.

    This style adapter rewrites a prompt so that it encourages the language model
    to respond with lyrical elegance, rich metaphors, and a reflective tone—without
    altering the core meaning of the prompt.

    Adaptation techniques include:
        - Incorporating poetic devices such as alliteration, personification, and rhythm
        - Using vivid imagery and emotional language
        - Evoking a sense of wonder, beauty, or depth
        - Crafting responses that feel like artful expressions

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to guide the LLM
            to respond in an artistic, emotional, and poetic voice.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply poetic style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a poetic tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit lyrical, expressive, and emotionally resonant responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.poetic import PoeticStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = PoeticStyle()
            >>> style.apply("What causes lightning?", llm)
            'Explain what causes lightning in a poetic tone. Use vivid, expressive language and
            artistic metaphors to capture the awe and intensity of nature’s electric dance.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a poetic tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - lyrical and expressive language,
        - rich imagery and metaphor,
        - rhythm, flow, and poetic devices (e.g., alliteration, personification),
        - a tone that evokes emotion, wonder, or reflection,
        - and a sense of artistry in phrasing.

        For example:
        Original text:
        Describe how the moon affects the tides

        Rewritten prompt:
        Describe how the moon influences the tides using a poetic tone. 
        Embrace vivid imagery, flowing rhythm, and a lyrical voice that 
        captures the elegance and mystery of nature.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
