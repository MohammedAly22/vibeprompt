"""
Teenagers Audience Adapter Module - vibeprompt.audiences.teenagers

This module provides a teenager-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for a teenage audience,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `TeenagersAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that are engaging, relatable,
and connect to topics relevant to their lives and interests, without altering the creative
integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.teenagers import TeenagersAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = TeenagersAudience()
    >>> prompt = (
    ...     "Explain the importance of financial literacy with a wise, guiding tone."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Explain the importance of financial literacy to teenagers with a wise, guiding tone.
    Connect it to their future independence, smart spending, and saving for goals like college
    or a first car, making it relevant to their stage of life while maintaining the wise tone.'

Classes:
    - TeenagersAudience: Adapts styled prompts for a teenage audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a teenager-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("teenagers")
class TeenagersAudience:
    """
    Adapt content for a teenage audience.

    This adapter modifies the input prompt to suit a teenage audience, emphasizing elements such as:
        - Relatability to their experiences, interests, and future
        - Engaging and dynamic language
        - Addressing topics relevant to school, social life, technology, or personal growth
        - Avoiding overly simplistic or condescending tones
        - Maintaining authenticity and a sense of understanding

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a teenager-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply teenager audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for teenagers.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.teenagers import TeenagersAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = TeenagersAudience()
            >>> adapter.apply("Discuss the history of rock music with a rebellious spirit.", llm)
            'Discuss the history of rock music for teenagers with a rebellious spirit.
            Connect its evolution to youth culture, social movements, and its impact on fashion
            and identity, making it resonate with their sense of independence and self-expression,
            while maintaining the rebellious tone.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for teenagers, while PRESERVING the original style and tone
        as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add teenager-focused elements

        Your goal is to make the styled prompt engaging and relevant to a teenage audience
        while keeping its unique character. Add teenager-focused elements like:
        - Connecting to their interests (e.g., social media, gaming, music, identity, future)
        - Using relatable scenarios or examples from their daily lives
        - Addressing issues they care about (e.g., fairness, independence, technology, environment)
        - Maintaining an authentic and non-condescending voice
        - Focusing on practical or personal relevance
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Explain the importance of cybersecurity with a dramatic, suspenseful narrative."

        Adapted for teenagers (preserving style):
        "Explain the critical importance of cybersecurity to teenagers with a dramatic, suspenseful narrative.
        Frame it as a thrilling defense against digital threats to their online presence, privacy, and gaming accounts,
        highlighting real-world consequences relevant to their digital lives, while maintaining the dramatic and
        suspenseful storytelling."

        ====
        Original text: {text}

        Adapted prompt for teenagers (preserving original style):
        """

        return llm.invoke(prompt).content
