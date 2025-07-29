"""
Intermediates Audience Adapter Module - vibeprompt.audiences.intermediates

This module provides an intermediate-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for individuals who have some foundational
knowledge of the subject matter but are not yet experts, while preserving the original stylistic tone
and creative structure.

The core functionality is exposed through the `IntermediatesAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that build upon existing knowledge, introduce
more detailed concepts, and offer deeper insights, without altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.intermediates import IntermediatesAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = IntermediatesAudience()
    >>> prompt = (
    ...     "Discuss the impact of climate change with a passionate, urgent call to action."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Discuss the multifaceted impact of climate change for an intermediate audience, assuming
    some prior knowledge, with a passionate, urgent call to action. Elaborate on key mechanisms,
    introduce nuanced effects, and provide actionable insights for those ready to delve deeper
    into solutions.'

Classes:
    - IntermediatesAudience: Adapts styled prompts for an intermediate audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to an intermediate-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("intermediates")
class IntermediatesAudience:
    """
    Adapt content for an intermediate audience.

    This adapter modifies the input prompt to suit an audience with some foundational knowledge,
    emphasizing elements such as:
        - Building upon existing basic understanding
        - Introduction of more specific or detailed concepts
        - Exploration of interconnectedness and deeper implications
        - Use of relevant terminology without excessive simplification
        - Examples that add nuance rather than just basic illustration

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns an intermediate-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply intermediate audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for intermediates.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.intermediates import IntermediatesAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = IntermediatesAudience()
            >>> adapter.apply("Explain the concept of AI ethics with a critical, investigative tone.", llm)
            'Explain the concept of AI ethics for an intermediate audience with a critical, investigative tone.
            Assume foundational knowledge of AI, delve into specific ethical dilemmas, explore different
            philosophical approaches, and encourage deeper analysis of complex issues, while maintaining
            the critical and investigative style.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for individuals with some foundational knowledge (intermediates),
        while PRESERVING the original style and tone as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add intermediate-focused elements

        Your goal is to make the styled prompt more insightful and detailed for an intermediate audience
        while keeping its unique character. Add intermediate-focused elements like:
        - Building upon basic understanding with more specific concepts
        - Introduction of relevant terminology that an intermediate would grasp
        - Exploration of nuances, interconnections, and deeper implications
        - Examples that provide further insight or illustrate complexities
        - Encouraging analytical thinking and deeper engagement with the topic
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Describe the beauty of classical music with an evocative, flowing prose."

        Adapted for intermediates (preserving style):
        "Describe the beauty of classical music for an intermediate audience with an evocative, flowing prose.
        Assume familiarity with basic musical concepts, delve into specific structural elements or historical
        contexts that contribute to its beauty, and explore how composers achieve their emotional impact
        through nuanced techniques, while maintaining the evocative and flowing prose."

        ====
        Original text: {text}

        Adapted prompt for intermediate professionals (preserving original style):
        """

        return llm.invoke(prompt).content
