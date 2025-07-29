"""
Seniors Audience Adapter Module - vibeprompt.audiences.seniors

This module provides a senior-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for an older adult audience,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `SeniorsAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that resonate with their life experiences,
priorities, and potential concerns such as health, legacy, wisdom, and community, without
altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.seniors import SeniorsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = SeniorsAudience()
    >>> prompt = (
    ...     "Reflect on the evolution of technology with a nostalgic, thoughtful tone."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Reflect on the evolution of technology for a senior audience with a nostalgic, thoughtful tone.
    Connect it to personal milestones, how it has transformed daily life over decades, and its impact
    on family connections and community, inviting contemplation on a lifetime of change while maintaining
    the nostalgic tone.'

Classes:
    - SeniorsAudience: Adapts styled prompts for a senior audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a senior-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("seniors")
class SeniorsAudience:
    """
    Adapt content for a senior audience.

    This adapter modifies the input prompt to suit a senior audience, emphasizing elements such as:
        - Connection to life experience and wisdom gained over time
        - Practical relevance to health, well-being, retirement, or leisure
        - Focus on legacy, family, community, and personal fulfillment
        - Clear, respectful, and thoughtful language
        - Acknowledgment of potentially different technological comfort levels
        - Opportunities for reflection and sharing of experiences

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a senior-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply senior audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for seniors.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.seniors import SeniorsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = SeniorsAudience()
            >>> adapter.apply("Discuss the benefits of lifelong learning with an inspiring, gentle tone.", llm)
            'Discuss the immense benefits of lifelong learning for a senior audience with an inspiring, gentle tone.
            Connect it to maintaining cognitive vitality, discovering new passions in retirement, enriching social
            connections, and contributing wisdom to the community, making it deeply relevant to their stage of life
            while maintaining the inspiring tone.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for seniors, while PRESERVING the original style and tone
        as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add senior-focused elements

        Your goal is to make the styled prompt relevant and resonate with the concerns and experiences of seniors
        while keeping its unique character. Add senior-focused elements like:
        - Connection to accumulated life experiences and wisdom
        - Practical implications for health, well-being, retirement, and leisure activities
        - Broader themes of legacy, family, community, and personal fulfillment
        - Respectful, clear, and unhurried language
        - Consideration of accessibility and varying levels of technological familiarity
        - Opportunities for reflection on historical changes or personal journeys
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Discuss the future of artificial intelligence with a curious, optimistic tone."

        Adapted for seniors (preserving style):
        "Discuss the future of artificial intelligence for a senior audience with a curious, optimistic tone.
        Frame it in terms of how AI might enhance quality of life, assist with health and daily living, foster
        intergenerational connections, and preserve legacies, making it personally relevant while maintaining
        the curious and optimistic outlook."

        ====
        Original text: {text}

        Adapted prompt for seniors (preserving original style):
        """

        return llm.invoke(prompt).content
