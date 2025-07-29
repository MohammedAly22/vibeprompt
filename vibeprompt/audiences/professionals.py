"""
Professionals Audience Adapter Module - vibeprompt.audiences.professionals

This module provides a professional-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for professionals in specific or general fields,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `ProfessionalsAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that emphasize practical application, industry
relevance, career development, and actionable insights, without altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.professionals import ProfessionalsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = ProfessionalsAudience()
    >>> prompt = (
    ...     "Explain the new economic policy with a clear, analytical perspective."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Explain the new economic policy for professionals with a clear, analytical perspective.
    Focus on its practical implications for various industries, potential market shifts, regulatory changes,
    and strategic adjustments required for businesses and career paths, providing actionable insights for their work.'

Classes:
    - ProfessionalsAudience: Adapts styled prompts for a professional audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a professional-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("professionals")
class ProfessionalsAudience:
    """
    Adapt content for a professional audience.

    This adapter modifies the input prompt to suit a professional audience, emphasizing elements such as:
        - Practical application and actionable insights
        - Industry-specific relevance and trends
        - Career growth, skill development, and professional challenges
        - Efficient use of time and resources
        - Strategic decision-making and problem-solving within a professional context

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a professional-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply professional audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for professionals.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.professionals import ProfessionalsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = ProfessionalsAudience()
            >>> adapter.apply("Discuss the future of work with an innovative, forward-looking tone.", llm)
            'Discuss the future of work for professionals with an innovative, forward-looking tone.
            Analyze emerging technologies, new skill requirements, organizational restructuring,
            and strategic adaptations for career progression and business resilience in the evolving
            professional landscape, while maintaining the innovative and forward-looking tone.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for professionals, while PRESERVING the original style and tone
        as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add professional-focused elements

        Your goal is to make the styled prompt relevant and actionable for professionals
        while keeping its unique character. Add professional-focused elements like:
        - Practical application within their industry or role
        - Impact on career development, skills, and opportunities
        - Strategic implications for business operations, efficiency, or growth
        - Considerations for decision-making, problem-solving, or innovation
        - Relevance to industry trends, best practices, or challenges
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Explain the art of public speaking with an inspiring, empowering narrative."

        Adapted for professionals (preserving style):
        "Explain the art of public speaking for professionals with an inspiring, empowering narrative.
        Focus on its strategic importance for career advancement, leadership communication, client engagement,
        and effective team presentations, providing actionable techniques for professional impact while
        maintaining the inspiring and empowering storytelling."

        ====
        Original text: {text}

        Adapted prompt for professionals (preserving original style):
        """

        return llm.invoke(prompt).content
