"""
Adults Audience Adapter Module - vibeprompt.audiences.adults

This module provides an adult-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for an adult audience,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `AdultsAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that consider their diverse life experiences,
responsibilities, and broader societal awareness, without altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.adults import AdultsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = AdultsAudience()
    >>> prompt = (
    ...     "Reflect on the concept of 'time' with a philosophical, introspective tone."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Reflect on the multifaceted concept of "time" for an adult audience with a philosophical, introspective tone.
    Consider its implications for long-term planning, personal growth, societal evolution, and the balance
    of responsibilities, prompting deeper thought on how it shapes adult lives while maintaining the philosophical tone.'

Classes:
    - AdultsAudience: Adapts styled prompts for an adult audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to an adult-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("adults")
class AdultsAudience:
    """
    Adapt content for an adult audience.

    This adapter modifies the input prompt to suit an adult audience, emphasizing elements such as:
        - Practical implications for daily life, career, and family
        - Broader societal or global relevance
        - Consideration of long-term consequences and planning
        - Nuanced perspectives on complex issues
        - Respect for diverse life experiences and responsibilities

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns an adult-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply adult audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for adults.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.adults import AdultsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = AdultsAudience()
            >>> adapter.apply("Explore the concept of happiness with a whimsical, reflective approach.", llm)
            'Explore the concept of happiness for an adult audience with a whimsical, reflective approach.
            Connect it to the complexities of adult responsibilities, work-life balance, family dynamics,
            and the pursuit of personal fulfillment in a mature context, while maintaining the whimsical
            and reflective style.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for adults, while PRESERVING the original style and tone
        as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add adult-focused elements

        Your goal is to make the styled prompt relevant and resonate with adult concerns and experiences
        while keeping its unique character. Add adult-focused elements like:
        - Practical implications for career, family, finances, and personal well-being
        - Broader societal, ethical, or global contexts
        - Consideration of long-term planning, responsibilities, and life choices
        - Nuanced perspectives on complex real-world issues
        - Acknowledgment of diverse life experiences and priorities
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Discuss the future of renewable energy with an optimistic, visionary outlook."

        Adapted for adults (preserving style):
        "Discuss the future of renewable energy for an adult audience with an optimistic, visionary outlook.
        Frame it in terms of economic impact, policy implications, job creation, and sustainable living
        for families and communities, highlighting tangible benefits and challenges relevant to adult
        decision-making, while maintaining the optimistic and visionary tone."

        ====
        Original text: {text}

        Adapted prompt for adults (preserving original style):
        """

        return llm.invoke(prompt).content
