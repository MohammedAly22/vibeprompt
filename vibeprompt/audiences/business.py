"""
Business Audience Adapter Module - vibeprompt.audiences.business

This module provides a business-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for business and professional audiences,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `BusinessAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that emphasize business relevance such as
ROI, strategic impact, and value proposition, without altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.business import BusinessAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = BusinessAudience()
    >>> prompt = (
    ...     "Craft an elegant explanation of artificial intelligence using "
    ...     "sophisticated metaphors and poetic language that flows like a gentle stream of knowledge."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Craft an elegant explanation of artificial intelligence using sophisticated business metaphors...'

Classes:
    - BusinessAudience: Adapts styled prompts for a business-oriented audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a business context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("business")
class BusinessAudience:
    """
    Adapt content for a business/professional audience.

    This adapter modifies the input prompt to suit a professional audience, emphasizing elements such as:
        - Strategic relevance
        - ROI and value proposition
        - Competitive advantage
        - Operational metrics
        - Market analysis

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a business-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply business audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for business professionals.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.business import BusinessAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = BusinessAudience()
            >>> adapter.apply("Explain quantum computing like you're writing a sonnet.", llm)
            'Explain quantum computing like you're writing a strategic investment sonnet...'
        """

        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for business professionals, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add business-focused elements
        
        Your goal is to make the styled prompt relevant to business context while 
        keeping its unique character. Add business-focused elements like:
        - ROI and value proposition considerations
        - Strategic implications and competitive advantages
        - Cost-benefit analysis and market impact
        - Business metrics and performance indicators
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of artificial intelligence using sophisticated 
        metaphors and poetic language that flows like a gentle stream of knowledge."

        Adapted for business (preserving style):
        "Craft an elegant explanation of artificial intelligence using sophisticated 
        business metaphors and strategic language that flows like a compelling investment 
        case. Frame AI as a transformative business asset, highlighting ROI potential, 
        competitive advantages, market disruption opportunities, and operational efficiency 
        gains while maintaining the poetic elegance of the explanation."

        ====
        Original text: {text}

        Adapted prompt for business professionals (preserving original style):
        """

        return llm.invoke(prompt).content
