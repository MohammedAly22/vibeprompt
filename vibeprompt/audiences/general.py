"""
General Audience Adapter Module - vibeprompt.audiences.general

This module provides a general-level audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version suitable for a broad, non-specialist audience—
such as curious adults and general readers—while preserving the original creative tone and structure.

The `GeneralAudience` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to ensure accessibility, relatability, and broad engagement without sacrificing intellectual or stylistic value.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.general import GeneralAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = GeneralAudience()
    >>> prompt = (
    ...     "Describe the blockchain as a symphony of trustless cooperation, "
    ...     "with notes of cryptography and rhythm of decentralization."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Describe the blockchain as a symphony of cooperation, using familiar analogies like shared digital ledgers...'

Classes:
    - GeneralAudience: Adapts styled prompts for a broad general audience, preserving tone and creativity.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for the general public, while maintaining style.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("general")
class GeneralAudience:
    """
    Adapt content for a general, non-specialist audience.

    This adapter modifies a prompt to make it accessible and engaging for curious adults
    without technical expertise, while preserving its original stylistic tone and creativity.

    Adaptation techniques may include:
        - Replacing niche jargon with clear, widely understood language
        - Using everyday analogies and relatable metaphors
        - Presenting ideas with clarity, simplicity, and universal appeal
        - Retaining original creative tone, metaphors, and structure

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a general-audience version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply general audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for a general adult audience.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for general readers or non-experts,
                   while preserving the original creative tone and formatting.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.general import GeneralAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = GeneralAudience()
            >>> adapter.apply("Explain relativity as a dreamlike dance of space and time.", llm)
            'Explain relativity as a dreamlike dance of space and time, using accessible imagery like moving trains and clocks...'
        """

        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for a general adult audience, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to ensure broad accessibility and engagement
        
        Your goal is to make the styled prompt accessible to educated adults while 
        keeping its unique character. Add general-audience elements like:
        - Clear, accessible language without oversimplification
        - Relatable everyday analogies and common references
        - Balanced depth that's informative but not overwhelming
        - Universal examples and widely understood concepts
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of blockchain technology using sophisticated 
        cryptographic metaphors and technical language that flows like precise algorithms."

        Adapted for general audience (preserving style):
        "Craft an elegant explanation of blockchain technology using sophisticated 
        but accessible metaphors and clear language that flows like a well-told story. 
        Compare blockchain to familiar concepts like digital ledgers or chain links, 
        use everyday examples people can relate to, while maintaining the technical 
        elegance and making it engaging for curious adults."

        ====
        Original text: {text}

        Adapted prompt for general audience (preserving original style):"""

        return llm.invoke(prompt).content
