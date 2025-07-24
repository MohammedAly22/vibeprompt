"""
Experts Audience Adapter Module - vibeprompt.audiences.experts

This module provides an expert-level audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version suitable for domain specialists and professionals,
while preserving the original creative style and tone.

The `ExpertsAudience` class integrates with LangChain-compatible LLMs and rewrites the given prompt
with a focus on intellectual rigor and technical precision. It adds expert-level content—such as advanced
terminology, research-backed frameworks, and nuanced analysis—without removing the original narrative flair.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.experts import ExpertsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = ExpertsAudience()
    >>> prompt = (
    ...     "Craft a poetic explanation of quantum computing that flows like a delicate symphony "
    ...     "of probability and logic intertwined."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Craft a poetic explanation of quantum computing using metaphors grounded in quantum field theory...'

Classes:
    - ExpertsAudience: Adapts styled prompts for specialized expert audiences while preserving the original tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for experts, maintaining the original tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("experts")
class ExpertsAudience:
    """
    Adapt content for domain experts and professionals.

    This adapter modifies a prompt to introduce deeper intellectual and technical rigor
    for expert audiences, while preserving its original stylistic tone and creativity.

    Adaptation techniques may include:
        - Inserting discipline-specific terminology and methodologies
        - Adding references to recent studies, whitepapers, or research trends
        - Enhancing analytical depth and theoretical precision
        - Maintaining metaphor, tone, and creative formatting

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns an expert-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply expert audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for an expert audience.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for domain specialists or experts,
                   while preserving the original creative tone and structure.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.experts import ExpertsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = ExpertsAudience()
            >>> adapter.apply("Describe blockchain as if narrating an epic saga.", llm)
            'Describe blockchain as if narrating an epic saga, integrating cryptographic consensus mechanisms...'
        """
        
        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for domain experts and specialists, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add expert-level depth and precision
        
        Your goal is to make the styled prompt intellectually rigorous for experts while 
        keeping its unique character. Add expert-focused elements like:
        - Advanced technical terminology and precise definitions
        - Cutting-edge research and theoretical frameworks
        - Nuanced analysis and sophisticated reasoning
        - Industry-specific methodologies and standards
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of quantum computing using sophisticated 
        metaphors and poetic language that flows like a gentle stream of knowledge."

        Adapted for experts (preserving style):
        "Craft an elegant explanation of quantum computing using sophisticated 
        quantum mechanical metaphors and theoretical language that flows like 
        rigorous mathematical proof. Incorporate advanced concepts like quantum 
        entanglement, superposition principles, decoherence effects, and algorithmic 
        complexity while maintaining the poetic elegance and intellectual depth."

        ====
        Original text: {text}

        Adapted prompt for experts (preserving original style):"""

        return llm.invoke(prompt).content
