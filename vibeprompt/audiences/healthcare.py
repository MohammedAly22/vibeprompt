"""
Healthcare Audience Adapter Module - vibeprompt.audiences.healthcare

This module provides a healthcare-focused audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version suitable for healthcare professionals—
such as clinicians, researchers, and medical educators—while preserving the original
creative tone and structure.

The `HealthcareAudience` class integrates with LangChain-compatible LLMs and rewrites the
given prompt to introduce medical relevance, clinical context, and evidence-based considerations
without losing the original narrative or stylistic flair.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.healthcare import HealthcareAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = HealthcareAudience()
    >>> prompt = (
    ...     "Describe artificial intelligence as if narrating a healing journey guided by logic and empathy."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Describe artificial intelligence as if narrating a clinical journey guided by evidence-based decision support...'

Classes:
    - HealthcareAudience: Adapts styled prompts for a medical audience, preserving tone and adding clinical relevance.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for healthcare professionals, while maintaining style.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("healthcare")
class HealthcareAudience:
    """
    Adapt content for a healthcare professional audience.

    This adapter modifies a stylized prompt to introduce medically relevant concepts and clinical context
    for audiences such as physicians, nurses, medical researchers, and health policy professionals—
    all while preserving the original stylistic tone and creativity.

    Adaptation techniques may include:
        - Reframing language using clinical terminology and patient-centered concepts
        - Including diagnostic, therapeutic, and procedural relevance
        - Referencing medical guidelines, protocols, or evidence-based practices
        - Maintaining original metaphors and tone while ensuring clinical applicability

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a healthcare-adapted version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply healthcare audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for a healthcare professional audience.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for healthcare professionals and medical audiences,
                   while preserving the original creative tone and structure.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.healthcare import HealthcareAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = HealthcareAudience()
            >>> adapter.apply("Describe machine learning as a poetic dance of data.", llm)
            'Describe machine learning as a poetic dance of data, framed as a clinical decision support journey...'
        """

        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for healthcare professionals, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add healthcare-focused elements
        
        Your goal is to make the styled prompt medically relevant for healthcare workers while 
        keeping its unique character. Add healthcare-focused elements like:
        - Clinical applications and patient care implications
        - Evidence-based practices and medical protocols
        - Safety considerations and regulatory compliance
        - Diagnostic and therapeutic relevance
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of machine learning using sophisticated 
        metaphors and poetic language that flows like a gentle stream of knowledge."

        Adapted for healthcare (preserving style):
        "Craft an elegant explanation of machine learning using sophisticated 
        medical metaphors and clinical language that flows like a diagnostic journey. 
        Frame ML as a diagnostic tool, highlight patient care applications, discuss 
        clinical validation requirements, regulatory considerations, and therapeutic 
        potential while maintaining the poetic elegance and medical relevance."

        ====
        Original text: {text}

        Adapted prompt for healthcare professionals (preserving original style):"""

        return llm.invoke(prompt).content
