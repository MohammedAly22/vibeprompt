"""
Researchers Audience Adapter Module - vibeprompt.audiences.researchers

This module provides a researcher-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for academic or scientific researchers,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `ResearchersAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that emphasize methodological rigor,
data-driven insights, theoretical frameworks, and implications for future study, without
altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.researchers import ResearchersAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = ResearchersAudience()
    >>> prompt = (
    ...     "Analyze the socio-economic effects of urbanization with a critical, academic tone."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Analyze the socio-economic effects of urbanization for researchers with a critical, academic tone.
    Discuss methodological approaches, potential biases in data collection, theoretical underpinnings,
    and implications for further empirical studies and policy recommendations within the academic discourse.'

Classes:
    - ResearchersAudience: Adapts styled prompts for a researcher audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a researcher-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("researchers")
class ResearchersAudience:
    """
    Adapt content for a researcher audience.

    This adapter modifies the input prompt to suit an academic or scientific research audience,
    emphasizing elements such as:
        - Methodological considerations and research design
        - Data analysis, statistical significance, and evidence-based conclusions
        - Theoretical frameworks, hypotheses, and peer-reviewed literature
        - Gaps in current knowledge and directions for future research
        - Objectivity, rigor, and scholarly discourse

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a researcher-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply researcher audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for researchers.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.researchers import ResearchersAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = ResearchersAudience()
            >>> adapter.apply("Explore the psychological effects of social media with a deeply empathetic narrative.", llm)
            'Explore the psychological effects of social media for researchers with a deeply empathetic narrative.
            Discuss potential research methodologies for qualitative and quantitative studies, highlight
            existing theoretical models, identify areas for further empirical investigation, and consider
            the ethical implications for human subjects, all while maintaining the empathetic narrative style.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for academic or scientific researchers, while PRESERVING
        the original style and tone as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add researcher-focused elements

        Your goal is to make the styled prompt relevant and rigorous for a research audience
        while keeping its unique character. Add researcher-focused elements like:
        - Emphasis on empirical evidence, data, and methodologies
        - Discussion of theoretical frameworks, hypotheses, and scholarly debates
        - Identification of research gaps, limitations, and future directions for study
        - Consideration of statistical significance, validity, and reliability
        - Relevance to peer-reviewed literature and academic discourse
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Present an imaginative vision for sustainable urban living with a bold, futuristic tone."

        Adapted for researchers (preserving style):
        "Present an imaginative vision for sustainable urban living for researchers with a bold, futuristic tone.
        Propose potential research questions, outline interdisciplinary methodologies for studying its feasibility
        and impact, discuss key performance indicators for evaluation, and identify areas requiring novel scientific
        or technological breakthroughs, all within the context of a visionary and futuristic presentation."

        ====
        Original text: {text}

        Adapted prompt for researchers (preserving original style):
        """

        return llm.invoke(prompt).content
