"""
Educators Audience Adapter Module - vibeprompt.audiences.educators

This module provides an educator-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for teachers, instructors, or trainers,
while preserving the original stylistic tone and creative structure.

The core functionality is exposed through the `EducatorsAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that emphasize pedagogical strategies,
curriculum development, student engagement, and learning outcomes, without altering the creative
integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.educators import EducatorsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = EducatorsAudience()
    >>> prompt = (
    ...     "Explain the concept of critical thinking with a clear, inspiring tone."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Explain the concept of critical thinking for educators with a clear, inspiring tone.
    Focus on practical pedagogical approaches to foster it in students, integrate it into
    curriculum design, assess its development, and highlight its importance for lifelong
    learning and civic engagement.'

Classes:
    - EducatorsAudience: Adapts styled prompts for an educator audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to an educator-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("educators")
class EducatorsAudience:
    """
    Adapt content for an educator audience.

    This adapter modifies the input prompt to suit an audience of teachers, instructors, or trainers,
    emphasizing elements such as:
        - Pedagogical methods, teaching strategies, and classroom management
        - Curriculum development, learning objectives, and assessment techniques
        - Student engagement, motivation, and differentiated instruction
        - Educational technology and innovative teaching tools
        - Professional development and best practices in education

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns an educator-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply educator audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for educators.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.educators import EducatorsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = EducatorsAudience()
            >>> adapter.apply("Discuss the challenges of remote learning with a thoughtful, empathetic perspective.", llm)
            'Discuss the challenges of remote learning for educators with a thoughtful, empathetic perspective.
            Focus on pedagogical strategies to overcome these challenges, effective student engagement techniques
            in virtual environments, assessment adaptations, and professional development opportunities for instructors,
            while maintaining the thoughtful and empathetic tone.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for educators (teachers, instructors, trainers), while PRESERVING
        the original style and tone as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add educator-focused elements

        Your goal is to make the styled prompt relevant and actionable for educators
        while keeping its unique character. Add educator-focused elements like:
        - Pedagogical strategies, teaching methods, and classroom application
        - Curriculum integration, learning objectives, and assessment techniques
        - Student engagement, motivation, and differentiated instruction approaches
        - Use of educational technology and innovative teaching tools
        - Professional development needs and educational best practices
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Present the wonders of the human brain with an awe-inspiring, intricate narrative."

        Adapted for educators (preserving style):
        "Present the wonders of the human brain for educators with an awe-inspiring, intricate narrative.
        Focus on how to effectively teach its complex functions, design curriculum units that engage students
        in neuroscience, develop innovative activities for understanding cognitive processes, and assess
        student comprehension of its wonders, all while maintaining the awe-inspiring and intricate storytelling."

        ====
        Original text: {text}

        Adapted prompt for educators (preserving original style):
        """

        return llm.invoke(prompt).content
