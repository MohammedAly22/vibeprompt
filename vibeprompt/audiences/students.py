"""
Students Audience Adapter Module - vibeprompt.audiences.students

This module provides a student-level audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version suitable for students ranging from high school
to university, while preserving the original creative tone, structure, and instructional potential.

The `StudentsAudience` class integrates with LangChain-compatible LLMs and rewrites the
input prompt to focus on educational value, relatable academic framing, and learning engagement
without removing the original stylistic creativity.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.students import StudentsAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = StudentsAudience()
    >>> prompt = (
    ...     "Describe the laws of thermodynamics as an epic battle between energy, entropy, and equilibrium."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Describe the laws of thermodynamics as an epic battle, adding examples from real-world systems students study in physics...'

Classes:
    - StudentsAudience: Adapts styled prompts for an educational audience, maintaining tone while enhancing learning relevance.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for student learning and engagement.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("students")
class StudentsAudience:
    """
    Adapt content for a student audience.

    This adapter modifies a stylized prompt to introduce academic relevance and instructional clarity,
    making it more engaging and educational for students at various learning levels (high school to undergraduate),
    while preserving its original stylistic tone and creativity.

    Adaptation techniques may include:
        - Aligning with academic learning outcomes and concepts
        - Adding memory aids, study strategies, and student-friendly analogies
        - Framing topics with coursework or exam relevance
        - Incorporating questions that spark curiosity and deeper understanding
        - Maintaining original creative tone and engaging structure

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a student-friendly version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply students audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for student learners.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for students, while preserving the original creative tone and structure.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.students import StudentsAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = StudentsAudience()
            >>> adapter.apply("Explain the water cycle like a mythic tale of transformation.", llm)
            'Explain the water cycle like a mythic tale of transformation, connecting it to science class, lab experiments...'
        """

        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for students (high school to university level), while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add educational and learning-focused elements
        
        Your goal is to make the styled prompt educationally engaging for students while 
        keeping its unique character. Add student-focused elements like:
        - Learning objectives and educational outcomes
        - Study tips and memory aids
        - Connection to coursework and academic concepts
        - Interactive examples and thought-provoking questions
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of calculus using sophisticated 
        mathematical metaphors and analytical language that flows like logical reasoning."

        Adapted for students (preserving style):
        "Craft an elegant explanation of calculus using sophisticated 
        mathematical metaphors and engaging language that flows like a learning 
        adventure. Connect calculus to real-world applications students encounter, 
        include study strategies, relate to other math courses, provide memorable 
        analogies, and make it feel like unlocking mathematical superpowers while 
        maintaining the analytical elegance."

        ====
        Original text: {text}

        Adapted prompt for students (preserving original style):"""

        return llm.invoke(prompt).content
