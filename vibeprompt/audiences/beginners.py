"""
Beginners Audience Adapter Module - vibeprompt.audiences.beginners

This module provides a beginner-oriented audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version tailored for individuals with little
to no prior knowledge of the subject matter, while preserving the original stylistic tone
and creative structure.

The core functionality is exposed through the `BeginnersAudience` class, which integrates with
LangChain-compatible LLMs to produce adapted prompts that emphasize simplicity, foundational
concepts, and clear, basic explanations, without altering the creative integrity of the prompt.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.beginners import BeginnersAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = BeginnersAudience()
    >>> prompt = (
    ...     "Explain the concept of blockchain using a fun, adventurous narrative "
    ...     "filled with quirky characters and a treasure hunt."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Explain the concept of blockchain for complete beginners using a fun, adventurous narrative
    filled with quirky characters and a treasure hunt. Simplify all complex terms, focus on core,
    basic principles, and ensure the explanation is easy to grasp for someone entirely new to the topic.'

Classes:
    - BeginnersAudience: Adapts styled prompts for a beginner audience while preserving tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a styled prompt to a beginner-friendly context using the provided LLM.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("beginners")
class BeginnersAudience:
    """
    Adapt content for a beginner audience.

    This adapter modifies the input prompt to suit an audience with little to no prior knowledge,
    emphasizing elements such as:
        - Simplification of complex ideas
        - Focus on foundational concepts
        - Use of basic vocabulary
        - Step-by-step explanations
        - Avoidance of jargon or technical terms without clear definitions

    The transformation preserves the original tone, style, and creative structure of the prompt.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a beginner-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply beginner audience adaptation.

        Parameters:
            - text (str): The styled prompt to be adapted.
            - llm (BaseChatModel): A LangChain-compatible chat model (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt tailored for beginners.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.beginners import BeginnersAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = BeginnersAudience()
            >>> adapter.apply("Describe the quantum entanglement with poetic, abstract language.", llm)
            'Describe the basic idea of quantum entanglement for beginners using poetic, abstract language.
            Simplify all underlying physics, focusing on the very fundamental concept so it’s graspable
            for someone with no science background, while retaining the poetic style.'
        """

        prompt = f"""Your task is to take the following styled prompt and
        adapt it to be suitable for individuals with no prior knowledge (beginners),
        while PRESERVING the original style and tone as much as possible.

        ⚠️ IMPORTANT:
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add beginner-focused elements

        Your goal is to make the styled prompt accessible and easy to understand for beginners
        while keeping its unique character. Add beginner-focused elements like:
        - Extreme simplification of complex concepts
        - Focus on fundamental, core ideas only
        - Use of basic, common vocabulary and avoidance of jargon (or define it simply)
        - Clear, step-by-step explanations
        - Relatable analogies for new learners
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Explain the intricacies of neural networks with a sci-fi flair."

        Adapted for beginners (preserving style):
        "Explain the very basic idea of neural networks for absolute beginners, with a sci-fi flair.
        Simplify the 'intricacies' down to core concepts, use approachable sci-fi analogies, and
        focus on what a neural network *is* at its simplest, ensuring it's easy to grasp for someone
        new to the concept, while maintaining the sci-fi tone."

        ====
        Original text: {text}

        Adapted prompt for beginners (preserving original style):
        """

        return llm.invoke(prompt).content
