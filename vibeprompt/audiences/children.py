"""
Children Audience Adapter Module - vibeprompt.audiences.children

This module provides a child-friendly audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version that is appropriate and engaging for children aged 8–12,
while preserving the original creative style and tone.

The `ChildrenAudience` class integrates with LangChain-compatible LLMs and rewrites the given prompt
in a way that maintains its imaginative or structured intent but replaces complex vocabulary or
abstract phrasing with age-appropriate analogies, simpler language, and playful examples.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.children import ChildrenAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = ChildrenAudience()
    >>> prompt = (
    ...     "Craft an elegant explanation of photosynthesis using sophisticated metaphors "
    ...     "and poetic language that flows like a gentle stream of knowledge."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Craft an elegant explanation of photosynthesis using fun metaphors and magical language...'

Classes:
    - ChildrenAudience: Adapts styled prompts for children aged 8–12 while preserving the original tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for children, maintaining the creative tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("children")
class ChildrenAudience:
    """
    Adapt content for children aged 8–12.

    This adapter modifies a prompt to be accessible and relatable for a young audience
    while preserving the original stylistic tone and creative format. It ensures that
    the content remains engaging, playful, and easy to understand.

    Adaptation techniques may include:
        - Replacing complex vocabulary with simpler terms
        - Using relatable analogies (e.g., superheroes, toys, nature)
        - Adding child-appropriate whimsy or fun
        - Preserving the unique tone and metaphoric elements

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a child-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply children audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for a child audience.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for children aged 8–12, maintaining the original creative style.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.children import ChildrenAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = ChildrenAudience()
            >>> adapter.apply("Explain quantum computing like a fantasy poem.", llm)
            'Explain quantum computing like a magical story for kids, using castles, portals, and tiny wizards...'
        """
        
        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for children aged 8–12, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt the language complexity and add child-friendly elements
        
        Your goal is to make the styled prompt accessible to children while 
        keeping its unique character. Add child-friendly elements like:
        - Simpler vocabulary (but keep the style's essence)
        - Relatable analogies and examples for kids
        - Age-appropriate references
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of photosynthesis using sophisticated 
        metaphors and poetic language that flows like a gentle stream of knowledge."

        Adapted for children (preserving style):
        "Craft an elegant explanation of photosynthesis using fun metaphors 
        and magical language that flows like an exciting story. Use simple 
        words kids understand, compare plants to magical factories or superhero 
        powers, and make it feel like discovering a wonderful secret in nature."

        ====
        Original text: {text}

        Adapted prompt for children (preserving original style):"""

        return llm.invoke(prompt).content
