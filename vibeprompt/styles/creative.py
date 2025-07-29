"""
Creative Style Transformation Module - vibeprompt.styles.creative

This module provides a creative writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with imagination, originality, and artistic flair.

The CreativeStyle class integrates with LangChain-compatible LLMs and rewrites the 
given prompt to encourage evocative, narrative-driven, and imaginative responses 
without changing the core intent or content structure of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.creative import CreativeStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = CreativeStyle()
    >>> prompt = "Describe a sunset."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Paint a picture with words describing a sunset. Use vivid, sensory language 
    and imaginative metaphors to capture the scene\\'s emotion and beauty, as if 
    writing a piece of poetry or fiction.'

Classes:
- CreativeStyle: Adapts prompts to guide the LLM toward an imaginative and artistic 
writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses in an imaginative, expressive, and 
narrative-driven tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("creative")
class CreativeStyle:
    """
    Transform text to a creative tone.

    This style adapter rewrites the prompt to encourage imaginative, expressive,
    and artistic responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Encouraging vivid imagery and sensory details
        - Promoting the use of storytelling, metaphor, and analogy
        - Asking for originality and a unique perspective
        - Moving beyond literal descriptions to capture emotion and mood

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit imaginative, 
            evocative, and artistic LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply creative style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a creative tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created 
            via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit imaginative, evocative, 
            and narrative-driven responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.creative import CreativeStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = CreativeStyle()
            >>> style.apply("Explain how a car engine works.", llm)
            'Describe the workings of a car engine as a living, breathing creature. 
            Use creative analogies and a narrative style to explain the process of 
            combustion, power, and motion from a unique and imaginative perspective.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a highly creative and imaginative style.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - vivid imagery and rich sensory details,
        - use of metaphors, analogies, and storytelling,
        - an original and unconventional perspective,
        - expressive and evocative language,
        - a focus on capturing emotion and atmosphere, not just facts.

        For example:
        Original text:
        What is rain?

        Rewritten prompt:
        Write a short story or poem about rain from the perspective of a single drop 
        of water on its journey from the cloud to the earth. Use imaginative language 
        to describe its experience and feelings.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content