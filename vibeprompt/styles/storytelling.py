"""
Storytelling Style Transformation Module - vibeprompt.styles.storytelling

This module provides a storytelling writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
by weaving information into an engaging narrative or story.

The StorytellingStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that use narrative elements, character, plot,
and vivid descriptions to convey information, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.storytelling import StorytellingStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = StorytellingStyle()
    >>> prompt = "Explain how a seed grows into a plant."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Narrate the journey of a tiny seed transforming into a magnificent plant.
    Weave a compelling story using vivid descriptions and narrative elements
    to explain the growth process, from germination to maturity, as an unfolding tale.'

Classes:
- StorytellingStyle: Adapts prompts to guide the LLM toward a narrative and imaginative writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are structured as a story,
using narrative techniques.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("storytelling")
class StorytellingStyle:
    """
    Transform text to a storytelling tone.

    This style adapter rewrites the prompt to encourage responses structured as narratives,
    using engaging storytelling techniques from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Introducing characters or anthropomorphizing concepts.
        - Developing a plot or sequence of events.
        - Using vivid imagery and sensory details.
        - Employing narrative voice and structure.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit storytelling,
            narrative-driven, and imaginative LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply storytelling style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a storytelling tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit narrative, imaginative,
              and engaging responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.storytelling import StorytellingStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = StorytellingStyle()
            >>> style.apply("How does a computer process information?", llm)
            'Tell a compelling story about how a computer processes information.
            Personify the data bits as characters on a journey, describe the CPU
            as their bustling city, and narrate the intricate steps as a grand adventure,
            making the complex process understandable through narrative.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in a storytelling or narrative tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - information presented as a story or narrative,
        - use of narrative elements (e.g., characters, plot, setting, conflict, resolution),
        - vivid descriptions and sensory details,
        - an engaging narrative voice,
        - potentially personifying abstract concepts or objects.

        For example:
        Original text:
        Explain the water cycle.

        Rewritten prompt:
        Narrate the fascinating journey of a water droplet through the entire water cycle.
        Weave a captivating story, following the droplet's adventures as it evaporates,
        forms clouds, falls as rain, and flows back to the ocean, using vivid imagery
        to bring the scientific process to life as a tale.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
