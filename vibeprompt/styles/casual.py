"""
Casual Style Transformation Module - vibeprompt.styles.casual

This module provides a casual writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
in a relaxed, informal, and conversational manner.

The CasualStyle class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage laid-back, easygoing, and informal responses, as if talking to a friend,
without changing the core intent or content structure of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.casual import CasualStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = CasualStyle()
    >>> prompt = "Describe the process of photosynthesis."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Hey, can you break down how photosynthesis works? Keep it simple and chill, 
    no need to be super formal.'

Classes:
- CasualStyle: Adapts prompts to guide the LLM toward a relaxed and informal writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses in a casual, laid-back, and conversational tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("casual")
class CasualStyle:
    """
    Transform text to a casual tone.

    This style adapter rewrites the prompt to encourage relaxed, informal,
    and conversational responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Using informal language, slang, and contractions
        - Adopting a relaxed, easygoing, and conversational tone
        - Removing formal structures and jargon
        - Phrasing the request as a simple question to a peer

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit casual, 
            laid-back, and clear LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply casual style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a casual tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., 
            created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit relaxed, informal, 
            and conversational responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.casual import CasualStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = CasualStyle()
            >>> style.apply("Provide a summary of the novel '1984'.", llm)
            "Yo, can you give me the lowdown on the book '1984'? Just the main 
            points, nothing too heavy."
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a casual and informal tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - a relaxed and laid-back tone,
        - use of informal language, contractions, or everyday slang,
        - no corporate jargon or overly formal phrasing,
        - a conversational and easygoing manner,
        - a style like you're talking to a buddy.

        For example:
        Original text:
        How does the stock market function?

        Rewritten prompt:
        Can you explain how the stock market works in a chill way? Like, 
        break it down for me so it's easy to get.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content