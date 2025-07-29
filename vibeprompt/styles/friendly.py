"""
Friendly Style Transformation Module - vibeprompt.styles.friendly

This module provides a friendly writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with warmth, empathy, and approachability.

The FriendlyStyle class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage warm, conversational, and encouraging responses without changing the core intent
or content structure of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.friendly import FriendlyStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = FriendlyStyle()
    >>> prompt = "Explain how blockchain works."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Could you please explain how blockchain works in a simple and friendly way? 
    I'd appreciate it if you made it easy to understand, as if you were explaining 
    it to a friend.'

Classes:
- FriendlyStyle: Adapts prompts to guide the LLM toward a warm and approachable writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses in a friendly, empathetic, and conversational tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("friendly")
class FriendlyStyle:
    """
    Transform text to a friendly tone.

    This style adapter rewrites the prompt to encourage warm, empathetic,
    and conversational responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Adding conversational or warm phrasing
        - Emphasizing an encouraging and positive tone
        - Encouraging the LLM to use simpler, more accessible language
        - Preserving the prompt's subject while elevating its approachability

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit friendly, warm, 
            and clear LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply friendly style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a friendly tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created 
            via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit warm, conversational, and 
            encouraging responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.friendly import FriendlyStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = FriendlyStyle()
            >>> style.apply("List the benefits of meditation.", llm)
            'Hi there! Could you kindly list the benefits of meditation in a warm and 
            encouraging tone? I would love to hear about it.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a friendly and approachable tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - a conversational and warm tone,
        - simple and accessible language,
        - an empathetic and encouraging vibe,
        - use of positive phrasing,
        - and an overall welcoming and helpful tone.

        For example:
        Original text:
        What are the main causes of climate change?

        Rewritten prompt:
        Hi! Could you explain the main causes of climate change in a clear 
        and friendly way? Please use simple terms that are easy to understand.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content