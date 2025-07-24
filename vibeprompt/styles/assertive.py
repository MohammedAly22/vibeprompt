"""
Assertive Style Transformation Module - vibeprompt.styles.assertive

This module provides an assertive writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond
with confidence, clarity, and authority.

The `AssertiveStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage firm, direct, and action-oriented responses without changing the core intent
or content structure of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.assertive import AssertiveStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = AssertiveStyle()
    >>> prompt = "Could you maybe explain how blockchain works?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain how blockchain works in a clear, direct, and confident tone. Avoid hedging and use decisive language.'

Classes:
    - AssertiveStyle: Adapts prompts to guide the LLM toward a confident and assertive writing style.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses in an assertive, bold, and goal-focused tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("assertive")
class AssertiveStyle:
    """
    Transform text to an assertive tone.

    This style adapter rewrites the prompt to encourage strong, confident,
    and direct responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Removing passive constructions or hesitant phrasing
        - Emphasizing strong, goal-driven language
        - Encouraging the LLM to use confident, no-nonsense instructions
        - Preserving the prompt's subject while elevating its decisiveness

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit assertive, clear, and confident LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply assertive style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in an assertive tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit confident, direct, and action-oriented responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.assertive import AssertiveStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = AssertiveStyle()
            >>> style.apply("Can you suggest ways to improve productivity?", llm)
            'Suggest ways to improve productivity using confident, direct, and goal-oriented language.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in an assertive tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - confident and direct language,
        - clear and decisive wording,
        - no hesitation or passive voice,
        - strong and goal-oriented phrasing,
        - and unambiguous, authoritative tone.

        For example:
        Original text:
        Can you maybe give me some tips on time management?

        Rewritten prompt:
        Provide effective time management strategies in a confident and assertive tone.
        Use direct and goal-focused language that clearly communicates actionable advice.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
