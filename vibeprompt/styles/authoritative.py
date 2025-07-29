"""
Authoritative Style Transformation Module - vibeprompt.styles.authoritative

This module provides an authoritative writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with expertise, confidence, and a commanding tone.

The AuthoritativeStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that present information as definitive,
well-researched, and backed by strong evidence, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.authoritative import AuthoritativeStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = AuthoritativeStyle()
    >>> prompt = "Explain the principles of quantum physics."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Provide a definitive and comprehensive explanation of the core principles of quantum physics.
    Present this information with absolute authority, citing established facts and theories,
    and ensure the language conveys an undeniable command of the subject matter.'

Classes:
- AuthoritativeStyle: Adapts prompts to guide the LLM toward a confident and expert writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are authoritative, definitive,
and demonstrate strong expertise.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("authoritative")
class AuthoritativeStyle:
    """
    Transform text to an authoritative tone.

    This style adapter rewrites the prompt to encourage expert, confident,
    and commanding responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Presenting information as definitive and factual.
        - Using strong, direct, and unambiguous language.
        - Implying extensive knowledge and expertise.
        - Avoiding hedging or tentative phrasing.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit authoritative,
            definitive, and expert LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply authoritative style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in an authoritative tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit expert, confident,
              and commanding responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.authoritative import AuthoritativeStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = AuthoritativeStyle()
            >>> style.apply("What are the best practices for cybersecurity?", llm)
            'State the definitive best practices for cybersecurity. Present these guidelines
            with unwavering authority, as established industry standards, and ensure the
            response exudes absolute confidence and expert knowledge.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in an authoritative and expert tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - definitive, confident, and commanding language,
        - presentation of information as factual and well-established,
        - a tone that implies deep expertise and knowledge,
        - avoidance of any hedging, uncertainty, or tentative phrasing.

        For example:
        Original text:
        Tell me about climate change.

        Rewritten prompt:
        Provide a conclusive and expert overview of climate change. Present the established
        scientific consensus, undeniable evidence, and projected impacts with absolute authority,
        leaving no room for doubt regarding the factual nature of the information.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
