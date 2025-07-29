"""
Empathic Style Transformation Module - vibeprompt.styles.empathic

This module provides an empathic writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with understanding, sensitivity, and a focus on emotional resonance.

The EmpathicStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that acknowledge feelings, show compassion,
and build rapport, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.empathic import EmpathicStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = EmpathicStyle()
    >>> prompt = "Explain the challenges of remote work."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Describe the challenges of remote work with an empathic tone, acknowledging
    the emotional toll and difficulties individuals may face. Focus on understanding
    and validating these experiences, offering supportive insights.'

Classes:
- EmpathicStyle: Adapts prompts to guide the LLM toward an understanding and
  sensitive writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are empathic, compassionate,
and emotionally resonant.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("empathic")
class EmpathicStyle:
    """
    Transform text to an empathic tone.

    This style adapter rewrites the prompt to encourage understanding, sensitive,
    and compassionate responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Acknowledging and validating emotions
        - Using gentle and supportive language
        - Focusing on shared human experiences
        - Avoiding judgmental or dismissive tones

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit empathic,
            compassionate, and emotionally resonant LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply empathic style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in an empathic tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit understanding, compassionate,
              and emotionally resonant responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.empathic import EmpathicStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = EmpathicStyle()
            >>> style.apply("How can I deal with stress?", llm)
            'Respond to the question "How can I deal with stress?" with deep empathy,
            acknowledging the difficulties stress brings. Offer supportive and gentle
            advice, focusing on understanding the emotional experience of stress
            and providing comforting, actionable strategies.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in an empathic and compassionate tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - sensitive, understanding, and compassionate language,
        - acknowledgment and validation of emotions,
        - a focus on building rapport and emotional resonance,
        - avoidance of judgmental or dismissive tones.

        For example:
        Original text:
        Tell me about dealing with loss.

        Rewritten prompt:
        Describe how to approach dealing with loss with immense empathy and sensitivity.
        Focus on acknowledging the profound emotional impact, validating feelings of grief,
        and offering comforting, understanding perspectives rather than quick solutions.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
