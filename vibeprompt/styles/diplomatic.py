"""
Diplomatic Style Transformation Module - vibeprompt.styles.diplomatic

This module provides a diplomatic writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with tact, neutrality, respect, and an emphasis on building consensus or managing sensitive situations.

The DiplomaticStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that are balanced, non-confrontational,
and considerate of diverse perspectives, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.diplomatic import DiplomaticStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = DiplomaticStyle()
    >>> prompt = "Discuss the conflict in the Middle East."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Address the complexities of the Middle East conflict with utmost diplomacy and neutrality.
    Present diverse perspectives respectfully, avoid taking sides or using inflammatory language,
    and focus on fostering understanding and potential common ground.'

Classes:
- DiplomaticStyle: Adapts prompts to guide the LLM toward a tactful and conciliatory writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are balanced, respectful,
and aimed at de-escalation or consensus.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("diplomatic")
class DiplomaticStyle:
    """
    Transform text to a diplomatic tone.

    This style adapter rewrites the prompt to encourage tactful, neutral,
    and respectful responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Using conciliatory and non-confrontational language.
        - Acknowledging multiple perspectives without endorsement.
        - Focusing on common ground and mutual understanding.
        - Avoiding strong opinions, accusations, or judgmental statements.
        - Prioritizing harmony and respect in communication.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit diplomatic,
            tactful, and conciliatory LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply diplomatic style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a diplomatic tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit balanced, respectful,
              and consensus-oriented responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.diplomatic import DiplomaticStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = DiplomaticStyle()
            >>> style.apply("Address the dispute over resource allocation.", llm)
            'Handle the discussion on resource allocation with utmost diplomacy.
            Present all sides of the disagreement fairly, emphasize areas of shared interest,
            and propose solutions that seek compromise and mutual benefit, maintaining a
            respectful and neutral tone throughout.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in a diplomatic and conciliatory tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - tactful, respectful, and neutral language,
        - acknowledgement of multiple perspectives without bias,
        - focus on common ground, understanding, and potential consensus,
        - avoidance of confrontational, judgmental, or inflammatory statements,
        - a tone aimed at de-escalation and constructive dialogue.

        For example:
        Original text:
        Why is Country A right in its stance on the border dispute?

        Rewritten prompt:
        Discuss the border dispute involving Country A with complete diplomatic neutrality.
        Present the perspectives of all relevant parties respectfully, outline the arguments
        and historical context from each side without bias, and focus on pathways for peaceful
        resolution and understanding.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
