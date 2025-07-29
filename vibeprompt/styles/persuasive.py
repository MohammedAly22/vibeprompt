"""
Persuasive Style Transformation Module - vibeprompt.styles.persuasive

This module provides a persuasive writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
in a convincing, compelling, and influential manner.

The PersuasiveStyle class integrates with LangChain-compatible LLMs and rewrites 
the given prompt to encourage responses that build a strong argument, appeal to 
logic and emotion, and motivate the reader toward a specific viewpoint or action, 
without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.persuasive import PersuasiveStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = PersuasiveStyle()
    >>> prompt = "List the benefits of renewable energy."
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Construct a compelling argument for the adoption of renewable energy. Your 
    response should be persuasive, highlighting the key benefits with emotional 
    and logical appeals, and concluding with a strong call to action for a 
    sustainable future.'

Classes:
- PersuasiveStyle: Adapts prompts to guide the LLM toward a convincing and 
influential writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are persuasive, compelling, 
and action-oriented.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("persuasive")
class PersuasiveStyle:
    """
    Transform text to a persuasive tone.

    This style adapter rewrites the prompt to encourage convincing, influential,
    and compelling responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Employing rhetorical questions and powerful language
        - Structuring the response as a clear, logical argument
        - Appealing to both logic (logos) and emotion (pathos)
        - Concluding with a clear call to action

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit persuasive, 
            convincing, and influential LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply persuasive style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a persuasive tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created 
            via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit compelling, convincing, 
            and action-oriented responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.persuasive import PersuasiveStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = PersuasiveStyle()
            >>> style.apply("Why should people read more books?", llm)
            'Craft a persuasive argument on the importance of reading books. Use 
            compelling language to appeal to the reader\\'s aspirations for knowledge 
            and personal growth, and structure the argument to convincingly advocate 
            for making reading a daily habit.'
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a persuasive and compelling tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - strong, convincing, and influential language,
        - a clear, logical argument structure,
        - appeals to both emotion (pathos) and logic (logos),
        - use of rhetorical devices to engage the reader,
        - a powerful call to action where appropriate.

        For example:
        Original text:
        Tell me about recycling.

        Rewritten prompt:
        Write a persuasive piece on the critical importance of recycling. Frame 
        it as an urgent call to action, using compelling facts and emotional appeals 
        to convince individuals and communities to adopt robust recycling practices.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content