"""
Educational Style Transformation Module - vibeprompt.styles.educational

This module provides an educational writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
with clarity, structure, and an emphasis on pedagogical effectiveness.

The EducationalStyle class integrates with LangChain-compatible LLMs and rewrites
the given prompt to encourage responses that simplify complex concepts, use examples,
and are structured for optimal learning, without changing the core intent of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.educational import EducationalStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = EducationalStyle()
    >>> prompt = "How does a car engine work?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain the workings of a car engine in an educational manner, as if teaching a student.
    Break down complex processes into simple, digestible steps, use clear analogies or examples,
    and structure the explanation for maximum understanding and retention.'

Classes:
- EducationalStyle: Adapts prompts to guide the LLM toward a clear and pedagogical writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses that are structured for learning,
simplified, and provide illustrative examples.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("educational")
class EducationalStyle:
    """
    Transform text to an educational tone.

    This style adapter rewrites the prompt to encourage clear, structured,
    and pedagogically effective responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Simplifying complex information into easily understandable parts.
        - Using analogies, examples, and illustrations.
        - Structuring content with clear headings, bullet points, or step-by-step guides.
        - Focusing on clarity, accuracy, and accessibility for a learner.

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit educational,
            instructive, and clear LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply educational style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in an educational tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created
              via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit clear, structured,
              and pedagogically sound responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.educational import EducationalStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = EducationalStyle()
            >>> style.apply("How do black holes form?", llm)
            'Provide an educational explanation of black hole formation, suitable for a learner.
            Simplify the astrophysics concepts, use clear analogies, and structure the information
            in logical steps to ensure easy comprehension and retention.'
        """

        prompt = f"""Your task is to take the following prompt and
        rewrite it so that it instructs a language model to provide
        an explanation in an educational and pedagogically effective tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself
        to explicitly request:
        - clear, simplified language for learning,
        - breakdown of complex topics into manageable parts,
        - use of analogies, examples, or metaphors for illustration,
        - logical and structured presentation (e.g., steps, headings),
        - focus on aiding understanding and retention for a student.

        For example:
        Original text:
        Tell me about supply and demand.

        Rewritten prompt:
        Explain the economic principles of supply and demand in an educational format.
        Break down the concepts into easily digestible lessons, provide relatable examples
        to illustrate their interaction, and structure the explanation to facilitate clear
        understanding for a student new to economics.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
