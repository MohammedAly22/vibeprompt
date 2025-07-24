"""
Sequential Style Transformation Module - vibeprompt.styles.sequential

This module provides a sequential (chain-of-thought) writing style adapter for the `vibeprompt` package.
It transforms a user’s prompt into one that instructs the language model to respond with
clear, logical, and step-by-step reasoning.

The `SequentialStyle` class integrates with LangChain-compatible LLMs and rewrites the given prompt
to encourage methodical thinking, transparent reasoning, and structured explanation—without changing
the original problem or request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.sequential import SequentialStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = SequentialStyle()
    >>> prompt = "How do I solve a math problem using the quadratic formula?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Explain how to solve a math problem using the quadratic formula in a step-by-step,
    chain-of-thought manner. Break down each stage of the reasoning clearly and sequentially.'

Classes:
    - SequentialStyle: Adapts prompts to guide the LLM toward structured, logical, and transparent step-by-step explanations.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Rewrites the original prompt to elicit responses that show logical progression and clear, sequential reasoning.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("sequential")
class SequentialStyle:
    """
    Transform text to a step-by-step, chain-of-thought tone.

    This style adapter rewrites a prompt so that it encourages the language model
    to respond with clearly ordered, methodical thinking and detailed reasoning—ideal
    for instructional, problem-solving, or analytical tasks.

    Adaptation techniques include:
        - Encouraging step-by-step breakdowns of thought
        - Using numbered or bulleted lists where appropriate
        - Maintaining a logical flow from premise to conclusion
        - Explaining reasoning at each step

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to guide the LLM
            to respond in a structured, sequential, and logically coherent manner.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply sequential (chain-of-thought) style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in a step-by-step reasoning tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit responses that proceed in a logical, methodical sequence.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.sequential import SequentialStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = SequentialStyle()
            >>> style.apply("How does photosynthesis work?", llm)
            'Explain how photosynthesis works in a step-by-step, chain-of-thought manner.
            Clearly walk through each stage of the process in order...'

        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to respond 
        using a sequential, chain-of-thought reasoning style.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - a clear step-by-step explanation,
        - logical progression of ideas,
        - transparent reasoning or thought process,
        - numbered or bulleted steps if appropriate,
        - and a tone that encourages methodical thinking.

        For example:
        Original text:
        How do I solve a math problem using the quadratic formula?

        Rewritten prompt:
        Explain how to solve a math problem using the quadratic 
        formula in a step-by-step, chain-of-thought manner. 
        Break down each stage of the reasoning clearly and sequentially.

        ----

        Original text:
        Mohammed needs to buy 3 black pens, each black pen costs 12.5 EGP. 
        He also wants to buy a blue pen that costs 15 EGP. 
        His brother will pay for one of the black pens. 
        How much does Mohammed need to pay to buy the pens?

        Rewritten prompt:
        Solve the following word problem using a sequential, step-by-step 
        explanation. Clearly show each stage of the reasoning to arrive 
        at the final amount Mohammed needs to pay:
        Mohammed needs to buy 3 black pens, each black pen costs 12.5 EGP. 
        He also wants to buy a blue pen that costs 15 EGP. 
        His brother will pay for one of the black pens. 
        How much does Mohammed need to pay to buy the pens?

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content
