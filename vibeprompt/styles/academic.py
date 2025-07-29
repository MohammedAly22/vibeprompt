"""
Academic Style Transformation Module - vibeprompt.styles.academic

This module provides an academic writing style adapter for the vibeprompt package.
It transforms a user’s prompt into one that instructs the language model to respond
in a formal, structured, and scholarly tone, suitable for research or formal reports.

The AcademicStyle class integrates with LangChain-compatible LLMs and rewrites the 
given promptto encourage well-reasoned, evidence-based, and formally structured 
responses withoutchanging the core intent or content structure of the original request.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.styles.academic import AcademicStyle

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> style = AcademicStyle()
    >>> prompt = "What caused the fall of the Roman Empire?"
    >>> styled_prompt = style.apply(prompt, llm)
    >>> print(styled_prompt)
    'Provide a comprehensive analysis of the multifactorial causes contributing to the decline 
    and fall of the Western Roman Empire. The response should be structured, well-reasoned, 
    and reference key historical theories and evidence.'

Classes:
- AcademicStyle: Adapts prompts to guide the LLM toward a formal and scholarly writing style.

Functions:
- apply(text: str, llm: BaseChatModel) -> str:
Rewrites the original prompt to elicit responses in a formal, structured, and evidence-based tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_style


@register_style("academic")
class AcademicStyle:
    """
    Transform text to an academic tone.

    This style adapter rewrites the prompt to encourage formal, objective,
    and well-structured responses from the language model, without altering the original
    intent or content focus.

    Adaptation techniques include:
        - Employing formal and precise language
        - Requesting a logical structure (e.g., introduction, body, conclusion)
        - Encouraging the use of evidence-based reasoning
        - Avoiding colloquialisms and subjective or anecdotal language

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a rewritten version of the prompt designed to elicit formal, structured, 
            and scholarly LLM responses.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply academic style transformation.

        Parameters:
            - text (str): The original prompt to be rewritten in an academic tone.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created 
            via LLMProviderFactory).

        Returns:
            - str: The rewritten prompt, designed to elicit structured, well-reasoned,
            and evidence-based responses from the LLM.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.styles.academic import AcademicStyle
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> style = AcademicStyle()
            >>> style.apply("Explain the theory of relativity.", llm)
            "Elucidate the fundamental principles of Einstein's theory of relativity, 
            distinguishing between special and general relativity. Your explanation 
            should be formal, precise, and suitable for an academic audience."
        """

        prompt = f"""Your task is to take the following prompt and 
        rewrite it so that it instructs a language model to provide 
        an explanation in a formal academic tone.

        ⚠️ Do NOT answer the prompt. Just rewrite the prompt itself 
        to explicitly request:
        - a formal and objective tone,
        - clear, structured organization (e.g., introduction, body, conclusion),
        - use of precise, domain-specific terminology,
        - an emphasis on evidence-based reasoning,
        - avoidance of colloquial language and personal opinion.

        For example:
        Original text:
        Why is Shakespeare still important?

        Rewritten prompt:
        Analyze the enduring relevance of William Shakespeare's literary works. The 
        response should adopt a scholarly tone, discuss his influence on language 
        and literature, and cite specific examples from his plays and sonnets.

        ====
        Original text: {text}

        Rewritten prompt:"""

        return llm.invoke(prompt).content