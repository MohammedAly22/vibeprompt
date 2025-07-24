"""
Developer Audience Adapter Module - vibeprompt.audiences.developers

This module provides a developer-focused audience adapter for the `vibeprompt` package.
It transforms a stylized prompt into a version that resonates with software developers and engineers,
while preserving the original creative style and tone.

The `DevelopersAudience` class integrates with LangChain-compatible LLMs and rewrites the given prompt
in a way that adds technical relevance without removing its stylistic depth. It introduces developer-centric
elements such as implementation insights, performance metrics, programming metaphors, and code structure—all
while maintaining the original creative narrative.

Usage:
    >>> from vibeprompt.core.llms.factory import LLMProviderFactory
    >>> from vibeprompt.audiences.developers import DevelopersAudience

    >>> llm = LLMProviderFactory.create_provider(
    ...     provider_name="cohere",
    ...     api_key="your-cohere-api-key"
    ... )

    >>> adapter = DevelopersAudience()
    >>> prompt = (
    ...     "Craft a poetic explanation of recursion that feels like a journey through mirrors, "
    ...     "with each reflection leading deeper into elegant logic."
    ... )
    >>> adapted_prompt = adapter.apply(prompt, llm)
    >>> print(adapted_prompt)
    'Craft a poetic explanation of recursion using software metaphors like stack frames and call depth...'

Classes:
    - DevelopersAudience: Adapts styled prompts for technical developer audiences while preserving the tone.

Functions:
    - apply(text: str, llm: BaseChatModel) -> str:
        Transforms a stylized prompt to one suitable for engineers, maintaining the original tone.
"""

from langchain.chat_models.base import BaseChatModel

# Local
from ..core.style_registry import register_audience


@register_audience("developers")
class DevelopersAudience:
    """
    Adapt content for a software developer or engineering audience.

    This adapter modifies a prompt to be technically informative and relevant
    for a developer audience, while preserving its original stylistic tone and creativity.

    Adaptation techniques may include:
        - Adding references to code constructs, patterns, or tools
        - Framing concepts through engineering metaphors or system architecture
        - Including implementation hints, performance trade-offs, or best practices
        - Maintaining original tone, metaphor, and creative formatting

    Methods:
        - apply(text: str, llm: BaseChatModel) -> str:
            Returns a developer-tailored version of the input prompt using the specified LLM.
    """

    def apply(self, text: str, llm: BaseChatModel) -> str:
        """
        Apply developer audience adaptation.

        Parameters:
            - text (str): The stylized prompt to be adapted for a developer audience.
            - llm (BaseChatModel): A LangChain-compatible LLM instance (e.g., created via LLMProviderFactory).

        Returns:
            - str: The adapted prompt, suitable for software developers or engineers,
                   while preserving the original style and narrative.

        Example:
            >>> from vibeprompt.core.llms.factory import LLMProviderFactory
            >>> from vibeprompt.audiences.developers import DevelopersAudience
            >>> llm = LLMProviderFactory.create_provider(
            ...     provider_name="cohere",
            ...     api_key="your-cohere-api-key"
            ... )
            >>> adapter = DevelopersAudience()
            >>> adapter.apply("Describe cloud computing like a mythic legend.", llm)
            'Describe cloud computing like a mythic legend, where distributed microservices are legendary heroes...'
        """

        prompt = f"""Your task is to take the following styled prompt and 
        adapt it to be suitable for software developers and engineers, while PRESERVING 
        the original style and tone as much as possible.

        ⚠️ IMPORTANT: 
        - Do NOT answer the prompt
        - Do NOT remove or change the original style elements
        - PRESERVE the creative style, tone, and approach from the original
        - ONLY adapt to add technical developer-focused elements
        
        Your goal is to make the styled prompt technically relevant for developers while 
        keeping its unique character. Add developer-focused elements like:
        - Implementation details and code examples
        - Architecture patterns and best practices
        - Performance considerations and optimization
        - Technical specifications and API references
        - Maintain the original creative approach/style

        Example:
        Original styled prompt:
        "Craft an elegant explanation of data structures using sophisticated 
        metaphors and poetic language that flows like a gentle stream of knowledge."

        Adapted for developers (preserving style):
        "Craft an elegant explanation of data structures using sophisticated 
        programming metaphors and architectural language that flows like clean, 
        efficient code. Include implementation examples, time complexity analysis, 
        memory optimization techniques, and real-world use cases while maintaining 
        the poetic elegance and making it practically applicable for coding."

        ====
        Original text: {text}

        Adapted prompt for developers (preserving original style):"""

        return llm.invoke(prompt).content
