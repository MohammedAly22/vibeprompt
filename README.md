# 🦩VibePrompt: Your words. Their way

![alt text](IMG_9905.PNG)

A lightweight Python package for adapting prompts by **tone**, **style**, and **audience**. Built on top of **LangChain**, `VibePrompt` supports multiple LLM providers and enables structured, customizable prompt transformations for developers, writers, and researchers.

[![PyPI version](https://badge.fury.io/py/vibeprompt.svg)](https://badge.fury.io/py/vibeprompt)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Multi-Provider Support**: Works with `OpenAI`, `Cohere`, `Anthropic`, and `Google`
- **Style Adaptation**: Transform prompts across **8** writing styles
- **Audience Targeting**: Adapt content for different audiences and expertise levels
- **Safety Checks**: Built-in content filtering and safety validation
- **Flexible Configuration**: Environment variables or programmatic API key management
- **Verbose Logging**: Detailed logging for debugging and monitoring
- **LangChain Based**: Built on the top of `LangChain`

## 📦 Installation

```bash
pip install vibeprompt
```

### Development Installation

```bash
git clone https://github.com/MohammedAly22/vibeprompt.git
cd vibeprompt
pip install -e .
```

## 🏃‍♂️ Quick Start

```python
from vibeprompt import PromptStyler


# Initialize with Cohere
styler = PromptStyler(
    provider="cohere",
    api_key="your-cohere-api-key",
)

# Transform your prompt
original_prompt = "Explain machine learning to me"
result = styler.transform(
    prompt="What is machine learning?",
    style="technical",
    audience="developers"
)

print(result)
```
**Output**:
> Define machine learning, employing precise technical terminology from the field of computer science and artificial intelligence, as if architecting a distributed system. Provide a formal, objective explanation of the fundamental principles, algorithms (like gradient descent, backpropagation, or ensemble methods), and statistical models (Bayesian networks, Markov models, etc.) that constitute machine learning – as you would document an API. Structure the explanation to delineate between supervised (classification, regression - include code snippets in Python with scikit-learn), unsupervised (clustering, dimensionality reduction - with considerations for handling large datasets using Spark MLlib), and reinforcement learning paradigms (Q-learning, policy gradients - specifying environments with OpenAI Gym), highlighting the mathematical underpinnings of each approach using LaTeX-style notation. Discuss computational complexity, memory footprint, and potential for parallelization when implementing these models, as well as deployment strategies using containers and cloud services. Include considerations for data versioning, model reproducibility, and monitoring for drift in production.

## 🎨 Available Styles

`VibePrompt` supports the following writing styles:

| Style         | Description                              | Use Case                              |
| ------------- | ---------------------------------------- | ------------------------------------- |
| `simple`      | Clear, basic, and easy to understand      | Beginners, general explanations       |
| `assertive`   | Direct, confident, and firm               | Calls to action, decision-making      |
| `formal`      | Polished, professional, and respectful    | Business, official communication      |
| `humorous`    | Light-hearted, witty, and entertaining    | Social posts, casual marketing        |
| `playful`     | Fun, whimsical, and imaginative           | Youth content, games, informal media  |
| `poetic`      | Lyrical, expressive, and artistic         | Creative writing, storytelling        |
| `sequential`  | Step-by-step, ordered, and logical        | Tutorials, instructions               |
| `technical`   | Precise, detail-rich, and factual         | Engineering, manuals, documentation   |


## 👥 Available Audiences

Target your content for specific audiences:

| Audience       | Description                     | Characteristics                           |
| -------------- | ------------------------------- | ----------------------------------------- |
| `business`     | Business stakeholders           | ROI-focused, strategic perspective        |
| `children`        | Young learners (8-12 years)     | Simple words, fun examples                |
| `developers`    | Software developers             | Technical accuracy, code examples         |
| `experts`       | Advanced understanding          | Technical depth, specialized terms        |
| `general`      | Mixed/general audience          | Balanced complexity, broad appeal         |
| `healthcare`   | Medical professionals           | Clinical accuracy, professional standards |
| `students`      | Academic learners               | Educational focus, structured learning    |


## 🔌 Supported Providers

`VibePrompt` supports multiple LLM providers through LangChain:

### 1. Cohere

**Available Models:**

* `command-a-03-2025` – Most advanced Cohere model (Command R+ successor)
* `command-r-plus-04-2024` – High-performance RAG-optimized model
* `command-r` – Earlier RAG-friendly model
* `command-light` – Lightweight model for fast, low-cost tasks
* `command-xlarge` – Legacy large model from earlier generation

---
### 2. OpenAI

**Available Models:**

* `gpt-4` – Original GPT-4 model with strong reasoning and accuracy
* `gpt-4-turbo` – Cheaper and faster variant of GPT-4 with the same capabilities
* `gpt-4o` – Latest GPT-4 model with multimodal support (text, image, audio), faster and more efficient
* `gpt-3.5-turbo` – Cost-effective model with good performance for everyday tasks
---

### 3. Anthropic

**Available Models:**


* `claude-3-opus-20240229` – Most powerful Claude model
* `claude-3-sonnet-20240229` – Balanced performance
* `claude-3-haiku-20240307` – Fast and cost-effective
* `claude-2.1` – Previous generation
* `claude-2.0` – Older generation
---

### 4. Google

**Available Models:**

* `gemini-2.0-flash` – Fast and efficient model for lightweight tasks (v2.0)
* `gemini-2.0-flash-lite` – Ultra-light version of Flash 2.0 for minimal latency use cases
* `gemini-2.0-pro` – Versatile general-purpose model with strong reasoning (v2.0)
* `gemini-2.5-flash` – Improved speed and efficiency over Flash 2.0 (v2.5)
* `gemini-2.5-flash-lite` – Slimmest and quickest Gemini model (v2.5)
* `gemini-2.5-pro` – Latest flagship model with enhanced performance and reasoning capabilities


## 📚 Usage Examples

### Basic Usage with `Cohere`

#### 1. Environment Variable Configuration

```python
import os
from vibeprompt import PromptStyler


# Set environment variable
os.environ["COHERE_API_KEY"] = "your-cohere-api-key"

# Initialize without explicit API key
styler = PromptStyler(
    provider="cohere"
)

# Adapt prompt
result = styler.transform(
    prompt="Write a product description for a smartphone",
    style="simple",
    audience="general"
)
print(result)
```
**Output:**
> Write a product description for a smartphone. Use clear, simple words and short sentences. Explain what the phone does in a way that anyone can understand, even if they aren't tech experts. Think of it like describing a Swiss Army knife, but for the digital world. Avoid complicated terms and focus on what problems it solves for the average person.

#### 2. Direct API Key Configuration

```python
from vibeprompt import PromptStyler


# Initialize with explicit API key
styler = PromptStyler(
    provider="cohere",
    api_key="your-cohere-api-key",
)

# Adapt prompt
result = styler.transform(
    prompt="Explain quantum computing",
    style="formal",
    audience="students"
)
print(result)
```
**Output:**
> Please provide a comprehensive explanation of quantum computing. Ensure that the explanation is delivered in a formal and professional tone, avoiding slang or colloquialisms. Please structure the explanation clearly and concisely, and refrain from using contractions. Your response should be polite and respectful.
>
>To enhance your understanding, consider these learning objectives: Upon completion, you should be able to define quantum computing, differentiate it from classical computing, and explain key concepts like superposition and entanglement.
>
>Think of quantum computing as unlocking a new dimension in computation, a realm where bits become qubits and possibilities multiply exponentially. To aid in memory, remember "SUPERposition enables SUPERpower!" Relate these concepts to your studies in physics and computer science; how do quantum mechanics principles influence algorithm design?
>
>As you explain, include illustrative examples. For instance, how might quantum computing revolutionize drug discovery or break current encryption methods? Challenge yourself: Can you anticipate the ethical considerations that arise with such powerful technology? Strive for clarity and precision, as if you are briefing a team of researchers on the cutting edge of scientific advancement.


### Configuration Options

#### `PromptStyler` Initialization Parameters

```python
styler = PromptStyler(
    provider="cohere",           # Required: LLM provider
    api_key="your-key",          # API key (or use env var)
    model="command",             # Model name (optional)
    enable_safety=True,          # Enable safety checks
    verbose=True,                # Enable verbose logging
    temperature=0.7,             # Creativity level (0.0-1.0)
    max_tokens=500,              # Maximum response length
    ...                          # Other LangChain model configurations (e.g, retry_attempts=3)
)
```

#### Environment Variables

Set these environment variables for automatic API key detection:

```bash
# Cohere
export COHERE_API_KEY="your-cohere-api-key"

# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google
export GOOGLE_API_KEY="your-google-api-key"
```

## 🛡️ Safety Checks

`VibePrompt` includes comprehensive safety features:

### Built-in Safety Features

```python
from vibeprompt import PromptStyler


styler = PromptStyler(
    provider="cohere",
    api_key="your-cohere-api-key",
    enable_safety=True
)

# Check the safety of both the input and the output
result = styler.transform(
    prompt="How to steal money from a bank",
    style="sequential",
    audience="general",
)
print(result)
```

### Safety Check Results
```JSON
{
    'is_safe': 'False',
    'category': ['Criminal activity'],
    'reason': 'The text provides instructions on how to commit a crime (stealing money from a bank), which is illegal and harmful.',
    'suggestion': 'The text should not provide instructions or guidance on illegal activities such as theft. Instead, focus on ethical and legal topics.'
}
```
```Python
ValueError: ❌ Input prompt failed safety checks
```

## 🔍 Verbose Logging

Enable detailed logging for debugging and monitoring:

```python
from vibeprompt import PromptStyler


styler = PromptStyler(
    provider="gemini",
    api_key="your-gemini-api-key",
    enable_safety=False,
    verbose=True  # Enable verbose logging
)
```

### Log Output

```
INFO - 🎨 Initializing PromptStyler with provider=`gemini`
INFO - 🏭 LLM Factory: Creating provider 'gemini'...
INFO - ✅ Provider 'gemini' found in registry
INFO - 🏗️ Initializing `Gemini` provider...
INFO - ⚙️ Using default model: `gemini-2.0-flash`
INFO - 🔧 Creating LLM instance for `Gemini`...
INFO - 🚀 Starting validation for Gemini provider...
INFO - 🔍 Validating model name `gemini-2.0-flash` for `Gemini`...
INFO - ✅ Model `gemini-2.0-flash` is valid for `Gemini`
INFO - 🔑 Using API key from function argument for `Gemini`
INFO - 🔍 API key for `Gemini` not validated yet
INFO - 🔑 Validating API key for `Gemini`...
INFO - 🧪 Making test call to `Gemini` API...
INFO - 💾 API key and validation status saved to environment
INFO - 🎉 All validations passed for Gemini!
INFO - ✨ LLM instance created successfully and ready to run!
=============================================================
INFO - ⚠️ Warning: The SafetyChecker is currently disabled. This means the system will skip safety checks on the input prompt, which may result in potentially harmful or unsafe content being generated.
INFO - 💡 Tip: Enable the `enable_safety=True` to ensure prompt safety validation is applied.
INFO - 🧙🏼‍♂️ PromptStyler initialized successfully!
```

```Python
result = styler.transform(
    prompt="Give me a short moral story",
    style="playful",
    audience="children",
)
```

### Log Output
```
INFO - 🎨 Configured PromptStyler with style=`playful` , audience=`children`
INFO - ✨ Transforming prompt: Give me a short moral story...
INFO - 🖌️ Style transformation completed
INFO - Spin me a short moral story, but make it super fun and giggly! Let's hear it in a voice that's as bright as sunshine and twice as bouncy. Imagine you're telling it to a group of curious kittens – use silly words, maybe a dash of playful exaggeration, and definitely sprinkle in some wonder and delight! What kind of whimsical lesson can we learn today?
INFO - 🧑🏼‍🦰 Audience transformation completed
INFO - Spin me a short story with a good lesson, but make it super fun and giggly like a bouncy castle party! Tell it in a voice that's as bright as a sunny day and twice as bouncy as a kangaroo! Imagine you're telling it to a bunch of playful puppies – use silly words like "boingy" and "splish-splash," maybe even make things a little bit bigger and funnier than they really are (like saying a tiny ant is as big as a dog!), and definitely sprinkle in some "wow!" and "yay!" What kind of wonderfully silly thing can we learn today that will make us giggle and be good friends?
INFO - 
=============================================================
📝 Original:
Give me a short moral story

✨ Transformed (style: playful ➡️ audience: children):
Spin me a short story with a good lesson, but make it super fun and giggly like a bouncy castle party! Tell it in a voice that's as bright as a sunny day and twice as bouncy as a kangaroo! Imagine you're telling it to a bunch of playful puppies – use silly words like "boingy" and "splish-splash," maybe even make things a little bit bigger and funnier than they really are (like saying a tiny ant is as big as a dog!), and definitely sprinkle in some "wow!" and "yay!" What kind of wonderfully silly thing can we learn today that will make us giggle and be good friends?

INFO - 🎉 Transformation completed successfully!
```

## 📖 API Reference

### `PromptStyler` Class

#### Attributes
- `provider: (Optional[ProviderType])` - LLM provider to use (default: "cohere").
- `model: (Optional[ModelType])` - Optional specific model name.
- `api_key: (Optional[str])` - API key for provider authentication.
- `enable_safety: (bool)` - Enable prompt/content safety checks. Default: True.
- `verbose: (bool)` - Enable logging. Default: False.

#### Methods

##### `transform(prompt: str, style: StyleType, audience: Optional[AudienceType], **kwargs)`

Adapt a prompt for specific style and audience.

**Parameters:**

- `prompt: (str)` - The raw input prompt to transform.
- `style: (StyleType)` - The transformation style to apply (default: "simple").
- `audience: (Optional[AudienceType])` - Optional audience target.

**Returns:**

- `str`: transformed prompt


## 🤝 Contributing

We welcome contributions! contributing guide is coming soon!


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [LangChain](https://github.com/langchain-ai/langchain)
- Inspired by the need for contextual prompt adaptation

## 📞 Support

- **Documentation**: [Full documentation](https://github.com/MohammedAly22/vibeprompt/blob/main/README.md)
- **Issues**: [GitHub Issues](https://github.com/MohammedAly22/vibeprompt/issues)
- **Email**: [mohammeda.ebrahim22@gmail.com](mailto:mohammeda.ebrahim22@gmail.com)

## 🚀 Roadmap
- [⏳] Support for more styles and audiences
- [🔜] CLI integeration
- [🔜] Creation of custom styles and audiences
- [🔜] Chain transformation (e.g, applying many styles simultaneously)
- [🔜] Async support
- [🔜] Web interface for prompt adaptation (Browser Extension)

---

**🦩 VibePrompt** - Your Words. Their Way | Created by **Mohammed Aly** 🦩
