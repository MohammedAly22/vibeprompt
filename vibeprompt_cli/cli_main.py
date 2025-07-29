"""
🦩 VibePrompt CLI - Your Words. Their Way.
A command-line interface for the VibePrompt package.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Confirm
from rich.text import Text


# Initialize Rich console and Typer app
console = Console()
app = typer.Typer(
    name="vibeprompt",
    help="🦩 VibePrompt - Your Words. Their Way. \n\nA CLI tool for adapting prompts by tone, style, and audience.",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


# Configuration file path
CONFIG_DIR = Path.home() / ".vibeprompt"
CONFIG_FILE = CONFIG_DIR / "config.json"


# Default configuration
DEFAULT_CONFIG = {
    "provider": None,
    "model": None,
    "api_key": None,
    "enable_safety": True,
}


# Create subcommands
config_app = typer.Typer(help="Configuration management commands")
styles_app = typer.Typer(help="List available styles")
audiences_app = typer.Typer(help="List available audiences")
providers_app = typer.Typer(help="List available providers")
models_app = typer.Typer(help="List available models")


app.add_typer(config_app, name="config")
app.add_typer(styles_app, name="styles")
app.add_typer(audiences_app, name="audiences")
app.add_typer(providers_app, name="providers")
app.add_typer(models_app, name="models")


def ensure_config_dir():
    """Ensure configuration directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)


def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()
    
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Merge with defaults to handle missing keys
        merged_config = DEFAULT_CONFIG.copy()
        merged_config.update(config)
        return merged_config
    except (json.JSONDecodeError, FileNotFoundError):
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]):
    """Save configuration to file."""
    ensure_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def display_header():
    """Display the VibePrompt header."""
    header = Text("🦩 VibePrompt", style="bold magenta")
    subtitle = Text("Your Words. Their Way.", style="italic cyan")
    console.print(Panel.fit(f"{header}\n{subtitle}", border_style="magenta"))


# Lazy import functions - only import when needed
def get_vibeprompt_data():
    """Lazy import of VibePrompt data to avoid slow startup."""
    try:
        # Add the parent directory to sys.path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from vibeprompt.core.style_registry import get_styles, get_audiences
        from vibeprompt.core.llms.factory import LLMProviderFactory
        
        return {
            'styles': get_styles(),
            'audiences': get_audiences(),
            'providers': LLMProviderFactory.get_available_providers()
        }
    except ImportError as e:
        console.print(f"[red]❌ Failed to import VibePrompt: {e}[/red]")
        raise typer.Exit(1)


def get_available_styles():
    """Get just the styles list - more optimized for validation."""
    try:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from vibeprompt.core.style_registry import get_styles
        return get_styles()
    except ImportError as e:
        console.print(f"[red]❌ Failed to import VibePrompt: {e}[/red]")
        raise typer.Exit(1)


def get_available_audiences():
    """Get just the audiences list - more optimized for validation."""
    try:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from vibeprompt.core.style_registry import get_audiences
        return get_audiences()
    except ImportError as e:
        console.print(f"[red]❌ Failed to import VibePrompt: {e}[/red]")
        raise typer.Exit(1)


def create_provider_instance(provider_name: str, model_name: Optional[str] = None, 
                           api_key: Optional[str] = None, verbose: bool = False):
    """Create a provider instance - lazy import."""
    try:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from vibeprompt.core.llms.factory import LLMProviderFactory
        return LLMProviderFactory.create_provider(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            verbose=verbose
        )
    except ImportError as e:
        console.print(f"[red]❌ Failed to import VibePrompt: {e}[/red]")
        raise typer.Exit(1)


def get_provider_models(provider_name: str):
    """Get available models for a provider - lazy import."""
    try:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from vibeprompt.core.llms.factory import LLMProviderFactory
        return LLMProviderFactory.get_provider_models(provider_name)
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not load models for {provider_name}: {e}[/yellow]")
        return []


@config_app.command("show")
def config_show():
    """Show current configuration."""
    display_header()
    config = load_config()

    table = Table(title="Current Configuration", show_header=True, header_style="bold blue")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    table.add_column("Status", justify="center")

    for key, value in config.items():
        if key == "api_key" and value:
            display_value = f"{'*' * 8}{value[-4:]}" if len(value) > 4 else "****"
        else:
            display_value = str(value) if value is not None else "Not set"
        
        status = "✅" if value is not None else "❌"
        table.add_row(key.replace("_", " ").title(), display_value, status)
    
    console.print(table)
    
    # Check if configuration is complete
    required_fields = ["provider", "api_key"]
    missing_fields = [field for field in required_fields if not config.get(field)]
    if missing_fields:
        console.print(f"\n[yellow]⚠️  Missing required configuration: {', '.join(missing_fields)}[/yellow]")
        console.print("[cyan]💡 Run 'vibeprompt config set' to configure these settings.[/cyan]")


@config_app.command("reset")
def config_reset():
    """Reset configuration to defaults."""
    display_header()
    
    if CONFIG_FILE.exists():
        confirm = Confirm.ask("Are you sure you want to reset all configuration?")
        if confirm:
            CONFIG_FILE.unlink()
            console.print("[green]✅ Configuration reset successfully![/green]")
        else:
            console.print("[yellow]❌ Configuration reset cancelled.[/yellow]")
    else:
        console.print("[yellow]No configuration file found. Nothing to reset.[/yellow]")


@config_app.command("set")
def config_set():
    """Set configuration interactively."""
    # Import the heavy configuration logic only when needed
    from vibeprompt_cli.config_setup import run_config_setup
    run_config_setup(console, load_config, save_config, get_vibeprompt_data)


@styles_app.command("list")
@styles_app.command("list-options")
@styles_app.command("ls")
def list_styles():
    """List all available writing styles."""
    display_header()
    data = get_vibeprompt_data()
    styles = data['styles']
    
    table = Table(title="Available Writing Styles", show_header=True, header_style="bold blue")
    table.add_column("Style", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    style_descriptions = {
        "academic": "Evidence-based, structured, and citation-aware",
        "assertive": "Direct, confident, and firm",
        "authoritative": "Commanding tone backed by expertise",
        "casual": "Conversational, laid-back, and friendly",
        "creative": "Original, imaginative, and artistic",
        "diplomatic": "Tactful, neutral, and conflict-averse",
        "educational": "Informative, structured for teaching",
        "empathic": "Compassionate and emotionally resonant",
        "formal": "Polished, professional, and respectful",
        "friendly": "Warm, supportive, and encouraging",
        "humorous": "Light-hearted, witty, and entertaining",
        "minimalist": "Concise, essential, and clean",
        "persuasive": "Convincing and benefit-oriented",
        "playful": "Fun, whimsical, and imaginative",
        "poetic": "Lyrical, expressive, and metaphor-rich",
        "sequential": "Step-by-step, ordered, and logical",
        "simple": "Clear, basic, and easy to understand",
        "storytelling": "Narrative-driven, emotional, and character-focused",
        "technical": "Precise, detail-rich, and factual"
    }
    
    if isinstance(styles, dict):
        for style, description in styles.items():
            table.add_row(style, description)
    else:
        for style in styles:
            description = style_descriptions.get(style, "Style description")
            table.add_row(style, description)
    
    console.print(table)
    console.print(f"\n[cyan]Total: {len(styles)} styles available[/cyan]")


@audiences_app.command("list")
@audiences_app.command("ls")
@audiences_app.command("list-options")
def list_audiences():
    """List all available target audiences."""
    display_header()
    data = get_vibeprompt_data()
    audiences = data['audiences']
    
    table = Table(title="Available Target Audiences", show_header=True, header_style="bold blue")
    table.add_column("Audience", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    
    audience_descriptions = {
        "adults": "General adult readers - mature tone, practical context",
        "beginners": "New learners in any domain - simple explanations, foundational concepts",
        "business": "Business stakeholders - ROI-focused, strategic perspective",
        "children": "Young learners (8-12 years) - simple words, fun examples",
        "developers": "Software developers - technical accuracy, code examples",
        "educators": "Teachers, instructors - pedagogical structure, learning outcomes",
        "experts": "Advanced understanding - technical depth, specialized terms",
        "general": "Mixed/general audience - balanced complexity, broad appeal",
        "healthcare": "Medical professionals - clinical accuracy, professional standards",
        "intermediates": "Mid-level learners - building on basics, transitional explanations",
        "professionals": "Industry professionals - formal tone, work-related context",
        "researchers": "Scientific and academic researchers - technical precision, citations",
        "seniors": "Older adults - clear, respectful, possibly slower-paced explanations",
        "students": "Academic learners - educational focus, structured learning",
        "teenagers": "Teen audience (13–18) - casual, relevant, age-appropriate language"
    }
    
    if isinstance(audiences, dict):
        for audience, description in audiences.items():
            table.add_row(audience, description)
    else:
        for audience in audiences:
            description = audience_descriptions.get(audience, "Audience description")
            table.add_row(audience, description)
    
    console.print(table)
    console.print(f"\n[cyan]Total: {len(audiences)} audiences available[/cyan]")


@providers_app.command("list")
@providers_app.command("ls")
@providers_app.command("list-options")
def list_providers():
    """List all available LLM providers."""
    display_header()
    data = get_vibeprompt_data()
    providers = data['providers']
    
    table = Table(title="Available LLM Providers", show_header=True, header_style="bold blue")
    table.add_column("Provider", style="cyan", no_wrap=True)
    table.add_column("Status", style="white")
    table.add_column("Description", style="white")
    
    provider_descriptions = {
        "cohere": "Cohere's command models - great for text generation",
        "openai": "OpenAI's GPT models - versatile and powerful",
        "anthropic": "Anthropic's Claude models - helpful and harmless",
        "gemini": "Google's Gemini models - multimodal capabilities"
    }
    
    for provider in providers:
        status = "✅ Available"
        description = provider_descriptions.get(provider, "Provider description")
        table.add_row(provider, status, description)
    
    console.print(table)
    console.print(f"\n[cyan]Total: {len(providers)} supported providers[/cyan]")


@models_app.command("list")
def list_models(
    provider: str = typer.Option(..., "--provider", "-p", help="Provider name to list models for")
):
    """List available models for a specific provider."""
    display_header()
    data = get_vibeprompt_data()
    
    if provider not in data['providers']:
        console.print(f"[red]❌ Unknown provider: {provider}[/red]")
        console.print(f"[cyan]Available providers: {data['providers']}[/cyan]")
        raise typer.Exit(1)
    
    models = get_provider_models(provider)
    
    if not models:
        console.print(f"[yellow]⚠️  No models available for provider: {provider}[/yellow]")
        raise typer.Exit(1)
    
    table = Table(title=f"Available Models for {provider.title()}", show_header=True, header_style="bold blue")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Type", style="white")
    
    # First model is typically the default
    for i, model in enumerate(models):
        model_type = "Default" if i == 0 else "Alternative"
        table.add_row(model, model_type)
    
    console.print(table)
    console.print(f"\n[cyan]Total: {len(models)} models available for {provider}[/cyan]")


@app.command("transform")
def transform(
    prompt: str = typer.Argument(..., help="The prompt text to adapt"),
    style: Optional[str] = typer.Option("simple", "--style", "-s", help="Writing style to use (default: simple)"),
    audience: Optional[str] = typer.Option(None, "--audience", "-a", help="Target audience (optional)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use (default: provider's default)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for the provider"),
    enable_safety: bool = typer.Option(True, "--enable-safety/--disable-safety", help="Enable/disable safety checks (default: enabled)"),
):
    """Transform a prompt for specific style and audience."""
    # Import the heavy transformation logic only when needed
    from vibeprompt_cli.transform_handler import handle_transform
    handle_transform(
        console, load_config, prompt, style, audience, 
        provider, model, api_key, enable_safety
    )


@app.command()
def main(
    prompt: str = typer.Argument(..., help="The prompt text to adapt"),
    style: Optional[str] = typer.Option("simple", "--style", "-s", help="Writing style to use (default: simple)"),
    audience: Optional[str] = typer.Option(None, "--audience", "-a", help="Target audience (optional)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider to use"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model to use (default: provider's default)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="API key for the provider"),
    enable_safety: bool = typer.Option(True, "--enable-safety/--disable-safety", help="Enable/disable safety checks (default: enabled)"),
):
    """Adapt a prompt for specific style and audience. Short form of 'vibeprompt transform'."""
    transform(prompt, style, audience, provider, model, api_key, enable_safety)


@app.command()
def version():
    """Show VibePrompt version."""
    display_header()
    console.print("[cyan]VibePrompt CLI v0.2.4[/cyan]")


if __name__ == "__main__":
    app()