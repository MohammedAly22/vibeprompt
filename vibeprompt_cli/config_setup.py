"""
Configuration setup module for VibePrompt CLI.
Separated to avoid slow imports on every command execution.
"""

import os
from pathlib import Path
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn


def create_provider_class(provider_name: str):
    """Create provider class instance - only imported when needed."""
    parent_dir = Path(__file__).parent.parent.parent
    import sys
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from vibeprompt.core.llms import (
        OpenAIProvider, 
        AnthropicProvider, 
        GeminiProvider, 
        CohereProvider
    )
    
    if provider_name == "cohere":
        return CohereProvider()
    elif provider_name == "gemini":
        return GeminiProvider()
    elif provider_name == "anthropic":
        return AnthropicProvider()
    elif provider_name == "openai":
        return OpenAIProvider()
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def validate_configurations(provider_name: str, model_name=None, api_key=None, verbose=False):
    """Validate provider configuration."""
    parent_dir = Path(__file__).parent.parent.parent
    import sys
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from vibeprompt.core.llms.factory import LLMProviderFactory
    from vibeprompt.utils.validation import Validator, ValidationError
    
    try:
        provider = LLMProviderFactory.create_provider(
            provider_name=provider_name,
            model_name=model_name,
            api_key=api_key,
            verbose=verbose
        )
        validator = Validator(verbose=verbose)
        validator.validate_provider(provider)
        return provider
    except (ValueError, ValidationError) as e:
        raise Exception(f"Provider validation failed: {e}")


def run_config_setup(console, load_config_func, save_config_func, get_vibeprompt_data_func):
    """Run the interactive configuration setup."""
    from vibeprompt_cli.cli_main import display_header
    
    display_header()
    console.print("[bold blue]Let's configure VibePrompt 🦩![/bold blue]\n")
    
    config = load_config_func()
    data = get_vibeprompt_data_func()
    
    # Provider selection
    console.print("[bold cyan]1. Select LLM Provider[/bold cyan]")
    provider_table = Table(show_header=False, box=None)
    for i, provider in enumerate(data['providers']):
        provider_table.add_row(f"[cyan]{i+1}[/cyan]", f"[white]{provider}[/white]")
    
    console.print(provider_table)

    provider_choice = Prompt.ask(
        "Select a provider",
        choices=[str(i) for i in range(1, len(data['providers']) + 1)],
        default="1"
    )
    provider = data['providers'][int(provider_choice) - 1]
    config["provider"] = provider

    # Model selection
    console.print(f"\n[bold cyan]2. Select Model for {provider}[/bold cyan]")
    try:
        provider_class = create_provider_class(provider_name=provider)
        available_models = provider_class.get_valid_models()
        default_model = provider_class.get_default_model()

        if available_models:
            models_table = Table(show_header=False, box=None)
            models_table.add_row("[cyan]0[/cyan]", f"[white]Use default model ({default_model})[/white]")
            for i, model in enumerate(available_models, 1):
                marker = " [green](default)[/green]" if model == default_model else ""
                models_table.add_row(f"[cyan]{i}[/cyan]", f"[white]{model}{marker}[/white]")
            console.print(models_table)

            model_choice = Prompt.ask(
                "Select a model",
                choices=[str(i) for i in range(len(available_models) + 1)],
                default="0"
            )

            if model_choice == "0":
                config["model"] = None
            else:
                config["model"] = available_models[int(model_choice) - 1]
        else:
            config["model"] = None
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not load models for {provider}: {e}. Using default model.[/yellow]")
        config["model"] = None

    # API Key
    console.print(f"\n[bold cyan]3. API Key for {provider}[/bold cyan]")
    env_var_name = f"{provider.upper()}_API_KEY"
    env_api_key = os.environ.get(env_var_name)

    if env_api_key:
        console.print(f"[green]✅ Found {env_var_name} in environment[/green]")
        use_env = Confirm.ask("Use the API key from environment variable?", default=True)
        if use_env:
            config["api_key"] = None
        else:
            api_key = Prompt.ask(
                f"Enter your {provider} API key",
                password=True,
                show_default=False
            )
            config["api_key"] = api_key
    else:
        console.print(f"[cyan]💡 You can also set the {env_var_name} environment variable[/cyan]")
        api_key = Prompt.ask(
            f"Enter your {provider} API key",
            password=True,
            show_default=False
        )
        config["api_key"] = api_key

    # Test configuration
    console.print(f"\n[bold cyan]4. Testing Configuration[/bold cyan]")
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("🔃 Validating provider configuration...", total=None)
            validate_configurations(
                provider_name=config['provider'],
                model_name=config["model"],
                api_key=config["api_key"]
            )
            progress.update(task, description="✅ Configuration validated!")
        
        console.print("[green]✅ Configuration test successful![/green]")
    except Exception as e:
        console.print(f"[red]❌ Configuration test failed: {e}[/red]")
        console.print("[yellow]⚠️ Configuration will be saved but may need adjustment.[/yellow]")

    # Safety settings
    console.print(f"\n[bold cyan]5. Safety Settings[/bold cyan]")
    config["enable_safety"] = Confirm.ask("Enable safety checks?", default=True)

    # Save configuration
    try:
        save_config_func(config)
        console.print("\n[green]✅ Configuration saved successfully![/green]")
        
        # Display summary
        console.print("\n[bold blue]Configuration Summary:[/bold blue]")
        summary_table = Table(show_header=False, box=None)
        summary_table.add_row("Provider:", f"[green]{config['provider']}[/green]")
        summary_table.add_row("Model:", f"[green]{config['model'] or 'Default'}[/green]")
        summary_table.add_row("API Key:", f"[green]{'Set' if config['api_key'] else 'From environment'}[/green]")
        summary_table.add_row("Safety Checks:", f"[green]{'Enabled' if config['enable_safety'] else 'Disabled'}[/green]")
        console.print(summary_table)
    
    except Exception as e:
        console.print(f"[red]❌ Failed to save configuration: {e}[/red]")