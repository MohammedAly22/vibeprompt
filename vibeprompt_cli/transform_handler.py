"""
Transform handler module for VibePrompt CLI.
Separated to avoid slow imports on every command execution.
"""
import traceback
from pathlib import Path
import typer
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn


def handle_transform(console, load_config_func, 
                    prompt, style, audience, 
                    provider, model, api_key, enable_safety):
    """Handle the prompt transformation process."""
    from vibeprompt_cli.cli_main import (
        display_header, 
        get_available_styles, 
        get_available_audiences
    )

    
    display_header()
    
    # Load configuration only when needed
    config = load_config_func()
    
    # Determine final provider - check config only if provider not provided
    final_provider = provider
    if not final_provider:
        final_provider = config.get("provider")
        if not final_provider:
            console.print("[red]❌ Provider is required[/red]")
            console.print("[cyan]Configure using: vibeprompt config set[/cyan]")
            console.print("[cyan]Or provide via command line: --provider openai[/cyan]")
            raise typer.Exit(1)
    
    # Determine final API key - check config only if api_key not provided
    final_api_key = api_key
    if not final_api_key:
        final_api_key = config.get("api_key")
        if not final_api_key:
            # Check environment variable as fallback
            import os
            env_var_name = f"{final_provider.upper()}_API_KEY"
            final_api_key = os.environ.get(env_var_name)
            if not final_api_key:
                console.print(f"[red]❌ API key is required for {final_provider}[/red]")
                console.print(f"[cyan]Options:[/cyan]")
                console.print(f"[cyan]1. Set environment variable: {env_var_name}[/cyan]")
                console.print(f"[cyan]2. Configure using: vibeprompt config set[/cyan]")
                console.print(f"[cyan]3. Provide via command line: --api-key YOUR_KEY[/cyan]")
                raise typer.Exit(1)
    
    # Use model from config if not provided, will use provider default if None
    final_model = model or config.get("model")
    
    # Validate style only if it's not the default "simple"
    if style != "simple":
        available_styles = get_available_styles()
        if style not in available_styles:
            console.print(f"[red]❌ Unknown style: {style}[/red]")
            console.print(f"[cyan]Available styles: {', '.join(available_styles)}[/cyan]")
            raise typer.Exit(1)
    
    # Validate audience only if it's provided
    if audience:
        available_audiences = get_available_audiences()
        if audience not in available_audiences:
            console.print(f"[red]❌ Unknown audience: {audience}[/red]")
            console.print(f"[cyan]Available audiences: {', '.join(available_audiences)}[/cyan]")
            raise typer.Exit(1)
    
    # Process the prompt
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🔃 Initializing VibePrompt...", total=None)
            
            # Lazy import PromptStyler only when needed
            parent_dir = Path(__file__).parent.parent.parent
            import sys
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from vibeprompt import PromptStyler
            
            progress.update(task, description=f"💫 Transforming prompt: {prompt[:30]}...")
            
            # Initialize PromptStyler
            styler = PromptStyler(
                provider=final_provider,
                model=final_model,
                api_key=final_api_key,
                enable_safety=enable_safety,
                verbose=False
            )
            
            # Transform the prompt
            result = styler.transform(
                prompt=prompt,
                style=style,
                audience=audience
            )
            
            progress.update(task, description="✅ Transformation Completed!")
        
        # Display results
        console.print("\n[bold green]✅ Prompt transformed successfully![/bold green]\n")
        
        # Original prompt
        console.print(Panel(prompt, title="[bold blue]📝 Original Prompt[/bold blue]", border_style="blue"))
        
        # Transformed prompt
        console.print(Panel(result, title="[bold green]💫 Transformed Prompt[/bold green]", border_style="green"))
        
        # Configuration used - show what was actually used  
        config_parts = [
            f"🧠 Provider: {final_provider}",
            f"🎯 Model: {final_model or 'Default'}",
            f"🎨 Style: {style}",
        ]
        
        if audience:
            config_parts.append(f"👥 Audience: {audience}")
        
        config_parts.append(f"🛡️ Safety: {'Enabled' if enable_safety else 'Disabled'}")
        
        config_info = " | ".join(config_parts)
        console.print(f"\n[dim]{config_info}[/dim]")
        
    except Exception as e:
        console.print("[red]❌ Error during prompt transformation:[/red]")
        console.print(traceback.format_exc())
        raise typer.Exit(1)