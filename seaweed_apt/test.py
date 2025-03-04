from logger import logger 
from rich.console import Console
# from rich.logging import RichHandler

# # Configure rich console
console = Console(force_terminal=True)

# # Set up handler
# handler = RichHandler(console=console, rich_tracebacks=True, markup=True)
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(message)s",
#     handlers=[handler]
# )


# Test various log levels
logger.debug("This is a debug message")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")

# Test with markup
console.print("[bold red]This text should be bold red[/bold red]")
console.print("[green]This text should be green[/green]")