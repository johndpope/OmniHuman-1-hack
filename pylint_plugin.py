# pylint_plugin.py
from pylint.checkers import BaseChecker
from pylint.typing import MessageDefinitionTuple
from astroid import nodes, Module, ClassDef, FunctionDef
from pylint.lint import PyLinter
import logging

# Use your existing logger
logger = logging.getLogger("vasa")

class DuplicateMethodChecker(BaseChecker):
    """
    A custom pylint checker that errors on duplicate method definitions.
    Uses rich logging for better visualization.
    """
    name = 'duplicate-methods'
    priority = -1
    msgs: dict[str, MessageDefinitionTuple] = {
        'E9001': (
            'Duplicate method %r found in classes %r',
            'duplicate-method',
            'Method is defined multiple times across different classes'
        ),
    }

    def __init__(self, linter: PyLinter = None):
        super().__init__(linter)
        self._method_definitions = {}
        self._method_nodes = {}
        self.logger = logger

    def visit_module(self, node: Module) -> None:
        self._method_definitions.clear()
        self._method_nodes.clear()
        self.logger.debug(f"[bold blue]Checking module: {node.name}[/]")

    def visit_classdef(self, node: ClassDef) -> None:
        self.logger.debug(f"[cyan]Analyzing class: {node.name}[/]")
        for method in node.mymethods():
            method_code = self._get_method_code(method)
            if method_code:
                if method_code not in self._method_definitions:
                    self._method_definitions[method_code] = [node.name]
                    self._method_nodes[method_code] = [method]
                else:
                    self._method_definitions[method_code].append(node.name)
                    self._method_nodes[method_code].append(method)
                    self.logger.warning(
                        f"[yellow]Potential duplicate method found: {method.name} "
                        f"in class {node.name}[/]"
                    )

    def leave_module(self, node: Module) -> None:
        for method_code, classes in self._method_definitions.items():
            if len(classes) > 1:
                method_name = self._method_nodes[method_code][0].name
                self.logger.error(
                    f"[red]ERROR: Duplicate method '{method_name}' found in classes {classes}[/]"
                )
                # Add the pylint message
                self.add_message(
                    'E9001',
                    args=(method_name, classes),
                    node=self._method_nodes[method_code][0]
                )

    def _get_method_code(self, node: FunctionDef) -> str:
        if not hasattr(node, 'body'):
            return ''
        
        try:
            method_code = '\n'.join(node.as_string().split('\n')[1:])
            method_code = ' '.join(method_code.split())
            return method_code
        except Exception as e:
            self.logger.error(f"[red]Error processing method {node.name}: {str(e)}[/]")
            return ''

def register(linter: PyLinter) -> None:
    linter.register_checker(DuplicateMethodChecker(linter))


    # seaweed.py
import sys
import glob
from pylint.lint import Run
import logging
from rich.logging import RichHandler

# Import your logger configuration
from logger import logger  # Adjust import path as needed

try:
    python_files = glob.glob('**/*.py', recursive=True)
    if not python_files:
        logger.error("[red]No Python files found[/]")
        sys.exit(1)
    
    logger.info("[green]Starting duplicate method analysis...[/]")
    logger.debug(f"[blue]Files to analyze: {len(python_files)}[/]")
    
    # Run pylint with fail-on-error enabled
    Run(
        [
            '--load-plugins=pylint_plugin',
            '--fail-under=10',
            '--disable=all',
            '--enable=duplicate-method',
        ] + python_files,
        exit=False
    )

except Exception as e:
    logger.error(f"[red]Error during analysis: {str(e)}[/]")
    sys.exit(1)