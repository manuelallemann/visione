"""
Visione CLI commands package.

This package contains all the available commands for the Visione CLI.
"""
from typing import List, Type

from .add import AddCommand
from .analyze import AnalyzeCommand
from .command import BaseCommand # Assuming BaseCommand is in command.py
from .compose import ComposeCommand
from .import_ import ImportCommand
from .index import IndexCommand
from .init import InitCommand
from .remove import RemoveCommand
from .serve import ServeCommand
from .transcode import TranscodeCommand

# List of all available commands
# The order here might influence the help output if the CLI tool sorts them as is.
COMMANDS: List[Type[BaseCommand]] = [
    InitCommand,
    AddCommand,
    ImportCommand,
    AnalyzeCommand,
    IndexCommand,
    ServeCommand,
    ComposeCommand,
    RemoveCommand,
    TranscodeCommand, # Added the new TranscodeCommand
]

__all__ = [
    "COMMANDS",
    "InitCommand",
    "AddCommand",
    "ImportCommand",
    "AnalyzeCommand",
    "IndexCommand",
    "ServeCommand",
    "ComposeCommand",
    "RemoveCommand",
    "TranscodeCommand",
    "BaseCommand", # Exporting BaseCommand if it's used externally
]
