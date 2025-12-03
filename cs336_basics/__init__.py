import os
import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))