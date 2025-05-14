# xderived/__init__.py

"""Initialize the xderived plugin."""

from .core import registry, DerivedVariable
from . import standard_variables
from . import accessor # This registers the accessor
from . import config # To make config accessible as xderived.config

# Register standard variables by default when the package is imported
standard_variables.register_standard_variables()

__all__ = ["registry", "DerivedVariable", "accessor", "config", "standard_variables"]

__version__ = "0.3.0" # Placeholder for version, incrementing due to significant changes

