# xderived/utils.py

"""Utility functions and custom error classes for the xderived plugin."""

class MissingDependencyError(AttributeError):
    """Custom error for when a derived variable dependency is missing."""
    pass

class ComputationError(RuntimeError):
    """Custom error for failures during derived variable computation."""
    pass

class RegistrationError(ValueError):
    """Custom error for issues during derived variable registration."""
    pass

