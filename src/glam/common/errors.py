"""The base error class shared by all pipeline steps."""


class GlamError(Exception):
    """Base class for user-facing glam errors — reported as a clean CLI message, not a traceback."""
