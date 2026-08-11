"""Validation stage placeholders."""


class ValidationError(Exception):
    ...


def validate_artifact(*args: object, **kwargs: object) -> object:
    ...
