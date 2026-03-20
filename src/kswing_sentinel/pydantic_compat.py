"""Pydantic compatibility layer.

Uses real pydantic when available; otherwise falls back to a small local subset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, get_args, get_origin

try:  # pragma: no cover - exercised when dependency exists
    from pydantic import BaseModel, Field, ValidationError  # type: ignore
except Exception:  # pragma: no cover
    class ValidationError(ValueError):
        pass

    @dataclass
    class _FieldInfo:
        default: Any = ...
        ge: float | None = None
        le: float | None = None
        min_length: int | None = None
        max_length: int | None = None

    def Field(default: Any = ..., **kwargs: Any) -> _FieldInfo:
        return _FieldInfo(default=default, **kwargs)

    class BaseModel:
        def __init__(self, **data: Any) -> None:
            annotations = getattr(self.__class__, "__annotations__", {})
            for name, typ in annotations.items():
                default = getattr(self.__class__, name, ...)
                field = default if isinstance(default, _FieldInfo) else None
                if field:
                    default = field.default
                if name in data:
                    value = data[name]
                elif default is not ...:
                    value = default
                else:
                    raise ValidationError(f"Missing required field: {name}")
                self._validate_type(name, value, typ)
                if field:
                    self._validate_constraints(name, value, field)
                setattr(self, name, value)

        def _validate_type(self, name: str, value: Any, typ: Any) -> None:
            origin = get_origin(typ)
            if origin is Literal:
                allowed = get_args(typ)
                if value not in allowed:
                    raise ValidationError(f"{name} must be one of {allowed}")
            elif origin is list:
                if not isinstance(value, list):
                    raise ValidationError(f"{name} must be list")
            elif typ in (float, int, str, bool):
                if not isinstance(value, typ):
                    if not (typ is float and isinstance(value, int)):
                        raise ValidationError(f"{name} must be {typ.__name__}")

        def _validate_constraints(self, name: str, value: Any, field: _FieldInfo) -> None:
            if field.ge is not None and value < field.ge:
                raise ValidationError(f"{name} must be >= {field.ge}")
            if field.le is not None and value > field.le:
                raise ValidationError(f"{name} must be <= {field.le}")
            if field.min_length is not None and hasattr(value, "__len__") and len(value) < field.min_length:
                raise ValidationError(f"{name} length must be >= {field.min_length}")
            if field.max_length is not None and hasattr(value, "__len__") and len(value) > field.max_length:
                raise ValidationError(f"{name} length must be <= {field.max_length}")

        def model_dump(self) -> dict[str, Any]:
            return self.__dict__.copy()
