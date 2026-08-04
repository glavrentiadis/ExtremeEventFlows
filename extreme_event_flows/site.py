from __future__ import annotations

from collections.abc import Mapping
import re

import torch
from torch import Tensor, nn


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    if not safe or safe[0].isdigit():
        safe = f"field_{safe}"
    return safe


class Site(nn.Module):
    """
    Site with an arbitrary set of scalar or tensor-valued attributes.

    Examples include ``x``, ``y``, ``depth``, ``vs30``, ``z1p0``, basin flags,
    or any other inputs needed by a ground-motion model.
    """

    def __init__(self, name: str, fields: Mapping[str, Tensor | float]) -> None:
        super().__init__()
        if not fields:
            raise ValueError("Site fields cannot be empty.")

        self.name = name
        self._field_to_buffer: dict[str, str] = {}

        for field_name, value in fields.items():
            buffer_name = f"_site_{_safe_name(field_name)}"
            if buffer_name in self._field_to_buffer.values():
                raise ValueError(f"Duplicate sanitized site field name: {field_name!r}")
            tensor = torch.as_tensor(value, dtype=torch.get_default_dtype())
            self.register_buffer(buffer_name, tensor.clone())
            self._field_to_buffer[field_name] = buffer_name

    def field(self, name: str) -> Tensor:
        try:
            return getattr(self, self._field_to_buffer[name])
        except KeyError as exc:
            raise KeyError(f"Unknown site field {name!r}.") from exc

    def fields(self) -> dict[str, Tensor]:
        return {name: self.field(name) for name in self._field_to_buffer}

    def require(self, *names: str) -> tuple[Tensor, ...]:
        return tuple(self.field(name) for name in names)

    def extra_repr(self) -> str:
        return f"name={self.name!r}, fields={list(self._field_to_buffer)}"
