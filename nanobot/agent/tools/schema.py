"""JSON Schema 片段类型：都继承 :class:`~nanobot.agent.tools.base.Schema`，用于描述和约束工具参数。

- ``to_json_schema()``：返回兼容 :meth:`~nanobot.agent.tools.base.Schema.validate_json_schema_value` /
  :class:`~nanobot.agent.tools.base.Tool`.
- ``validate_value(value, path)``：按此 schema 校验单个值；返回错误消息列表（空表示有效）。

共享校验和片段规范化位于 :class:`~nanobot.agent.tools.base.Schema` 的类方法中。

注意：Python 不允许继承 ``bool``，因此布尔值使用 :class:`BooleanSchema`。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nanobot.agent.tools.base import Schema


class StringSchema(Schema):
    """字符串参数：``description`` 描述字段，可选长度边界和枚举。"""

    def __init__(
        self,
        description: str = "",
        *,
        min_length: int | None = None,
        max_length: int | None = None,
        enum: tuple[Any, ...] | list[Any] | None = None,
        nullable: bool = False,
    ) -> None:
        self._description = description
        self._min_length = min_length
        self._max_length = max_length
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "string"
        if self._nullable:
            t = ["string", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._min_length is not None:
            d["minLength"] = self._min_length
        if self._max_length is not None:
            d["maxLength"] = self._max_length
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class IntegerSchema(Schema):
    """整数参数：可选占位整数（旧版构造签名）、描述和边界。"""

    def __init__(
        self,
        value: int = 0,
        *,
        description: str = "",
        minimum: int | None = None,
        maximum: int | None = None,
        enum: tuple[int, ...] | list[int] | None = None,
        nullable: bool = False,
    ) -> None:
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "integer"
        if self._nullable:
            t = ["integer", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._minimum is not None:
            d["minimum"] = self._minimum
        if self._maximum is not None:
            d["maximum"] = self._maximum
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class NumberSchema(Schema):
    """数值参数（JSON number）：描述和可选边界。"""

    def __init__(
        self,
        value: float = 0.0,
        *,
        description: str = "",
        minimum: float | None = None,
        maximum: float | None = None,
        enum: tuple[float, ...] | list[float] | None = None,
        nullable: bool = False,
    ) -> None:
        self._value = value
        self._description = description
        self._minimum = minimum
        self._maximum = maximum
        self._enum = tuple(enum) if enum is not None else None
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "number"
        if self._nullable:
            t = ["number", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._minimum is not None:
            d["minimum"] = self._minimum
        if self._maximum is not None:
            d["maximum"] = self._maximum
        if self._enum is not None:
            d["enum"] = list(self._enum)
        return d


class BooleanSchema(Schema):
    """布尔参数（独立类，因为 Python 禁止继承 ``bool``）。"""

    def __init__(
        self,
        *,
        description: str = "",
        default: bool | None = None,
        nullable: bool = False,
    ) -> None:
        self._description = description
        self._default = default
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "boolean"
        if self._nullable:
            t = ["boolean", "null"]
        d: dict[str, Any] = {"type": t}
        if self._description:
            d["description"] = self._description
        if self._default is not None:
            d["default"] = self._default
        return d


class ArraySchema(Schema):
    """数组参数：元素 schema 由 ``items`` 给出。"""

    def __init__(
        self,
        items: Any | None = None,
        *,
        description: str = "",
        min_items: int | None = None,
        max_items: int | None = None,
        nullable: bool = False,
    ) -> None:
        self._items_schema: Any = items if items is not None else StringSchema("")
        self._description = description
        self._min_items = min_items
        self._max_items = max_items
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "array"
        if self._nullable:
            t = ["array", "null"]
        d: dict[str, Any] = {
            "type": t,
            "items": Schema.fragment(self._items_schema),
        }
        if self._description:
            d["description"] = self._description
        if self._min_items is not None:
            d["minItems"] = self._min_items
        if self._max_items is not None:
            d["maxItems"] = self._max_items
        return d


class ObjectSchema(Schema):
    """对象参数：``properties`` 或关键字参数是字段名，值为子 Schema 或 JSON Schema 字典。"""

    def __init__(
        self,
        properties: Mapping[str, Any] | None = None,
        *,
        required: list[str] | None = None,
        description: str = "",
        additional_properties: bool | dict[str, Any] | None = None,
        nullable: bool = False,
        **kwargs: Any,
    ) -> None:
        self._properties = dict(properties or {}, **kwargs)
        self._required = list(required or [])
        self._root_description = description
        self._additional_properties = additional_properties
        self._nullable = nullable

    def to_json_schema(self) -> dict[str, Any]:
        t: Any = "object"
        if self._nullable:
            t = ["object", "null"]
        props = {k: Schema.fragment(v) for k, v in self._properties.items()}
        out: dict[str, Any] = {"type": t, "properties": props}
        if self._required:
            out["required"] = self._required
        if self._root_description:
            out["description"] = self._root_description
        if self._additional_properties is not None:
            out["additionalProperties"] = self._additional_properties
        return out


def tool_parameters_schema(
    *,
    required: list[str] | None = None,
    description: str = "",
    **properties: Any,
) -> dict[str, Any]:
    """为 :meth:`Tool.parameters` 构建根工具参数 ``{"type": "object", "properties": ...}``。"""
    return ObjectSchema(
        required=required,
        description=description,
        **properties,
    ).to_json_schema()
