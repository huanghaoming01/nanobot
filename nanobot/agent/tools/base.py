"""agent 工具基类。"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from typing import Any, TypeVar

_ToolT = TypeVar("_ToolT", bound="Tool")

# 匹配 :meth:`Tool._cast_value` / :meth:`Schema.validate_json_schema_value` 的行为
_JSON_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class Schema(ABC):
    """描述工具参数的 JSON Schema 片段抽象基类。

    具体类型位于 :mod:`nanobot.agent.tools.schema`；它们都实现
    :meth:`to_json_schema` 和 :meth:`validate_value`。类方法
    :meth:`validate_json_schema_value` 和 :meth:`fragment` 是共享校验与规范化入口。
    """

    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        """从 JSON Schema ``type`` 解析非 null 类型名（如 ``['string','null']`` -> ``'string'``）。"""
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t  # type: ignore[return-value]

    @staticmethod
    def subpath(path: str, key: str) -> str:
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(val: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        """使用 JSON Schema 片段校验 ``val``；返回错误消息（空表示有效）。

        供 :class:`Tool` 以及每个具体 Schema 的 :meth:`validate_value` 使用。
        """
        raw_type = schema.get("type")
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get(
            "nullable", False
        )
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        if nullable and val is None:
            return []
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} should be integer"]
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} should be number"]
        if (
            t in _JSON_TYPE_MAP
            and t not in ("integer", "number")
            and not isinstance(val, _JSON_TYPE_MAP[t])
        ):
            return [f"{label} should be {t}"]

        errors: list[str] = []
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} must be one of {schema['enum']}")
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} must be >= {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} must be <= {schema['maximum']}")
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} must be at least {schema['minLength']} chars")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} must be at most {schema['maxLength']} chars")
        if t == "object":
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"missing required {Schema.subpath(path, k)}")
            for k, v in val.items():
                if k in props:
                    errors.extend(
                        Schema.validate_json_schema_value(v, props[k], Schema.subpath(path, k))
                    )
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} must have at least {schema['minItems']} items")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} must be at most {schema['maxItems']} items")
            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                for i, item in enumerate(val):
                    errors.extend(
                        Schema.validate_json_schema_value(item, schema["items"], prefix.format(i))
                    )
        return errors

    @staticmethod
    def fragment(value: Any) -> dict[str, Any]:
        """将 Schema 实例或已有 JSON Schema 字典规范化为片段字典。"""
        # 先尝试 to_json_schema：必须区分 Schema 实例和已经是 JSON Schema 的字典
        to_js = getattr(value, "to_json_schema", None)
        if callable(to_js):
            return to_js()
        if isinstance(value, dict):
            return value
        raise TypeError(f"Expected schema object or dict, got {type(value).__name__}")

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """返回兼容 :meth:`validate_json_schema_value` 的片段字典。"""
        ...

    def validate_value(self, value: Any, path: str = "") -> list[str]:
        """校验单个值；返回错误消息（空表示通过）。子类可覆盖以添加额外规则。"""
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)


class Tool(ABC):
    """agent 能力：读取文件、运行命令等。"""

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    _BOOL_TRUE = frozenset(("true", "1", "yes"))
    _BOOL_FALSE = frozenset(("false", "0", "no"))

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        """从 ``['string','null']`` 这类 JSON Schema 联合类型中选取第一个非 null 类型。"""
        return Schema.resolve_json_schema_type(t)

    @property
    @abstractmethod
    def name(self) -> str:
        """函数调用中使用的工具名称。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具功能描述。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具参数的 JSON Schema。"""
        ...

    @property
    def read_only(self) -> bool:
        """此工具是否无副作用且适合并行执行。"""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """此工具是否可与其他并发安全工具一起运行。"""
        return self.read_only and not self.exclusive

    @property
    def exclusive(self) -> bool:
        """即使启用并发，此工具是否也应单独运行。"""
        return False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """运行工具；返回字符串或内容块列表。"""
        ...

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obj, dict):
            return obj
        props = schema.get("properties", {})
        return {k: self._cast_value(v, props[k]) if k in props else v for k, v in obj.items()}

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """在校验前应用安全的 schema 驱动类型转换。"""
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        t = self._resolve_type(schema.get("type"))

        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[t]
            if isinstance(val, expected):
                return val

        if isinstance(val, str) and t in ("integer", "number"):
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val

        if t == "string":
            return val if val is None else str(val)

        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                return True
            if low in self._BOOL_FALSE:
                return False
            return val

        if t == "array" and isinstance(val, list):
            items = schema.get("items")
            return [self._cast_value(x, items) for x in val] if items else val

        if t == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        return val

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """按 JSON schema 校验；空列表表示有效。"""
        if not isinstance(params, dict):
            return [f"parameters must be an object, got {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema must be object type, got {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        """OpenAI 函数 schema。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """类装饰器：附加 JSON Schema，并注入具体 ``parameters`` 属性。

    用于 ``Tool`` 子类，替代手写 ``@property def parameters``。
    schema 存储在类上，每次访问都会返回新的副本。

    示例::

        @tool_parameters({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        class ReadFileTool(Tool):
            ...
    """

    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        frozen = deepcopy(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return deepcopy(frozen)

        cls._tool_parameters_schema = deepcopy(frozen)
        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]

        return cls

    return decorator
