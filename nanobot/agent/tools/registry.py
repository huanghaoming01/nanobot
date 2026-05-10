"""用于动态工具管理的工具注册表。"""

from typing import Any

from nanobot.agent.tools.base import Tool


class ToolRegistry:
    """
    agent 工具注册表。

    支持动态注册和执行工具。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """注册工具。"""
        self._tools[tool.name] = tool
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """按名称注销工具。"""
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """按名称获取工具。"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """检查工具是否已注册。"""
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """从 OpenAI 或扁平 schema 中提取规范化工具名。"""
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """获取稳定排序的工具定义，以便提示词缓存友好。

        内置工具会先排序作为稳定前缀，随后排序并追加 MCP 工具。
        结果会缓存到下一次 register/unregister 调用。
        """
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for schema in definitions:
            name = self._schema_name(schema)
            if name.startswith("mcp_"):
                mcp_tools.append(schema)
            else:
                builtins.append(schema)

        builtins.sort(key=self._schema_name)
        mcp_tools.sort(key=self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
        self,
        name: str,
        params: dict[str, Any],
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        """解析、转换并校验一次工具调用。"""
        # 防御无效参数类型（例如 list 而不是 dict）
        if not isinstance(params, dict) and name in ("write_file", "read_file"):
            return (
                None,
                params,
                (
                    f"Error: Tool '{name}' parameters must be a JSON object, got {type(params).__name__}. "
                    'Use named parameters: tool_name(param1="value1", param2="value2")'
                ),
            )

        tool = self._tools.get(name)
        if not tool:
            return (
                None,
                params,
                (f"Error: Tool '{name}' not found. Available: {', '.join(self.tool_names)}"),
            )

        cast_params = tool.cast_params(params)
        errors = tool.validate_params(cast_params)
        if errors:
            return (
                tool,
                cast_params,
                (f"Error: Invalid parameters for tool '{name}': " + "; ".join(errors)),
            )
        return tool, cast_params, None

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """按名称和给定参数执行工具。"""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        tool, params, error = self.prepare_call(name, params)
        if error:
            return error + _HINT

        try:
            assert tool is not None  # 已由 prepare_call() 保护
            result = await tool.execute(**params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"Error executing {name}: {str(e)}" + _HINT

    @property
    def tool_names(self) -> list[str]:
        """获取已注册工具名称列表。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
