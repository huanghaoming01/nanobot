"""MyTool：用于检查和配置 agent 循环的运行时状态。"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.agent.subagent import SubagentStatus
from nanobot.agent.tools.base import Tool

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


def _has_real_attr(obj: Any, key: str) -> bool:
    """检查 obj 是否有真实（显式设置的）属性，而非 mock 自动生成的属性。"""
    if isinstance(obj, dict):
        return key in obj
    d = getattr(obj, "__dict__", None)
    if d is not None and key in d:
        return True
    for cls in type(obj).__mro__:
        if key in cls.__dict__:
            return True
    return False


class MyTool(Tool):
    """检查并设置 agent 循环的运行时配置。"""

    BLOCKED = frozenset(
        {
            # 核心基础设施
            "bus",
            "provider",
            "_running",
            "tools",
            # 配置管理
            "_runtime_vars",
            # 子系统
            "runner",
            "sessions",
            "consolidator",
            "dream",
            "auto_compact",
            "context",
            "commands",
            # 敏感运行时状态（凭据、消息路由、任务跟踪）
            "_mcp_servers",
            "_mcp_stacks",
            "_pending_queues",
            "_session_locks",
            "_active_tasks",
            "_background_tasks",
            # 安全边界（检查和修改都阻止）
            "restrict_to_workspace",
            "channels_config",
            "_concurrency_gate",
            "_unified_session",
            "_extra_hooks",
        }
    )

    READ_ONLY = frozenset(
        {
            "subagents",  # 可观察，但替换它会破坏系统
            "_current_iteration",  # 仅由 runner 更新
            "exec_config",  # 允许检查（例如查看 sandbox），禁止修改
            "web_config",  # 允许检查（例如查看 enable），禁止修改
        }
    )

    _DENIED_ATTRS = frozenset(
        {
            "__class__",
            "__dict__",
            "__bases__",
            "__subclasses__",
            "__mro__",
            "__init__",
            "__new__",
            "__reduce__",
            "__getstate__",
            "__setstate__",
            "__del__",
            "__call__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
            "__code__",
            "__globals__",
            "func_globals",
            "func_code",
            "__wrapped__",
            "__closure__",
        }
    )

    # 无论父路径如何都敏感的子字段名称
    _SENSITIVE_NAMES = frozenset(
        {
            "api_key",
            "secret",
            "password",
            "token",
            "credential",
            "private_key",
            "access_token",
            "refresh_token",
            "auth",
        }
    )

    @classmethod
    def _is_sensitive_field_name(cls, name: str) -> bool:
        lowered = name.lower()
        return lowered in cls._SENSITIVE_NAMES or any(
            part in cls._SENSITIVE_NAMES for part in lowered.split("_")
        )

    RESTRICTED: dict[str, dict[str, Any]] = {
        "max_iterations": {"type": int, "min": 1, "max": 100},
        "context_window_tokens": {"type": int, "min": 4096, "max": 1_000_000},
        "model": {"type": str, "min_len": 1},
    }

    _MAX_RUNTIME_KEYS = 64

    def __init__(self, loop: AgentLoop, modify_allowed: bool = True) -> None:
        self._loop = loop
        self._modify_allowed = modify_allowed
        self._channel = ""
        self._chat_id = ""

    def __deepcopy__(self, memo: dict[int, Any]) -> MyTool:
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result._loop = self._loop
        result._modify_allowed = self._modify_allowed
        result._channel = self._channel
        result._chat_id = self._chat_id
        return result

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "my"

    @property
    def description(self) -> str:
        base = (
            "Check and set your own runtime state.\n"
            "Actions: check, set.\n"
            "- check (no key): full config overview — start here.\n"
            "- check (key): drill into a value. Dot-paths allowed "
            "(e.g. '_last_usage.prompt_tokens', 'web_config.enable').\n"
            "- set (key, value): change config or store notes in your scratchpad. "
            "Scratchpad keys persist across turns but not restarts.\n"
            "Key values: _current_iteration (current progress), "
            "max_iterations - _current_iteration = remaining iterations.\n"
            "Note: web_config and exec_config are readable but read-only.\n"
            "\n"
            "When to use:\n"
            "- User asks about your model, settings, or token usage → check that key.\n"
            "- A tool fails or behaves unexpectedly → check the related config to diagnose.\n"
            "- User asks you to remember a preference for this session → set to store it in your scratchpad.\n"
            "- About to start a large task → check context_window_tokens and max_iterations first."
        )
        if not self._modify_allowed:
            base += "\nREAD-ONLY MODE: set is disabled."
        else:
            base += (
                "\nIMPORTANT: Before setting state, predict the potential impact. "
                "If the operation could cause crashes or instability "
                "(e.g. changing model), warn the user first."
            )
        return base

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["check", "set"],
                    "description": "Action to perform",
                },
                "key": {
                    "type": "string",
                    "description": "Dot-path for check/set. Examples: 'max_iterations', 'workspace', 'provider_retry_mode'. "
                    "For check without key, shows all config values.",
                },
                "value": {
                    "description": "New value (for set). Type must match target (int for max_iterations/context_window_tokens, str for model)."
                },
            },
            "required": ["action"],
        }

    def _audit(self, action: str, detail: str) -> None:
        session = f"{self._channel}:{self._chat_id}" if self._channel else "unknown"
        logger.info("self.{} | {} | session:{}", action, detail, session)

    # ------------------------------------------------------------------
    # 路径解析
    # ------------------------------------------------------------------

    def _resolve_path(self, path: str) -> tuple[Any, str | None]:
        parts = path.split(".")
        obj = self._loop
        for part in parts:
            if part in self._DENIED_ATTRS or part.startswith("__"):
                return None, f"'{part}' is not accessible"
            if part in self.BLOCKED:
                return None, f"'{part}' is not accessible"
            if part.lower() in self._SENSITIVE_NAMES:
                return None, f"'{part}' is not accessible"
            try:
                if isinstance(obj, dict):
                    if part in obj:
                        obj = obj[part]
                    else:
                        return None, f"'{part}' not found in dict"
                else:
                    obj = getattr(obj, part)
            except (KeyError, AttributeError) as e:
                return None, f"'{part}' not found: {e}"
        return obj, None

    @staticmethod
    def _validate_key(key: str | None, label: str = "key") -> str | None:
        if not key or not key.strip():
            return f"Error: '{label}' cannot be empty or whitespace"
        return None

    # ------------------------------------------------------------------
    # 智能格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_status(st: SubagentStatus, indent: str = "  ") -> str:
        elapsed = time.monotonic() - st.started_at
        tool_summary = (
            ", ".join(f"{e.get('name', '?')}({e.get('status', '?')})" for e in st.tool_events[-5:])
            or "none"
        )
        lines = [
            f"{indent}phase: {st.phase}, iteration: {st.iteration}, elapsed: {elapsed:.1f}s",
            f"{indent}tools: {tool_summary}",
            f"{indent}usage: {st.usage or 'n/a'}",
        ]
        if st.error:
            lines.append(f"{indent}error: {st.error}")
        if st.stop_reason:
            lines.append(f"{indent}stop_reason: {st.stop_reason}")
        return "\n".join(lines)

    @staticmethod
    def _format_value(val: Any, key: str = "") -> str:
        if isinstance(val, SubagentStatus):
            header = f"Subagent [{val.task_id}] '{val.label}'"
            detail = MyTool._format_status(val, "  ")
            return f"{header}\n  task: {val.task_description}\n{detail}"
        # SubagentManager：委托给其 _task_statuses 字典
        if hasattr(val, "_task_statuses") and isinstance(val._task_statuses, dict):
            return MyTool._format_value(val._task_statuses, key)
        if isinstance(val, dict) and val and isinstance(next(iter(val.values())), SubagentStatus):
            prefix = f"{key}: " if key else ""
            lines = [f"{prefix}{len(val)} subagent(s):"]
            for tid, st in val.items():
                detail = MyTool._format_status(st, "    ")
                lines.append(f"  [{tid}] '{st.label}'\n{detail}")
            return "\n".join(lines)
        if hasattr(val, "tool_names"):
            return f"tools: {len(val.tool_names)} registered — {val.tool_names}"
        # 标量类型：repr 即可
        if isinstance(val, (str, int, float, bool, type(None))):
            r = repr(val)
            return f"{key}: {r}" if key else r
        # 字典：小字典展示内容；大字典展示键，便于点路径导航
        if isinstance(val, dict):
            ks = list(val.keys())
            if not ks:
                return f"{key}: {{}}" if key else "{}"
            if len(ks) <= 5:
                r = repr(val)
                if len(r) <= 200:
                    return f"{key}: {r}" if key else r
            preview = ", ".join(str(k) for k in ks[:15])
            suffix = ", ..." if len(ks) > 15 else ""
            return f"{key}: {{{preview}{suffix}}}" if key else f"{{{preview}{suffix}}}"
        # List/tuple：大对象展示数量，小对象展示 repr
        if isinstance(val, (list, tuple)):
            if len(val) > 20:
                return f"{key}: [{len(val)} items]" if key else f"[{len(val)} items]"
            r = repr(val)
            return f"{key}: {r}" if key else r
        # 复杂对象：小型 Pydantic 模型展示值；其他对象展示字段名便于导航
        cls_name = type(val).__name__
        model_fields = getattr(type(val), "model_fields", None)
        if model_fields:
            fields = list(model_fields.keys())
            if len(fields) <= 8:
                # 小型配置对象：展示 field=value 对
                pairs = []
                for f in fields:
                    fv = getattr(val, f, "?")
                    if MyTool._is_sensitive_field_name(f):
                        continue
                    if isinstance(fv, (str, int, float, bool, type(None))):
                        pairs.append(f"{f}={fv!r}")
                    else:
                        pairs.append(f"{f}=<{type(fv).__name__}>")
                preview = ", ".join(pairs)
                return f"{key}: {preview}" if key else preview
        else:
            fields = [a for a in getattr(val, "__dict__", {}) if not a.startswith("__")]
        if fields:
            preview = ", ".join(str(f) for f in fields[:20])
            suffix = ", ..." if len(fields) > 20 else ""
            return (
                f"{key}: <{cls_name}> [{preview}{suffix}]"
                if key
                else f"<{cls_name}> [{preview}{suffix}]"
            )
        r = repr(val)
        return f"{key}: {r}" if key else r

    # ------------------------------------------------------------------
    # 操作分发
    # ------------------------------------------------------------------

    async def execute(
        self,
        action: str,
        key: str | None = None,
        value: Any = None,
        **_kwargs: Any,
    ) -> str:
        if action in ("inspect", "check"):
            return self._inspect(key)
        if not self._modify_allowed:
            return "Error: set is disabled (tools.my.allow_set is false)"
        if action in ("modify", "set"):
            return self._modify(key, value)
        return f"Unknown action: {action}"

    # -- 检查 --

    def _inspect(self, key: str | None) -> str:
        if not key:
            return self._inspect_all()
        top = key.split(".")[0]
        if top in self._DENIED_ATTRS or top.startswith("__"):
            return f"Error: '{top}' is not accessible"
        obj, err = self._resolve_path(key)
        if err:
            # _runtime_vars 的 "scratchpad" 别名
            if key == "scratchpad":
                rv = self._loop._runtime_vars
                return self._format_value(rv, "scratchpad") if rv else "scratchpad is empty"
            # 回退：检查由 modify 存入 _runtime_vars 的简单键
            if "." not in key and key in self._loop._runtime_vars:
                return self._format_value(self._loop._runtime_vars[key], key)
            return f"Error: {err}"
        # 防御 mock 自动生成的属性
        if "." not in key and not _has_real_attr(self._loop, key):
            if key in self._loop._runtime_vars:
                return self._format_value(self._loop._runtime_vars[key], key)
            return f"Error: '{key}' not found"
        return self._format_value(obj, key)

    def _inspect_all(self) -> str:
        loop = self._loop
        parts: list[str] = []
        # RESTRICTED 键
        for k in self.RESTRICTED:
            parts.append(self._format_value(getattr(loop, k, None), k))
        # 描述中展示的其他有用顶层键
        for k in (
            "workspace",
            "provider_retry_mode",
            "max_tool_result_chars",
            "_current_iteration",
            "web_config",
            "exec_config",
            "subagents",
        ):
            if _has_real_attr(loop, k):
                parts.append(self._format_value(getattr(loop, k, None), k))
        # token 使用量
        usage = loop._last_usage
        if usage:
            parts.append(self._format_value(usage, "_last_usage"))
        rv = loop._runtime_vars
        if rv:
            parts.append(self._format_value(rv, "scratchpad"))
        return "\n".join(parts)

    # -- 修改 --

    def _modify(self, key: str | None, value: Any) -> str:
        if err := self._validate_key(key):
            return err
        top = key.split(".")[0]
        if (
            top in self.BLOCKED
            or top in self._DENIED_ATTRS
            or top.startswith("__")
            or top.lower() in self._SENSITIVE_NAMES
        ):
            self._audit("modify", f"BLOCKED {key}")
            return f"Error: '{key}' is protected and cannot be modified"
        if top in self.READ_ONLY:
            self._audit("modify", f"READ_ONLY {key}")
            return f"Error: '{key}' is read-only and cannot be modified"
        if "." in key:
            parent_path, leaf = key.rsplit(".", 1)
            if leaf in self._DENIED_ATTRS or leaf.startswith("__"):
                self._audit("modify", f"BLOCKED leaf '{leaf}'")
                return f"Error: '{leaf}' is not accessible"
            if leaf.lower() in self._SENSITIVE_NAMES:
                self._audit("modify", f"BLOCKED sensitive leaf '{leaf}'")
                return f"Error: '{leaf}' is not accessible"
            parent, err = self._resolve_path(parent_path)
            if err:
                return f"Error: {err}"
            if isinstance(parent, dict):
                parent[leaf] = value
            else:
                setattr(parent, leaf, value)
            self._audit("modify", f"{key} = {value!r}")
            return f"Set {key} = {value!r}"
        if key in self.RESTRICTED:
            return self._modify_restricted(key, value)
        return self._modify_free(key, value)

    def _modify_restricted(self, key: str, value: Any) -> str:
        spec = self.RESTRICTED[key]
        expected = spec["type"]
        if expected is int and isinstance(value, bool):
            return f"Error: '{key}' must be {expected.__name__}, got bool"
        if not isinstance(value, expected):
            try:
                value = expected(value)
            except (ValueError, TypeError):
                return f"Error: '{key}' must be {expected.__name__}, got {type(value).__name__}"
        old = getattr(self._loop, key)
        if "min" in spec and value < spec["min"]:
            return f"Error: '{key}' must be >= {spec['min']}"
        if "max" in spec and value > spec["max"]:
            return f"Error: '{key}' must be <= {spec['max']}"
        if "min_len" in spec and len(str(value)) < spec["min_len"]:
            return f"Error: '{key}' must be at least {spec['min_len']} characters"
        setattr(self._loop, key, value)
        if key == "max_iterations" and hasattr(self._loop, "_sync_subagent_runtime_limits"):
            self._loop._sync_subagent_runtime_limits()
        self._audit("modify", f"{key}: {old!r} -> {value!r}")
        return f"Set {key} = {value!r} (was {old!r})"

    def _modify_free(self, key: str, value: Any) -> str:
        if _has_real_attr(self._loop, key):
            old = getattr(self._loop, key)
            if isinstance(old, (str, int, float, bool)):
                old_t, new_t = type(old), type(value)
                if old_t is float and new_t is int:
                    pass  # 允许 int → float 强制转换
                elif old_t is not new_t:
                    self._audit(
                        "modify",
                        f"REJECTED type mismatch {key}: expects {old_t.__name__}, got {new_t.__name__}",
                    )
                    return f"Error: '{key}' expects {old_t.__name__}, got {new_t.__name__}"
            setattr(self._loop, key, value)
            self._audit("modify", f"{key}: {old!r} -> {value!r}")
            return f"Set {key} = {value!r} (was {old!r})"
        if callable(value):
            self._audit("modify", f"REJECTED callable {key}")
            return "Error: cannot store callable values"
        err = self._validate_json_safe(value)
        if err:
            self._audit("modify", f"REJECTED {key}: {err}")
            return f"Error: {err}"
        if (
            key not in self._loop._runtime_vars
            and len(self._loop._runtime_vars) >= self._MAX_RUNTIME_KEYS
        ):
            self._audit("modify", f"REJECTED {key}: max keys ({self._MAX_RUNTIME_KEYS}) reached")
            return f"Error: scratchpad is full (max {self._MAX_RUNTIME_KEYS} keys). Remove unused keys first."
        old = self._loop._runtime_vars.get(key)
        self._loop._runtime_vars[key] = value
        self._audit("modify", f"scratchpad.{key}: {old!r} -> {value!r}")
        return f"Set scratchpad.{key} = {value!r}"

    @classmethod
    def _validate_json_safe(cls, value: Any, depth: int = 0) -> str | None:
        if depth > 10:
            return "value nesting too deep (max 10 levels)"
        if isinstance(value, (str, int, float, bool, type(None))):
            return None
        if isinstance(value, list):
            for i, item in enumerate(value):
                if err := cls._validate_json_safe(item, depth + 1):
                    return f"list[{i}] contains {err}"
            return None
        if isinstance(value, dict):
            for k, v in value.items():
                if not isinstance(k, str):
                    return f"dict key must be str, got {type(k).__name__}"
                if err := cls._validate_json_safe(v, depth + 1):
                    return f"dict key '{k}' contains {err}"
            return None
        return f"unsupported type {type(value).__name__}"
