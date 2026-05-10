"""跟踪文件读取状态，用于编辑前读取警告和读取去重。"""

from __future__ import annotations

import hashlib
import os
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReadState:
    mtime: float
    offset: int
    limit: int | None
    content_hash: str | None
    can_dedup: bool


def _hash_file(p: str) -> str | None:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except OSError:
        return None


class FileStates:
    """每个会话的读写跟踪器。

    每个实例拥有自己的状态字典，让读取去重（"File unchanged since last read"）
    和编辑前读取警告限定在单个 agent 会话内，不会泄漏到同一进程中的其他会话。
    """

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: dict[str, ReadState] = {}

    def record_read(self, path: str | Path, offset: int = 1, limit: int | None = None) -> None:
        """记录文件已被读取（成功读取后调用）。"""
        p = str(Path(path).resolve())
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            return
        self._state[p] = ReadState(
            mtime=mtime,
            offset=offset,
            limit=limit,
            content_hash=_hash_file(p),
            can_dedup=True,
        )

    def record_write(self, path: str | Path) -> None:
        """记录文件已被写入（更新状态中的 mtime）。"""
        p = str(Path(path).resolve())
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            self._state.pop(p, None)
            return
        self._state[p] = ReadState(
            mtime=mtime,
            offset=1,
            limit=None,
            content_hash=_hash_file(p),
            can_dedup=False,
        )

    def check_read(self, path: str | Path) -> str | None:
        """检查文件是否已读取且仍然新鲜。

        如果正常则返回 None，否则返回警告字符串。
        当 mtime 变化但文件内容相同（例如 touch 或编辑器保存）时，
        校验会通过，以避免误报过期警告。
        """
        p = str(Path(path).resolve())
        entry = self._state.get(p)
        if entry is None:
            return "Warning: file has not been read yet. Read it first to verify content before editing."
        try:
            current_mtime = os.path.getmtime(p)
        except OSError:
            return None
        if current_mtime != entry.mtime:
            if entry.content_hash and _hash_file(p) == entry.content_hash:
                entry.mtime = current_mtime
                return None
            return "Warning: file has been modified since last read. Re-read to verify content before editing."
        # mtime 未变化：仍检查内容哈希，以检测快速修改
        if entry.content_hash and _hash_file(p) != entry.content_hash:
            return "Warning: file has been modified since last read. Re-read to verify content before editing."
        return None

    def is_unchanged(self, path: str | Path, offset: int = 1, limit: int | None = None) -> bool:
        """如果文件之前以相同参数读取且内容未变化，则返回 True。"""
        p = str(Path(path).resolve())
        entry = self._state.get(p)
        if entry is None:
            return False
        if not entry.can_dedup:
            return False
        if entry.offset != offset or entry.limit != limit:
            return False
        try:
            current_mtime = os.path.getmtime(p)
        except OSError:
            return False
        if current_mtime != entry.mtime:
            # mtime 已变化：检查内容是否也变化
            current_hash = _hash_file(p)
            if current_hash != entry.content_hash:
                # 内容确实变化：不去重
                entry.can_dedup = False
                return False
            # mtime 变化但内容相同（例如 touch）：标记为不可去重，强制下次完整读取
            entry.can_dedup = False
            return True
        # mtime 未变化：内容必须相同
        return True

    def get(self, path: str | Path) -> ReadState | None:
        """返回路径对应的原始 ReadState 记录；没有则返回 None。"""
        return self._state.get(str(Path(path).resolve()))

    def clear(self) -> None:
        """清除所有跟踪状态（便于测试）。"""
        self._state.clear()


class FileStateStore:
    """每个会话文件读写状态的查找表。"""

    __slots__ = ("_states_by_key",)

    def __init__(self) -> None:
        self._states_by_key: dict[str, FileStates] = {}

    def for_session(self, session_key: str | None) -> FileStates:
        key = session_key or "__default__"
        states = self._states_by_key.get(key)
        if states is None:
            states = FileStates()
            self._states_by_key[key] = states
        return states

    def clear(self) -> None:
        self._states_by_key.clear()


_current_file_states: ContextVar[FileStates | None] = ContextVar(
    "nanobot_file_states",
    default=None,
)


def current_file_states(default: FileStates) -> FileStates:
    """返回绑定到当前 agent 任务的 FileStates；没有则使用回退值。"""
    return _current_file_states.get() or default


def bind_file_states(file_states: FileStates) -> Token[FileStates | None]:
    """为当前异步任务绑定文件读写状态。"""
    return _current_file_states.set(file_states)


def reset_file_states(token: Token[FileStates | None]) -> None:
    _current_file_states.reset(token)


# 模块级默认实例，为了向后兼容直接访问的测试和调用方而保留。
# 按会话工作的调用方应持有自己的 FileStates 实例，而不是触碰此实例。
_default = FileStates()


def record_read(path: str | Path, offset: int = 1, limit: int | None = None) -> None:
    _default.record_read(path, offset=offset, limit=limit)


def record_write(path: str | Path) -> None:
    _default.record_write(path)


def check_read(path: str | Path) -> str | None:
    return _default.check_read(path)


def is_unchanged(path: str | Path, offset: int = 1, limit: int | None = None) -> bool:
    return _default.is_unchanged(path, offset=offset, limit=limit)


def clear() -> None:
    _default.clear()


# 旧版属性，用于兼容直接访问模块级字典的调用方（filesystem.py 曾这样做）。
# 以类似 property 的访问器保留，确保现有导入继续工作。
def __getattr__(name: str):
    if name == "_state":
        return _default._state
    raise AttributeError(name)
