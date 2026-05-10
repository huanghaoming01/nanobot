"""用于 shell 命令执行的沙箱后端。

如需添加新后端，实现以下签名的函数：
    _wrap_<name>(command: str, workspace: str, cwd: str) -> str
并将其注册到下方 _BACKENDS。
"""

import shlex
from pathlib import Path

from nanobot.config.paths import get_media_dir


def _bwrap(command: str, workspace: str, cwd: str) -> str:
    """用 bubblewrap 沙箱包装命令（容器内需要 bwrap）。

    只有工作区会以读写方式 bind mount；其父目录（包含 config.json）
    会隐藏在一个新的 tmpfs 后。媒体目录会以只读方式 bind mount，
    使 exec 命令可以读取上传的附件。
    """
    ws = Path(workspace).resolve()
    media = get_media_dir().resolve()

    try:
        sandbox_cwd = str(ws / Path(cwd).resolve().relative_to(ws))
    except ValueError:
        sandbox_cwd = str(ws)

    required = ["/usr"]
    optional = [
        "/bin",
        "/lib",
        "/lib64",
        "/etc/alternatives",
        "/etc/ssl/certs",
        "/etc/resolv.conf",
        "/etc/ld.so.cache",
    ]

    args = ["bwrap", "--new-session", "--die-with-parent"]
    for p in required:
        args += ["--ro-bind", p, p]
    for p in optional:
        args += ["--ro-bind-try", p, p]
    args += [
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        str(ws.parent),  # 遮蔽配置目录
        "--dir",
        str(ws),  # 重新创建工作区挂载点
        "--bind",
        str(ws),
        str(ws),
        "--ro-bind-try",
        str(media),
        str(media),  # 只读访问媒体目录
        "--chdir",
        sandbox_cwd,
        "--",
        "sh",
        "-c",
        command,
    ]
    return shlex.join(args)


_BACKENDS = {"bwrap": _bwrap}


def wrap_command(sandbox: str, command: str, workspace: str, cwd: str) -> str:
    """使用指定沙箱后端包装 *command*。"""
    if backend := _BACKENDS.get(sandbox):
        return backend(command, workspace, cwd)
    raise ValueError(f"Unknown sandbox backend {sandbox!r}. Available: {list(_BACKENDS)}")
