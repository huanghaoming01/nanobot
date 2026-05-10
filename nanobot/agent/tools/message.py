"""用于向用户发送消息的 message 工具。"""

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable

from nanobot.agent.tools.base import Tool, tool_parameters
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema
from nanobot.bus.events import OutboundMessage
from nanobot.config.paths import get_workspace_path


@tool_parameters(
    tool_parameters_schema(
        content=StringSchema(
            "Message content for proactive or cross-channel delivery. "
            "Do not use this for a normal reply in the current chat."
        ),
        channel=StringSchema(
            "Optional target channel for cross-channel/proactive delivery. "
            "Do not set this to the current runtime channel for a normal reply."
        ),
        chat_id=StringSchema(
            "Optional target chat/user ID for cross-channel/proactive delivery. "
            "Do not set this to the current runtime chat for a normal reply."
        ),
        media=ArraySchema(
            StringSchema(""),
            description=(
                "Optional list of existing file paths to attach for proactive or cross-channel delivery. "
                "Do not use this to resend generate_image outputs in the current chat."
            ),
        ),
        buttons=ArraySchema(
            ArraySchema(StringSchema("Button label")),
            description="Optional: inline keyboard buttons as list of rows, each row is list of button labels.",
        ),
        required=["content"],
    )
)
class MessageTool(Tool):
    """用于在聊天频道向用户发送消息的工具。"""

    def __init__(
        self,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        default_message_id: str | None = None,
        workspace: str | Path | None = None,
    ):
        self._send_callback = send_callback
        self._workspace = (
            Path(workspace).expanduser() if workspace is not None else get_workspace_path()
        )
        self._default_channel: ContextVar[str] = ContextVar(
            "message_default_channel", default=default_channel
        )
        self._default_chat_id: ContextVar[str] = ContextVar(
            "message_default_chat_id", default=default_chat_id
        )
        self._default_message_id: ContextVar[str | None] = ContextVar(
            "message_default_message_id",
            default=default_message_id,
        )
        self._default_metadata: ContextVar[dict[str, Any]] = ContextVar(
            "message_default_metadata",
            default={},
        )
        self._sent_in_turn_var: ContextVar[bool] = ContextVar("message_sent_in_turn", default=False)
        self._record_channel_delivery_var: ContextVar[bool] = ContextVar(
            "message_record_channel_delivery",
            default=False,
        )

    def set_context(
        self,
        channel: str,
        chat_id: str,
        message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """设置当前消息上下文。"""
        self._default_channel.set(channel)
        self._default_chat_id.set(chat_id)
        self._default_message_id.set(message_id)
        self._default_metadata.set(metadata or {})

    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """设置发送消息的回调。"""
        self._send_callback = callback

    def start_turn(self) -> None:
        """重置单回合发送跟踪。"""
        self._sent_in_turn = False

    def set_record_channel_delivery(self, active: bool):
        """将工具发送的消息标记为主动频道投递。"""
        return self._record_channel_delivery_var.set(active)

    def reset_record_channel_delivery(self, token) -> None:
        """恢复之前的主动投递记录状态。"""
        self._record_channel_delivery_var.reset(token)

    @property
    def _sent_in_turn(self) -> bool:
        return self._sent_in_turn_var.get()

    @_sent_in_turn.setter
    def _sent_in_turn(self, value: bool) -> None:
        self._sent_in_turn_var.set(value)

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Proactively send a message to a user/channel, optionally with file attachments. "
            "Use this for reminders, cross-channel delivery, or explicit proactive sends. "
            "Do not use this for the normal reply in the current chat: answer naturally instead. "
            "If channel/chat_id would target the current runtime conversation, do not call this tool "
            "unless the user explicitly asked you to proactively send an existing file attachment. "
            "When generate_image creates images in the current chat, the final assistant reply "
            "automatically attaches them; do not call message just to announce or resend them. "
            "For proactive attachment delivery, use the 'media' parameter with file paths. "
            "Do NOT use read_file to send files — that only reads content for your own analysis."
        )

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        message_id: str | None = None,
        media: list[str] | None = None,
        buttons: list[list[str]] | None = None,
        **kwargs: Any,
    ) -> str:
        from nanobot.utils.helpers import strip_think

        content = strip_think(content)

        if buttons is not None:
            if not isinstance(buttons, list) or any(
                not isinstance(row, list) or any(not isinstance(label, str) for label in row)
                for row in buttons
            ):
                return "Error: buttons must be a list of list of strings"
        default_channel = self._default_channel.get()
        default_chat_id = self._default_chat_id.get()
        channel = channel or default_channel
        chat_id = chat_id or default_chat_id
        # 仅当目标是同一 channel+chat 时继承默认 message_id。
        # 跨聊天发送不能携带原始 message_id，因为某些频道（例如飞书）
        # 会通过 Reply API 用它确定目标会话，从而把消息路由到完全错误的聊天。
        same_target = channel == default_channel and chat_id == default_chat_id
        if same_target:
            message_id = message_id or self._default_message_id.get()
        else:
            message_id = None

        if not channel or not chat_id:
            return "Error: No target channel/chat specified"

        if not self._send_callback:
            return "Error: Message sending not configured"

        if media:
            resolved = []
            for p in media:
                if p.startswith(("http://", "https://")) or os.path.isabs(p):
                    resolved.append(p)
                else:
                    resolved.append(str(self._workspace / p))
            media = resolved

        metadata = dict(self._default_metadata.get()) if same_target else {}
        if message_id:
            metadata["message_id"] = message_id
        if self._record_channel_delivery_var.get() or media:
            metadata["_record_channel_delivery"] = True

        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content,
            media=media or [],
            buttons=buttons or [],
            metadata=metadata,
        )

        try:
            await self._send_callback(msg)
            if channel == default_channel and chat_id == default_chat_id:
                self._sent_in_turn = True
            media_info = f" with {len(media)} attachments" if media else ""
            button_info = f" with {sum(len(row) for row in buttons)} button(s)" if buttons else ""
            return f"Message sent to {channel}:{chat_id}{media_info}{button_info}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
