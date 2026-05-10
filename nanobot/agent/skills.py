"""用于加载 agent 能力技能的加载器。"""

import json
import os
import re
import shutil
from pathlib import Path

import yaml

# 默认内置技能目录（相对于此文件）
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# 起始 ---、YAML 正文（分组 1）、独占一行的结束 ---；支持 CRLF。
_STRIP_SKILL_FRONTMATTER = re.compile(
    r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?",
    re.DOTALL,
)


class SkillsLoader:
    """
    agent 技能加载器。

    技能是 Markdown 文件（SKILL.md），用于教 agent 如何使用特定工具
    或完成特定任务。
    """

    def __init__(
        self,
        workspace: Path,
        builtin_skills_dir: Path | None = None,
        disabled_skills: set[str] | None = None,
    ):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.disabled_skills = disabled_skills or set()

    def _skill_entries_from_dir(
        self, base: Path, source: str, *, skip_names: set[str] | None = None
    ) -> list[dict[str, str]]:
        if not base.exists():
            return []
        entries: list[dict[str, str]] = []
        for skill_dir in base.iterdir():
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = skill_dir.name
            if skip_names is not None and name in skip_names:
                continue
            entries.append({"name": name, "path": str(skill_file), "source": source})
        return entries

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        列出所有可用技能。

        Args:
            filter_unavailable: 如果为 True，过滤掉未满足依赖的技能。

        Returns:
            包含 'name'、'path'、'source' 的技能信息字典列表。
        """
        skills = self._skill_entries_from_dir(self.workspace_skills, "workspace")
        workspace_names = {entry["name"] for entry in skills}
        if self.builtin_skills and self.builtin_skills.exists():
            skills.extend(
                self._skill_entries_from_dir(
                    self.builtin_skills, "builtin", skip_names=workspace_names
                )
            )

        if self.disabled_skills:
            skills = [s for s in skills if s["name"] not in self.disabled_skills]

        if filter_unavailable:
            return [
                skill
                for skill in skills
                if self._check_requirements(self._get_skill_meta(skill["name"]))
            ]
        return skills

    def load_skill(self, name: str) -> str | None:
        """
        按名称加载技能。

        Args:
            name: 技能名称（目录名）。

        Returns:
            技能内容；未找到时返回 None。
        """
        roots = [self.workspace_skills]
        if self.builtin_skills:
            roots.append(self.builtin_skills)
        for root in roots:
            path = root / name / "SKILL.md"
            if path.exists():
                return path.read_text(encoding="utf-8")
        return None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        加载指定技能，以便纳入 agent 上下文。

        Args:
            skill_names: 要加载的技能名称列表。

        Returns:
            格式化后的技能内容。
        """
        parts = [
            f"### Skill: {name}\n\n{self._strip_frontmatter(markdown)}"
            for name in skill_names
            if (markdown := self.load_skill(name))
        ]
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self, exclude: set[str] | None = None) -> str:
        """
        构建所有技能的摘要（名称、描述、路径、可用性）。

        这用于渐进式加载：agent 可在需要时通过 read_file 读取完整技能内容。

        Args:
            exclude: 要从摘要中排除的技能名称集合。

        Returns:
            Markdown 格式的技能摘要。
        """
        all_skills = self.list_skills(filter_unavailable=False)
        if not all_skills:
            return ""

        lines: list[str] = []
        for entry in all_skills:
            skill_name = entry["name"]
            if exclude and skill_name in exclude:
                continue
            meta = self._get_skill_meta(skill_name)
            available = self._check_requirements(meta)
            desc = self._get_skill_description(skill_name)
            if available:
                lines.append(f"- **{skill_name}** — {desc}  `{entry['path']}`")
            else:
                missing = self._get_missing_requirements(meta)
                suffix = f" (unavailable: {missing})" if missing else " (unavailable)"
                lines.append(f"- **{skill_name}** — {desc}{suffix}  `{entry['path']}`")
        return "\n".join(lines)

    def _get_missing_requirements(self, skill_meta: dict) -> str:
        """获取缺失依赖的描述。"""
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return ", ".join(
            [
                f"CLI: {command_name}"
                for command_name in required_bins
                if not shutil.which(command_name)
            ]
            + [f"ENV: {env_name}" for env_name in required_env_vars if not os.environ.get(env_name)]
        )

    def _get_skill_description(self, name: str) -> str:
        """从技能 frontmatter 获取技能描述。"""
        meta = self.get_skill_metadata(name)
        if meta and meta.get("description"):
            return meta["description"]
        return name  # 回退到技能名称

    def _strip_frontmatter(self, content: str) -> str:
        """从 Markdown 内容中移除 YAML frontmatter。"""
        if not content.startswith("---"):
            return content
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if match:
            return content[match.end() :].strip()
        return content

    def _parse_nanobot_metadata(self, raw: object) -> dict:
        """从 frontmatter 字段中提取 nanobot/openclaw 元数据。

        ``raw`` 可以是字典（已由 yaml.safe_load 解析）或 JSON 字符串。
        """
        if isinstance(raw, dict):
            data = raw
        elif isinstance(raw, str):
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}
        if not isinstance(data, dict):
            return {}
        payload = data.get("nanobot", data.get("openclaw", {}))
        return payload if isinstance(payload, dict) else {}

    def _check_requirements(self, skill_meta: dict) -> bool:
        """检查技能依赖是否满足（可执行文件、环境变量）。"""
        requires = skill_meta.get("requires", {})
        required_bins = requires.get("bins", [])
        required_env_vars = requires.get("env", [])
        return all(shutil.which(cmd) for cmd in required_bins) and all(
            os.environ.get(var) for var in required_env_vars
        )

    def _get_skill_meta(self, name: str) -> dict:
        """获取技能的 nanobot 元数据（缓存在 frontmatter 中）。"""
        raw_meta = self.get_skill_metadata(name) or {}
        return self._parse_nanobot_metadata(raw_meta.get("metadata"))

    def get_always_skills(self) -> list[str]:
        """获取标记为 always=true 且满足依赖的技能。"""
        return [
            entry["name"]
            for entry in self.list_skills(filter_unavailable=True)
            if (meta := self.get_skill_metadata(entry["name"]) or {})
            and (
                self._parse_nanobot_metadata(meta.get("metadata")).get("always")
                or meta.get("always")
            )
        ]

    def get_skill_metadata(self, name: str) -> dict | None:
        """
        从技能 frontmatter 获取元数据。

        Args:
            name: 技能名称。

        Returns:
            元数据字典；没有时返回 None。
        """
        content = self.load_skill(name)
        if not content or not content.startswith("---"):
            return None
        match = _STRIP_SKILL_FRONTMATTER.match(content)
        if not match:
            return None
        try:
            parsed = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return None
        if not isinstance(parsed, dict):
            return None
        # yaml.safe_load 会返回原生类型（int、bool、list 等）；
        # 保持原样，让下游消费者拿到正确类型。
        metadata: dict[str, object] = {}
        for key, value in parsed.items():
            metadata[str(key)] = value
        return metadata
