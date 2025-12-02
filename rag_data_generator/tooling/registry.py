"""Tool registry and helper used by the interruption pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from .tools import Tool, build_default_tools


@dataclass
class ToolResult:
    plugin: str
    ok: bool
    data: Any = None
    error: Optional[str] = None

    def to_observation(self) -> str:
        payload = {"plugin": self.plugin, "ok": self.ok, "data": self.data, "error": self.error}
        return json.dumps(payload, ensure_ascii=False)


class ToolRegistry:
    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        if tools is None:
            tools = build_default_tools()
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.spec.name_for_model] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def describe(self) -> List[dict]:
        return [tool.spec.__dict__ for tool in self._tools.values()]

    def invoke(self, plugin_name: str, plugin_args: str) -> ToolResult:
        tool = self.get(plugin_name)
        if tool is None:
            return ToolResult(plugin=plugin_name, ok=False, error=f"Tool {plugin_name} not found")

        payload: Dict[str, Any]
        try:
            payload = json.loads(plugin_args)
            if not isinstance(payload, dict):
                payload = {"input": plugin_args}
        except Exception:
            payload = {"input": plugin_args}

        try:
            result = tool.invoke(payload)
            return ToolResult(plugin=plugin_name, ok=True, data=result)
        except Exception as exc:  # pragma: no cover - defensive
            return ToolResult(plugin=plugin_name, ok=False, error=str(exc))


__all__ = ["ToolRegistry", "ToolResult"]
