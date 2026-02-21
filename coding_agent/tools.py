"""Tool definitions (OpenAI format) and local execution for the coding agent."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from .client import AntiSlopClient

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using Google. Returns titles, URLs, and descriptions. Use for factual questions, current events, or when you need to look something up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file. Returns the full text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (absolute or relative to working directory)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory, optionally filtered by a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path (default: working directory)"},
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py' or '**/*.ts' (default: '*')"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. Overwrites existing files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Safely patch a file by finding and replacing a specific text block. "
                "The old_content must appear exactly once in the file. "
                "Creates a .bak backup before modifying. Returns a unified diff of the change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to patch"},
                    "old_content": {"type": "string", "description": "Exact text to find (must appear exactly once)"},
                    "new_content": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_content", "new_content"],
            },
        },
    },
]


MAX_READ_BYTES = 100_000
MAX_LIST_ENTRIES = 200


class ToolExecutor:
    def __init__(self, client: AntiSlopClient, working_dir: str = "."):
        self.client = client
        self.working_dir = Path(working_dir).resolve()

    def _resolve(self, path_str: str) -> Path:
        p = (self.working_dir / path_str).resolve()
        if not str(p).startswith(str(self.working_dir)):
            raise ValueError(f"path escapes working directory: {path_str}")
        return p

    async def execute(self, name: str, arguments_json: str) -> str:
        try:
            args = json.loads(arguments_json)
        except json.JSONDecodeError:
            args = {}

        match name:
            case "web_search":
                return await self._web_search(args)
            case "read_file":
                return self._read_file(args)
            case "list_files":
                return self._list_files(args)
            case "write_file":
                return self._write_file(args)
            case "patch_file":
                return self._patch_file(args)
            case _:
                return f"Unknown tool: {name}"

    async def _web_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return "Error: query is required"
        try:
            results = await self.client.search(query)
        except Exception as e:
            return f"Search error: {e}"
        if not results:
            return "No results found."
        out = []
        for r in results:
            out.append(f"{r.get('title', '')}\n{r.get('url', '')}\n{r.get('description', '')}")
        return "\n\n".join(out)

    def _read_file(self, args: dict) -> str:
        path_str = args.get("path", "")
        if not path_str:
            return "Error: path is required"
        try:
            p = self._resolve(path_str)
        except ValueError as e:
            return f"Error: {e}"
        if not p.exists():
            return f"Error: file not found: {path_str}"
        if p.is_dir():
            return f"Error: {path_str} is a directory, use list_files instead"
        size = p.stat().st_size
        if size > MAX_READ_BYTES:
            return f"Error: file is {size:,} bytes, exceeds {MAX_READ_BYTES:,} byte limit"
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

    def _list_files(self, args: dict) -> str:
        path_str = args.get("path", ".")
        pattern = args.get("pattern", "*")
        try:
            p = self._resolve(path_str)
        except ValueError as e:
            return f"Error: {e}"
        if not p.exists():
            return f"Error: directory not found: {path_str}"
        if not p.is_dir():
            return f"Error: {path_str} is not a directory"
        entries = []
        try:
            for i, item in enumerate(sorted(p.glob(pattern))):
                if i >= MAX_LIST_ENTRIES:
                    entries.append(f"... truncated at {MAX_LIST_ENTRIES} entries")
                    break
                rel = item.relative_to(self.working_dir)
                kind = "dir" if item.is_dir() else f"{item.stat().st_size:,}B"
                entries.append(f"  {rel}  ({kind})")
        except Exception as e:
            return f"Error listing files: {e}"
        if not entries:
            return f"No files matching '{pattern}' in {path_str}"
        return "\n".join(entries)

    def _write_file(self, args: dict) -> str:
        path_str = args.get("path", "")
        content = args.get("content", "")
        if not path_str:
            return "Error: path is required"
        try:
            p = self._resolve(path_str)
        except ValueError as e:
            return f"Error: {e}"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Wrote {len(content):,} chars to {path_str}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _patch_file(self, args: dict) -> str:
        path_str = args.get("path", "")
        old_content = args.get("old_content", "")
        new_content = args.get("new_content", "")
        if not path_str:
            return "Error: path is required"
        if not old_content:
            return "Error: old_content is required"
        try:
            p = self._resolve(path_str)
        except ValueError as e:
            return f"Error: {e}"
        if not p.exists():
            return f"Error: file not found: {path_str}"

        try:
            original = p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"

        count = original.count(old_content)
        if count == 0:
            return "Error: old_content not found in file"
        if count > 1:
            return f"Error: old_content appears {count} times, must be unique. Provide more context."

        # create backup
        bak = p.with_suffix(p.suffix + ".bak")
        try:
            bak.write_text(original, encoding="utf-8")
        except Exception:
            pass  # best effort backup

        patched = original.replace(old_content, new_content, 1)

        try:
            p.write_text(patched, encoding="utf-8")
        except Exception as e:
            return f"Error writing patched file: {e}"

        # generate diff
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{path_str}",
            tofile=f"b/{path_str}",
        )
        diff_text = "".join(diff)
        if len(diff_text) > 3000:
            diff_text = diff_text[:3000] + "\n... (diff truncated)"

        return f"Patched {path_str} (backup: {bak.name})\n\n{diff_text}"
