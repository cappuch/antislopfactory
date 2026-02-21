"""Coding agent: interactive REPL with streaming chat and tool execution."""

from __future__ import annotations

import asyncio
import json as _json
import os
import sys

from dotenv import load_dotenv

from .client import AntiSlopClient, Classification
from .tools import TOOL_DEFINITIONS, ToolExecutor
from util.tool_classifier import classify_tool

# ── ANSI colors ───────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"

UNSAFE_LABELS = frozenset({
    "violence", "hate", "harassment", "sexual", "self_harm",
    "criminal", "malware", "illegal", "fraud", "harmful",
    "unethical", "unsafe_other",
})

SYSTEM_PROMPT = """\
You are a coding assistant with access to the local filesystem, web search, and bash.

Available tools:
- web_search: search the web for current information
- read_file: read file contents
- list_files: list directory contents with optional glob pattern
- write_file: create or overwrite files
- patch_file: safely edit files with find-and-replace (preferred for existing files)
- run_bash: execute bash commands (scripts, git, builds, tests, package management, etc.)

When modifying code:
1. First read the relevant files to understand the current state
2. Use patch_file for surgical edits (preferred) or write_file for new files
3. Explain what you changed and why

Use run_bash for running tests, installing dependencies, git operations, building projects, \
and other shell tasks. Always prefer patch_file over write_file for existing files."""


# ── display helpers ───────────────────────────────────────────────────

def _print_token(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _print_classification(clf: Classification) -> None:
    if clf.final:
        label = clf.label
        risk = clf.risk
        if risk >= 6:
            color = RED
        elif risk >= 3:
            color = YELLOW
        else:
            color = GREEN
        sys.stdout.write(f"\n{DIM}[{color}{label} risk:{risk}/10{RESET}{DIM}]{RESET}\n")
        sys.stdout.flush()


def _print_tool_call(name: str, arguments: str) -> None:
    try:
        pretty = __import__("json").dumps(__import__("json").loads(arguments), indent=2)
    except Exception:
        pretty = arguments
    print(f"\n{BLUE}{BOLD}tool: {name}{RESET}")
    print(f"{DIM}{pretty}{RESET}")


def _print_tool_result(name: str, content: str) -> None:
    preview = content[:500] + "..." if len(content) > 500 else content
    print(f"{GREEN}result ({name}):{RESET}")
    print(f"{DIM}{preview}{RESET}\n")


# ── agent ─────────────────────────────────────────────────────────────

class CodingAgent:
    def __init__(self, base_url: str = "http://localhost:8000", working_dir: str = "."):
        self.client = AntiSlopClient(base_url)
        self.tools = ToolExecutor(self.client, working_dir)
        self.thread_id: str | None = None

    async def start(self) -> None:
        health = await self.client.health()
        if health.get("status") != "ok":
            raise ConnectionError("API health check failed")
        self.thread_id = await self.client.create_thread(SYSTEM_PROMPT)
        print(f"{BOLD}antislopfactory coding agent{RESET}")
        print(f"{DIM}thread: {self.thread_id}{RESET}")
        print(f"{DIM}tools: web_search, read_file, list_files, write_file, patch_file, run_bash{RESET}")
        print(f"{DIM}type /quit to exit, /help for commands{RESET}\n")

    async def _classify_and_approve(self, name: str, arguments: str) -> bool:
        """Classify a tool call and handle approval. Returns True if approved."""
        try:
            args_pretty = _json.dumps(_json.loads(arguments), indent=2)
        except Exception:
            args_pretty = arguments

        tool_desc = f"Tool: {name}\nArguments:\n{args_pretty}"

        try:
            raw = await asyncio.to_thread(classify_tool, tool_desc)
            parsed = _json.loads(raw)
            classification = parsed.get("classification", "approval_required").strip()
            thinking = parsed.get("thinking", "")
        except Exception as e:
            print(f"{DIM}[classifier error: {e}, defaulting to approval_required]{RESET}")
            classification = "approval_required"
            thinking = ""

        if classification == "safe":
            print(f"{DIM}[{GREEN}safe{RESET}{DIM}]{RESET}")
            return True

        if classification == "unsafe":
            print(f"\n{RED}{BOLD}  THIS COMMAND IS UNSAFE.{RESET}")
            if thinking:
                print(f"{DIM}  reason: {thinking}{RESET}")
            try:
                response = await asyncio.to_thread(
                    input, f"{RED}  execute anyway? (y/N): {RESET}",
                )
            except (EOFError, KeyboardInterrupt):
                return False
            return response.strip().lower() in ("y", "yes")

        # approval_required (or unknown)
        print(f"{YELLOW}[approval required]{RESET}")
        if thinking:
            print(f"{DIM}  reason: {thinking}{RESET}")
        try:
            response = await asyncio.to_thread(
                input, f"{YELLOW}  allow? (Y/n): {RESET}",
            )
        except (EOFError, KeyboardInterrupt):
            return False
        return response.strip().lower() not in ("n", "no")

    async def run_turn(self, message: str | None, depth: int = 0) -> None:
        if depth >= 15:
            print(f"{RED}Tool loop exceeded 15 iterations, stopping.{RESET}")
            return

        result = await self.client.chat_stream(
            thread_id=self.thread_id,
            message=message,
            tools=TOOL_DEFINITIONS,
            on_token=_print_token,
            on_classification=_print_classification,
        )

        if result.content and not result.final_classification:
            sys.stdout.write("\n")

        if not result.tool_calls:
            return

        # execute tool calls with classification gate
        tool_results = []
        for tc in result.tool_calls:
            _print_tool_call(tc.name, tc.arguments)

            approved = await self._classify_and_approve(tc.name, tc.arguments)
            if not approved:
                output = "Tool execution denied by user."
                print(f"{RED}  denied{RESET}\n")
            else:
                try:
                    output = await self.tools.execute(tc.name, tc.arguments)
                except Exception as e:
                    output = f"Error: {e}"
                _print_tool_result(tc.name, output)

            tool_results.append({"tool_call_id": tc.id, "content": output})

        # submit results and continue
        print(f"{DIM}submitting tool results...{RESET}")
        await self.client.submit_tool_results(self.thread_id, tool_results)
        await self.run_turn(message=None, depth=depth + 1)


# ── REPL ──────────────────────────────────────────────────────────────

def _print_help() -> None:
    print(f"""
{BOLD}commands:{RESET}
  /quit, /exit, /q   exit the agent
  /thread             show current thread ID
  /help               show this help
""")


async def main() -> None:
    load_dotenv()
    base_url = os.environ.get("ANTISLOPFACTORY_URL", "http://localhost:8000")
    working_dir = os.environ.get("WORKING_DIR", os.getcwd())

    agent = CodingAgent(base_url=base_url, working_dir=working_dir)

    try:
        await agent.start()
    except Exception as e:
        print(f"{RED}Failed to connect: {e}{RESET}")
        sys.exit(1)

    while True:
        try:
            user_input = await asyncio.to_thread(input, f"{CYAN}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break

        text = user_input.strip()
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            print("bye!")
            break
        if text == "/help":
            _print_help()
            continue
        if text == "/thread":
            print(f"{DIM}{agent.thread_id}{RESET}")
            continue

        try:
            await agent.run_turn(text)
        except KeyboardInterrupt:
            print(f"\n{DIM}(interrupted){RESET}")
        except Exception as e:
            print(f"{RED}Error: {e}{RESET}")
