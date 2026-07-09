from __future__ import annotations

import argparse
import asyncio
import socket
import subprocess
import sys
import tempfile
import struct
import time
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import BaseModel

from simple_agent_base import (
    Agent,
    AgentConfig,
    ChatMessage,
    FilePart,
    ImagePart,
    MCPApprovalRequest,
    MCPServer,
    TextPart,
    tool,
)
from simple_agent_base.config import ReasoningEffort

class Person(BaseModel):
    name: str
    age: int


class ColorAnswer(BaseModel):
    dominant_color: str


class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "fail", "skip"]
    error: str | None = None


PARALLEL_TOOL_EVENTS: list[tuple[str, str, float]] = []


class SkippedCheck(RuntimeError):
    """Raised when the configured backend cannot support a specific live check."""


@tool
async def ping_tool(message: str) -> str:
    """Echo a message back with a fixed prefix."""
    return f"pong: {message}"


@tool
def sync_ping_tool(message: str) -> str:
    """Echo a message back synchronously with a fixed prefix."""
    return f"sync-pong: {message}"


@tool
async def slow_tool_a(message: str) -> str:
    """Return a labeled message after a short delay."""
    PARALLEL_TOOL_EVENTS.append(("slow_tool_a", "start", time.perf_counter()))
    await asyncio.sleep(0.75)
    PARALLEL_TOOL_EVENTS.append(("slow_tool_a", "end", time.perf_counter()))
    return f"a:{message}"


@tool
async def slow_tool_b(message: str) -> str:
    """Return a labeled message after a short delay."""
    PARALLEL_TOOL_EVENTS.append(("slow_tool_b", "start", time.perf_counter()))
    await asyncio.sleep(0.75)
    PARALLEL_TOOL_EVENTS.append(("slow_tool_b", "end", time.perf_counter()))
    return f"b:{message}"


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_config(
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> AgentConfig:
    return AgentConfig(temperature=0, timeout=120, max_turns=6, reasoning_effort=reasoning_effort)


def make_demo_mcp_server(*, require_approval: bool) -> MCPServer:
    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mcp_demo_server.py"
    ensure(fixture_path.exists(), f"Missing MCP fixture server: {fixture_path}")
    return MCPServer.stdio(
        name="demo",
        command=sys.executable,
        args=[str(fixture_path), "stdio"],
        require_approval=require_approval,
    )


def make_red_png_file() -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="simple-agent-base-live-", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.write(make_solid_png(64, 64, red=255, green=0, blue=0))
    handle.close()
    return path


def make_pdf_file(text: str) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="simple-agent-base-live-", suffix=".pdf", delete=False)
    path = Path(handle.name)
    handle.write(make_simple_pdf(text))
    handle.close()
    return path


def make_solid_png(width: int, height: int, *, red: int, green: int, blue: int) -> bytes:
    png_header = b"\x89PNG\r\n\x1a\n"

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes([red, green, blue]) * width
    raw = row * height
    idat = zlib.compress(raw)
    return png_header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def make_simple_pdf(text: str) -> bytes:
    escaped_text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content_stream = f"BT /F1 18 Tf 72 720 Td ({escaped_text}) Tj ET".encode("latin-1")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n%s\nendstream" % (len(content_stream), content_stream),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    parts = [b"%PDF-1.4\n"]
    offsets: list[int] = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(sum(len(part) for part in parts))
        parts.append(f"{index} 0 obj\n".encode("ascii"))
        parts.append(obj)
        parts.append(b"\nendobj\n")

    xref_offset = sum(len(part) for part in parts)
    parts.append(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    parts.append(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        parts.append(f"{offset:010d} 00000 n \n".encode("ascii"))
    parts.append(
        b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF"
        % (len(objects) + 1, xref_offset)
    )

    return b"".join(parts)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_server(host: str, port: int) -> None:
    deadline = time.time() + 10
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.1)

    raise RuntimeError(f"HTTP MCP fixture did not start: {last_error}")


@contextmanager
def demo_mcp_http_server() -> Iterator[MCPServer]:
    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "mcp_demo_server.py"
    ensure(fixture_path.exists(), f"Missing MCP fixture server: {fixture_path}")
    port = _free_port()
    process = subprocess.Popen(
        [sys.executable, str(fixture_path), "http", str(port)],
        cwd=str(fixture_path.parent.parent.parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_http_server("127.0.0.1", port)
        yield MCPServer.http(name="demohttp", url=f"http://127.0.0.1:{port}/mcp")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def print_config_summary() -> None:
    config = make_config()
    print("Runtime configuration:")
    print(f"- model: {config.model}")
    print(f"- base_url: {config.base_url}")
    print(f"- reasoning_effort: {config.reasoning_effort}")
    print(f"- temperature: {config.temperature}")
    print(f"- timeout: {config.timeout}")
    print(f"- max_turns: {config.max_turns}")
    print("- api_key: provided via environment")


def describe_message_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [f"text: {content}"]

    lines: list[str] = []
    if isinstance(content, list):
        for index, part in enumerate(content, start=1):
            if isinstance(part, TextPart):
                lines.append(f"part {index}: text -> {part.text}")
            elif isinstance(part, ImagePart):
                if part.image_url.startswith("data:"):
                    preview = f"data-url ({len(part.image_url)} chars)"
                else:
                    preview = part.image_url
                lines.append(f"part {index}: image -> {preview} [detail={part.detail}]")
            elif isinstance(part, FilePart):
                if part.file_url is not None:
                    preview = part.file_url
                elif part.file_data is not None:
                    preview = f"data-url ({len(part.file_data)} chars)"
                else:
                    preview = "unknown"
                filename_suffix = f" [filename={part.filename}]" if part.filename else ""
                lines.append(f"part {index}: file -> {preview}{filename_suffix}")
            else:
                lines.append(f"part {index}: raw -> {part}")
    return lines


def print_input(label: str, input_data: str | list[ChatMessage]) -> None:
    print(label)
    if isinstance(input_data, str):
        print(f"- user: {input_data}")
        return

    for index, message in enumerate(input_data, start=1):
        print(f"- message {index}: role={message.role}")
        for line in describe_message_content(message.content):
            print(f"  {line}")


def print_result_details(result: Any) -> None:
    print("Result summary:")
    print(f"- response_id: {result.response_id}")
    print(f"- output_text: {result.output_text}")
    print(f"- reasoning_summary: {getattr(result, 'reasoning_summary', None)}")
    print(f"- structured_output: {result.output_data}")
    print(f"- tool_result_count: {len(result.tool_results)}")
    if result.tool_results:
        for tool_result in result.tool_results:
            print(
                f"  - tool {tool_result.name}("
                f"arguments={tool_result.arguments}) -> {tool_result.output}"
            )
    mcp_calls = getattr(result, "mcp_calls", [])
    print(f"- mcp_call_count: {len(mcp_calls)}")
    for mcp_call in mcp_calls:
        output_preview = (mcp_call.output or "")[:120]
        print(
            f"  - mcp {mcp_call.server_name}.{mcp_call.name}("
            f"arguments={mcp_call.arguments}) -> {output_preview!r}"
            + (f" error={mcp_call.error!r}" if mcp_call.error else "")
        )


def print_chat_history(chat_history: list[ChatMessage]) -> None:
    print("Chat history:")
    for index, message in enumerate(chat_history, start=1):
        print(f"- history message {index}: role={message.role}")
        for line in describe_message_content(message.content):
            print(f"  {line}")


async def run_check(name: str, fn: Any) -> CheckResult:
    section(name)
    try:
        await fn()
    except SkippedCheck as exc:
        print("STATUS: SKIP")
        print(f"REASON: {exc}")
        return CheckResult(name=name, status="skip", error=str(exc))
    except Exception as exc:
        print("STATUS: FAIL")
        print(f"ERROR: {exc}")
        return CheckResult(name=name, status="fail", error=str(exc))

    print("STATUS: PASS")
    return CheckResult(name=name, status="pass")


async def run_plain_text() -> None:
    agent = Agent(config=make_config())
    prompt = "Reply with exactly the text plain-text-ok."
    try:
        print_input("Sending prompt:", prompt)
        result = await agent.run(prompt)
        print_result_details(result)
        ensure("plain-text-ok" in result.output_text.lower(), "Plain text response did not contain expected text.")
    finally:
        await agent.aclose()


async def run_structured_output() -> None:
    agent = Agent(config=make_config())
    prompt = "Return a person record with the name Sarah and the age 29."
    try:
        print_input("Sending prompt:", prompt)
        print("Requested response_model: Person")
        result = await agent.run(
            prompt,
            response_model=Person,
        )
        print_result_details(result)
        ensure(result.output_data is not None, "Structured output was not returned.")
        ensure(result.output_data.name.lower() == "sarah", "Structured output name was incorrect.")
        ensure(result.output_data.age == 29, "Structured output age was incorrect.")
    finally:
        await agent.aclose()


async def run_system_prompt() -> None:
    agent = Agent(
        config=make_config(),
        system_prompt="For the next reply, answer with exactly the text system-default-ok.",
    )
    default_prompt = "What should you reply with?"
    override_prompt = "What should you reply with now?"
    try:
        print("Agent default system_prompt:")
        print("- For the next reply, answer with exactly the text system-default-ok.")
        print_input("Sending prompt:", default_prompt)
        default_result = await agent.run(default_prompt)
        print_result_details(default_result)
        ensure(
            "system-default-ok" in default_result.output_text.lower(),
            "Agent-level system prompt did not steer the response as expected.",
        )

        print()
        print("Per-run override system_prompt:")
        print("- For the next reply, answer with exactly the text system-override-ok.")
        print_input("Sending prompt:", override_prompt)
        override_result = await agent.run(
            override_prompt,
            system_prompt="For the next reply, answer with exactly the text system-override-ok.",
        )
        print_result_details(override_result)
        ensure(
            "system-override-ok" in override_result.output_text.lower(),
            "Per-run system prompt override did not steer the response as expected.",
        )

        print()
        print("Chat session default system_prompt:")
        print("- For every reply in this chat, answer with exactly the text chat-system-ok.")
        chat = agent.chat(system_prompt="For every reply in this chat, answer with exactly the text chat-system-ok.")
        chat_prompt = "What should you reply with in this chat?"
        print_input("Sending prompt:", chat_prompt)
        chat_result = await chat.run(chat_prompt)
        print_result_details(chat_result)
        print_chat_history(chat.history)
        ensure(
            "chat-system-ok" in chat_result.output_text.lower(),
            "Chat session system prompt did not steer the response as expected.",
        )
        ensure(
            all(message.role != "developer" for message in chat.history),
            "System prompt leaked into chat history.",
        )
    finally:
        await agent.aclose()


async def run_tool_call() -> None:
    agent = Agent(config=make_config(), tools=[ping_tool])
    prompt = "Call the ping_tool with the message 'live-test'. Then tell me the tool result."
    try:
        print_input("Sending prompt:", prompt)
        print("Registered tools:")
        print("- ping_tool(message: str) -> str")
        result = await agent.run(prompt)
        print_result_details(result)
        ensure(len(result.tool_results) >= 1, "Expected at least one tool result.")
        ensure(result.tool_results[0].name == "ping_tool", "Unexpected tool name.")
        ensure(result.tool_results[0].output == "pong: live-test", "Unexpected tool output.")
    finally:
        await agent.aclose()


def _run_sync_usage_blocking() -> None:
    agent = Agent(config=make_config(), tools=[sync_ping_tool])
    try:
        prompt = "Call sync_ping_tool with the message 'sync-live-test'. Then tell me the tool result."
        print_input("Sending sync prompt:", prompt)
        print("Registered sync tools:")
        print("- sync_ping_tool(message: str) -> str")
        result = agent.run_sync(prompt)
        print_result_details(result)
        ensure(len(result.tool_results) == 1, "Expected one sync tool result.")
        ensure(result.tool_results[0].output == "sync-pong: sync-live-test", "Unexpected sync tool output.")

        print()
        streaming_prompt = "Explain sync wrappers in one short sentence."
        print_input("Sending sync streaming prompt:", streaming_prompt)
        print("Sync streaming events:")
        saw_delta = False
        final_text = None
        for event in agent.stream_sync(streaming_prompt):
            if event.type == "text_delta" and event.delta:
                saw_delta = True
                print(f"- delta: {event.delta}")
            elif event.type == "completed" and event.result is not None:
                final_text = event.result.output_text
                print("Completed event received.")
                print_result_details(event.result)
        ensure(saw_delta, "Sync streaming did not emit any text deltas.")
        ensure(final_text is not None and final_text.strip(), "Sync streaming did not produce a final result.")
    finally:
        agent.close()


async def run_sync_usage() -> None:
    await asyncio.to_thread(_run_sync_usage_blocking)


async def run_parallel_tool_calls() -> None:
    agent = Agent(
        config=make_config().model_copy(update={"parallel_tool_calls": True}),
        tools=[slow_tool_a, slow_tool_b],
    )
    prompt = (
        "Call slow_tool_a with the message 'alpha' and slow_tool_b with the message 'beta'. "
        "Then tell me both tool results."
    )
    try:
        print_input("Sending prompt:", prompt)
        print("Registered tools:")
        print("- slow_tool_a(message: str) -> str")
        print("- slow_tool_b(message: str) -> str")
        print("- parallel_tool_calls: enabled")
        PARALLEL_TOOL_EVENTS.clear()
        start = time.perf_counter()
        result = await agent.run(prompt)
        elapsed = time.perf_counter() - start
        print_result_details(result)
        print(f"- elapsed_seconds: {elapsed:.3f}")
        print(f"- tool_events: {PARALLEL_TOOL_EVENTS}")
        ensure(len(result.tool_results) == 2, "Expected exactly two tool results in the parallel batch.")
        ensure(result.tool_results[0].output == "a:alpha", "Unexpected output from slow_tool_a.")
        ensure(result.tool_results[1].output == "b:beta", "Unexpected output from slow_tool_b.")
        ensure(len(PARALLEL_TOOL_EVENTS) == 4, "Did not capture the expected tool execution events.")
        event_lookup = {(tool_name, phase): timestamp for tool_name, phase, timestamp in PARALLEL_TOOL_EVENTS}
        overlap_span = max(
            event_lookup[("slow_tool_a", "end")],
            event_lookup[("slow_tool_b", "end")],
        ) - min(
            event_lookup[("slow_tool_a", "start")],
            event_lookup[("slow_tool_b", "start")],
        )
        print(f"- observed_tool_batch_seconds: {overlap_span:.3f}")
        ensure(overlap_span < 1.25, "Tool batch did not overlap enough to indicate parallel execution.")
    finally:
        await agent.aclose()


async def run_chat_history() -> None:
    agent = Agent(config=make_config())
    chat = agent.chat()
    first_prompt = "My favorite color is teal. Remember it for later."
    second_prompt = "What is my favorite color? Reply with the color only."
    try:
        print_input("First turn:", first_prompt)
        first = await chat.run(first_prompt)
        print_result_details(first)
        print_input("Second turn:", second_prompt)
        second = await chat.run(second_prompt)
        print_result_details(second)
        print_chat_history(chat.history)
        ensure("teal" in second.output_text.lower(), "Chat history was not preserved across turns.")
        ensure(len(chat.history) >= 4, "Chat history did not retain both turns.")
    finally:
        await agent.aclose()


async def run_chat_persistence() -> None:
    agent = Agent(config=make_config())
    chat = agent.chat(system_prompt="You are concise.")
    first_prompt = "My name is Anson."
    second_prompt = "My favorite color is teal."
    follow_up_prompt = "What is my favorite color? Reply with the color only."
    try:
        print("Original chat session system_prompt:")
        print("- You are concise.")
        print_input("First turn:", first_prompt)
        first = await chat.run(first_prompt)
        print_result_details(first)

        print_input("Second turn:", second_prompt)
        second = await chat.run(second_prompt)
        print_result_details(second)

        snapshot = chat.snapshot()
        payload = chat.export()

        print()
        print("Snapshot summary:")
        print(f"- version: {snapshot.version}")
        print(f"- system_prompt: {snapshot.system_prompt}")
        print(f"- item_count: {len(snapshot.items)}")
        print(f"- export_keys: {list(payload.keys())}")

        restored = agent.chat_from_snapshot(payload)
        print()
        print("Restored chat history:")
        print_chat_history(restored.history)

        print_input("Follow-up turn on restored chat:", follow_up_prompt)
        follow_up = await restored.run(follow_up_prompt)
        print_result_details(follow_up)

        ensure(
            snapshot.system_prompt == "You are concise.",
            "Snapshot did not preserve the chat session system prompt.",
        )
        ensure(
            len(snapshot.items) >= 4,
            "Snapshot did not preserve the completed chat turns.",
        )
        ensure(
            payload.get("version") == "v1",
            "Exported payload did not include the expected snapshot version.",
        )
        ensure(
            "teal" in follow_up.output_text.lower(),
            "Restored chat did not preserve the original conversation context.",
        )
        ensure(
            restored.history[: len(chat.history)] == chat.history,
            "Restored chat history did not match the original persisted history.",
        )
    finally:
        await agent.aclose()


async def run_file_input() -> None:
    file_path = make_pdf_file("Favorite color: teal")
    agent = Agent(config=make_config())
    input_data = [
        ChatMessage(
            role="user",
            content=[
                TextPart("Read this file and answer with the favorite color only."),
                FilePart.from_file(str(file_path)),
            ],
        )
    ]
    try:
        print(f"Created temporary file: {file_path}")
        print_input("Sending file input:", input_data)
        result = await agent.run(input_data)
        print_result_details(result)
        ensure("teal" in result.output_text.lower(), "File analysis did not identify teal from the PDF.")
    finally:
        await agent.aclose()
        file_path.unlink(missing_ok=True)


async def run_streaming() -> None:
    agent = Agent(config=make_config())
    saw_delta = False
    final_text = None
    prompt = "Explain async agents in one short sentence."
    try:
        print_input("Sending prompt:", prompt)
        print("Streaming deltas:")
        async for event in agent.stream(prompt):
            if event.type == "text_delta" and event.delta:
                saw_delta = True
                print(f"- delta: {event.delta}")
            elif event.type == "completed" and event.result is not None:
                final_text = event.result.output_text
                print("Completed event received.")
                print_result_details(event.result)
        ensure(saw_delta, "Streaming did not emit any text deltas.")
        ensure(final_text is not None and final_text.strip(), "Streaming did not produce a final result.")
    finally:
        await agent.aclose()


async def run_reasoning() -> None:
    agent = Agent(config=make_config(reasoning_effort="low"))
    saw_reasoning_delta = False
    final_result = None
    prompt = """
Alex, Blake, and Casey each own a different pet: a cat, a dog, or a bird.
One always tells the truth, one always lies, and one alternates between truth and lies, starting with a truthful statement.

Alex says: 'Blake owns the dog.'
Blake says: 'Casey does not own the bird.'
Casey says: 'I am the one who alternates truth and lies.'
Alex also says: 'I do not own the cat.'

Figure out who owns each pet and who has each truthfulness pattern.
Show your reasoning step by step and verify each statement against the final answer.
"""
    try:
        print_input("Sending prompt:", prompt)
        print("- reasoning_effort: low")
        print("Streaming deltas:")
        async for event in agent.stream(prompt):
            if event.type == "reasoning_delta" and event.delta:
                saw_reasoning_delta = True
                print(f"- reasoning: {event.delta}")
            elif event.type == "text_delta" and event.delta:
                print(f"- text: {event.delta}")
            elif event.type == "completed" and event.result is not None:
                final_result = event.result
                print("Completed event received.")
                print_result_details(event.result)
        ensure(saw_reasoning_delta, "Streaming did not emit any reasoning deltas.")
        ensure(final_result is not None, "Reasoning check did not produce a final result.")
        ensure(
            final_result.reasoning_summary is not None and final_result.reasoning_summary.strip(),
            "Reasoning check did not produce a final reasoning summary.",
        )
        output_text = (final_result.output_text or "").lower()
        ensure(output_text.strip(), "Reasoning response did not contain any final text.")
        ensure("alex" in output_text, "Reasoning response did not mention Alex.")
        ensure("blake" in output_text, "Reasoning response did not mention Blake.")
        ensure("casey" in output_text, "Reasoning response did not mention Casey.")
        ensure(
            any(pet in output_text for pet in ("cat", "dog", "bird")),
            "Reasoning response did not mention any of the pets.",
        )
        ensure(
            any(term in output_text for term in ("truth", "lie", "alternat")),
            "Reasoning response did not mention the truthfulness assignment.",
        )
    finally:
        await agent.aclose()


async def run_image_structured_output() -> None:
    image_path = make_red_png_file()
    agent = Agent(config=make_config())
    input_data = [
        ChatMessage(
            role="user",
            content=[
                TextPart(
                    "The attached image is a solid color square. "
                    "Choose the dominant color from red, green, blue, black, or white. "
                    "Return the answer in the structured output."
                ),
                ImagePart.from_file(str(image_path), detail="high"),
            ],
        )
    ]
    try:
        print(f"Created temporary image file: {image_path}")
        print_input("Sending multimodal input:", input_data)
        print("Requested response_model: ColorAnswer")
        result = await agent.run(
            input_data,
            response_model=ColorAnswer,
        )
        print_result_details(result)
        ensure(result.output_data is not None, "Image structured output was not returned.")
        ensure("red" in result.output_data.dominant_color.lower(), "Image analysis did not identify red.")
    finally:
        await agent.aclose()
        image_path.unlink(missing_ok=True)


async def run_chat_with_image_follow_up() -> None:
    image_path = make_red_png_file()
    agent = Agent(config=make_config())
    chat = agent.chat()
    first_input = [
        ChatMessage(
            role="user",
            content=[
                TextPart("Remember this image for a follow-up question."),
                ImagePart.from_file(str(image_path), detail="high"),
            ],
        )
    ]
    second_prompt = "What was the dominant color in the image? Choose from red, green, blue, black, or white. Reply with one word."
    try:
        print(f"Created temporary image file: {image_path}")
        print_input("First turn:", first_input)
        first = await chat.run(first_input)
        print_result_details(first)
        print_input("Second turn:", second_prompt)
        follow_up = await chat.run(second_prompt)
        print_result_details(follow_up)
        print_chat_history(chat.history)
        ensure(first.output_text.strip(), "Initial image chat turn returned an empty response.")
        ensure("red" in follow_up.output_text.lower(), "Image context was not preserved in chat history.")
        ensure(len(chat.history) >= 4, "Chat history did not preserve the image turn.")
    finally:
        await agent.aclose()
        image_path.unlink(missing_ok=True)


async def run_mcp_server() -> None:
    server = make_demo_mcp_server(require_approval=False)

    agent = Agent(
        config=make_config(),
        mcp_servers=[server],
    )
    prompt = (
        "Use the demo MCP server echo tool with the message 'live-mcp'. "
        "Then reply with one short sentence that includes the tool result."
    )
    try:
        print("MCP servers:")
        print(f"- demo -> stdio via {sys.executable} {server.args}")
        print_input("Sending prompt:", prompt)
        result = await agent.run(prompt)
        print_result_details(result)
        ensure(result.output_text.strip(), "MCP run returned empty output_text.")
        ensure(len(result.mcp_calls) >= 1, "Expected at least one mcp_call in result.mcp_calls.")
        ensure(
            all(call.server_name == "demo" for call in result.mcp_calls),
            "Unexpected mcp_call server_name.",
        )
    finally:
        await agent.aclose()


async def run_mcp_server_with_approval() -> None:
    approvals_seen: list[MCPApprovalRequest] = []

    def approve(request: MCPApprovalRequest) -> bool:
        approvals_seen.append(request)
        print(
            f"- approval requested: {request.server_name}.{request.name}("
            f"arguments={request.arguments})"
        )
        return True

    server = make_demo_mcp_server(require_approval=True)

    agent = Agent(
        config=make_config(),
        mcp_servers=[server],
        approval_handler=approve,
    )
    prompt = (
        "Use the demo MCP server add tool with 2 and 3, then reply with one short sentence."
    )
    try:
        print("MCP servers:")
        print(f"- demo -> stdio via {sys.executable} {server.args} (require_approval=True)")
        print_input("Sending prompt:", prompt)
        result = await agent.run(prompt)
        print_result_details(result)
        print(f"- approval_handler_invocations: {len(approvals_seen)}")
        ensure(result.output_text.strip(), "MCP approval run returned empty output_text.")
        ensure(len(approvals_seen) >= 1, "Expected the approval handler to be invoked at least once.")
        ensure(len(result.mcp_calls) >= 1, "Expected at least one mcp_call after approval.")
    finally:
        await agent.aclose()


async def run_mcp_http_server() -> None:
    with demo_mcp_http_server() as server:
        agent = Agent(
            config=make_config(),
            mcp_servers=[server],
        )
        prompt = (
            "Use the demohttp MCP server add tool with 2 and 3. "
            "Then reply with one short sentence that includes the result."
        )
        try:
            print("MCP servers:")
            print(f"- demohttp -> streamable_http via {server.url}")
            print_input("Sending prompt:", prompt)
            result = await agent.run(prompt)
            print_result_details(result)
            ensure(result.output_text.strip(), "HTTP MCP run returned empty output_text.")
            ensure(len(result.mcp_calls) >= 1, "Expected at least one mcp_call in result.mcp_calls.")
            ensure(
                all(call.server_name == "demohttp" for call in result.mcp_calls),
                "Unexpected HTTP mcp_call server_name.",
            )
            ensure(
                any(call.name == "add" and call.output == "5" for call in result.mcp_calls),
                "HTTP MCP run did not record the expected add tool result.",
            )
        finally:
            await agent.aclose()


async def run_mcp_streaming() -> None:
    await ensure_streaming_function_calls_supported()
    server = make_demo_mcp_server(require_approval=False)

    agent = Agent(
        config=make_config(),
        mcp_servers=[server],
    )
    prompt = (
        "You must call the demo MCP server echo tool with the message 'stream-mcp'. "
        "Do not answer until after you use the tool, then reply with one short sentence."
    )
    saw_mcp_started = False
    saw_mcp_completed = False
    final_result = None
    try:
        print("MCP servers:")
        print(f"- demo -> stdio via {sys.executable} {server.args}")
        print_input("Sending prompt:", prompt)
        print("Streaming events:")
        async for event in agent.stream(prompt):
            if event.type == "text_delta" and event.delta:
                print(f"- delta: {event.delta}")
            elif event.type == "mcp_call_started":
                saw_mcp_started = True
                mcp_call = event.mcp_call
                label = mcp_call.server_name if mcp_call else None
                name = mcp_call.name if mcp_call else None
                print(f"- mcp_call_started: {label}.{name}")
            elif event.type == "mcp_call_completed":
                saw_mcp_completed = True
                mcp_call = event.mcp_call
                label = mcp_call.server_name if mcp_call else None
                name = mcp_call.name if mcp_call else None
                print(f"- mcp_call_completed: {label}.{name}")
            elif event.type == "completed" and event.result is not None:
                final_result = event.result
                print("Completed event received.")
                print_result_details(event.result)
        ensure(saw_mcp_started, "Streaming did not emit mcp_call_started.")
        ensure(saw_mcp_completed, "Streaming did not emit mcp_call_completed.")
        ensure(final_result is not None, "Streaming did not emit a completed event.")
        ensure(
            final_result is not None and len(final_result.mcp_calls) >= 1,
            "Streaming final result did not include any mcp_calls.",
        )
    finally:
        await agent.aclose()


async def ensure_streaming_function_calls_supported() -> None:
    agent = Agent(config=make_config(), tools=[ping_tool])
    saw_tool_event = False
    try:
        async for event in agent.stream(
            "Call the tool named ping_tool with the message 'stream-capability'. Then say done."
        ):
            if event.type == "tool_call_completed":
                saw_tool_event = True
                break
    finally:
        await agent.aclose()

    if not saw_tool_event:
        raise SkippedCheck(
            "Configured backend did not issue function-tool calls during streaming, so MCP streaming "
            "cannot be validated in this environment."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live end-to-end checks.")
    parser.add_argument(
        "--mcp-only",
        action="store_true",
        help="Run only the client-side MCP live checks.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    print_config_summary()

    mcp_checks = [
        ("MCP Server", run_mcp_server),
        ("MCP Server With Approval", run_mcp_server_with_approval),
        ("MCP HTTP Server", run_mcp_http_server),
        ("MCP Streaming", run_mcp_streaming),
    ]
    checks: list[tuple[str, Any]] = (
        mcp_checks
        if args.mcp_only
        else [
            ("Plain Text", run_plain_text),
            ("Structured Output", run_structured_output),
            ("System Prompt", run_system_prompt),
            ("Sync Usage", run_sync_usage),
            ("Tool Call", run_tool_call),
            ("Parallel Tool Calls", run_parallel_tool_calls),
            ("Chat History", run_chat_history),
            ("Chat Persistence", run_chat_persistence),
            ("File Input", run_file_input),
            ("Streaming", run_streaming),
            ("Reasoning", run_reasoning),
            ("Image Structured Output", run_image_structured_output),
            ("Chat With Image Follow-Up", run_chat_with_image_follow_up),
            *mcp_checks,
        ]
    )

    results = [await run_check(name, fn) for name, fn in checks]

    section("Summary")
    passed = [result for result in results if result.status == "pass"]
    failed = [result for result in results if result.status == "fail"]
    skipped = [result for result in results if result.status == "skip"]
    print(f"Passed: {len(passed)}")
    print(f"Failed: {len(failed)}")
    print(f"Skipped: {len(skipped)}")
    for result in results:
        status = result.status.upper()
        suffix = "" if result.error is None else f" -> {result.error}"
        print(f"- {status}: {result.name}{suffix}")

    if failed:
        raise SystemExit(1)

    print("All live end-to-end checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
