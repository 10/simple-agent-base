from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from simple_agent_base.json_types import ConversationItem, JSONObject
from simple_agent_base.mcp import MCPApprovalRequest, MCPCallRecord


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: JSONObject
    func: Callable[..., object] = field(repr=False)
    arguments_model: type[BaseModel] = field(repr=False)
    is_async: bool = False


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: JSONObject = Field(default_factory=dict)
    raw_arguments: str = "{}"


class ToolExecutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_id: str
    name: str
    arguments: JSONObject = Field(default_factory=dict)
    output: str
    raw_output: object | None = None


ImageDetail = Literal["low", "high", "auto", "original"]

SUPPORTED_DOCUMENT_FILE_MIME_TYPES = {
    "application/csv",
    "application/json",
    "application/msword",
    "application/pdf",
    "application/rtf",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/xml",
    "application/yaml",
    "message/rfc822",
}

FALLBACK_DOCUMENT_FILE_MIME_TYPES = {
    ".csv": "text/csv",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".json": "application/json",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".pdf": "application/pdf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".rtf": "application/rtf",
    ".tsv": "text/tab-separated-values",
    ".txt": "text/plain",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def _guess_supported_file_mime_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    guessed_mime_type, _ = mimetypes.guess_type(file_path.name)

    if guessed_mime_type is not None and (
        guessed_mime_type.startswith("text/") or guessed_mime_type in SUPPORTED_DOCUMENT_FILE_MIME_TYPES
    ):
        return guessed_mime_type

    fallback = FALLBACK_DOCUMENT_FILE_MIME_TYPES.get(suffix)
    if fallback is not None:
        return fallback

    raise ValueError(
        "Unsupported file type for FilePart.from_file(...). "
        "Use a document or text file such as PDF, DOCX, XLSX, PPTX, CSV, TXT, JSON, or XML."
    )


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str

    def __init__(self, text: str | None = None, **data: object) -> None:
        if text is not None and "text" not in data:
            data["text"] = text
        super().__init__(**data)


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["image"] = "image"
    image_url: str
    detail: ImageDetail = "auto"

    @classmethod
    def from_url(cls, url: str, detail: ImageDetail = "auto") -> ImagePart:
        if not url.strip():
            raise ValueError("Image URL cannot be empty.")
        return cls(image_url=url, detail=detail)

    @classmethod
    def from_file(cls, path: str, detail: ImageDetail = "auto") -> ImagePart:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"Image file does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Image path is not a file: {file_path}")

        mime_type, _ = mimetypes.guess_type(file_path.name)
        supported_mime_types = {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        }
        if mime_type not in supported_mime_types:
            raise ValueError(
                "Unsupported image file type. Supported formats are PNG, JPEG, WEBP, and GIF."
            )

        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read image file: {file_path}") from exc

        encoded = base64.b64encode(raw_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        return cls(image_url=data_url, detail=detail)


class FilePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["file"] = "file"
    file_url: str | None = None
    file_data: str | None = None
    filename: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> FilePart:
        if (self.file_url is None) == (self.file_data is None):
            raise ValueError("FilePart requires exactly one of file_url or file_data.")

        if self.file_data is not None and not self.filename:
            raise ValueError("FilePart requires filename when using file_data.")

        return self

    @classmethod
    def from_url(cls, url: str) -> FilePart:
        if not url.strip():
            raise ValueError("File URL cannot be empty.")
        return cls(file_url=url)

    @classmethod
    def from_file(cls, path: str) -> FilePart:
        file_path = Path(path)
        if not file_path.exists():
            raise ValueError(f"File does not exist: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"File path is not a file: {file_path}")

        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Could not read file: {file_path}") from exc

        mime_type = _guess_supported_file_mime_type(file_path)
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        return cls(file_data=data_url, filename=file_path.name)


ContentPart = Annotated[TextPart | ImagePart | FilePart, Field(discriminator="type")]


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "developer", "system"]
    content: str | list[ContentPart]


MessageInput = ChatMessage | JSONObject


class ChatSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal["v1"] = "v1"
    items: list[ConversationItem] = Field(default_factory=list)
    system_prompt: str | None = None


class UsageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_tokens_details: JSONObject | None = None
    output_tokens_details: JSONObject | None = None
    raw: JSONObject | None = None


class HostedToolCallUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    tool_type: str
    status: str
    output_index: int | None = None
    sequence_number: int | None = None
    item: JSONObject | None = None


class AgentRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_text: str
    reasoning_summary: str | None = None
    output_data: BaseModel | None = None
    response_id: str | None = None
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    mcp_calls: list[MCPCallRecord] = Field(default_factory=list)
    usage: UsageMetadata | None = None
    usage_by_response: list[UsageMetadata] = Field(default_factory=list)
    raw_responses: list[JSONObject] = Field(default_factory=list)


class AgentEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "text_delta",
        "reasoning_delta",
        "tool_arguments_delta",
        "hosted_tool_call_started",
        "hosted_tool_call_updated",
        "hosted_tool_call_completed",
        "tool_call_started",
        "tool_call_completed",
        "mcp_call_started",
        "mcp_call_completed",
        "mcp_approval_requested",
        "completed",
    ]
    delta: str | None = None
    tool_item_id: str | None = None
    tool_name: str | None = None
    tool_call: ToolCallRequest | None = None
    tool_result: ToolExecutionResult | None = None
    mcp_call: MCPCallRecord | None = None
    mcp_approval: MCPApprovalRequest | None = None
    hosted_tool_call: HostedToolCallUpdate | None = None
    result: AgentRunResult | None = None
