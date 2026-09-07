from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from simple_agent_base.types import (
    ChatMessage,
    ConversationItem,
    FilePart,
    ImagePart,
    JSONObject,
    MessageInput,
    TextPart,
    ToolExecutionResult,
)


def clean_system_prompt(system_prompt: str | None) -> str | None:
    if system_prompt is None:
        return None
    return system_prompt.strip() or None


def normalize_input(input_data: str | Sequence[MessageInput]) -> list[ConversationItem]:
    if isinstance(input_data, str):
        return [user_message(input_data)]

    items: list[ConversationItem] = []
    for message in input_data:
        if isinstance(message, str):
            items.append(user_message(message))
        else:
            chat_message = ChatMessage.model_validate(message)
            items.append(message_to_item(chat_message))

    return items


def build_transcript(
    input_data: str | Sequence[MessageInput],
    *,
    system_prompt: str | None,
    prefix_items: Sequence[ConversationItem] | None = None,
) -> list[ConversationItem]:
    transcript = list(prefix_items or [])
    transcript.extend(normalize_input(input_data))
    if system_prompt is None:
        return transcript
    return [
        message_to_item(ChatMessage(role="developer", content=system_prompt)),
        *transcript,
    ]


def persist_chat_items(
    transcript: Sequence[ConversationItem],
    *,
    system_prompt: str | None,
) -> list[ConversationItem]:
    items = list(transcript)
    if system_prompt is not None:
        expected_item = message_to_item(ChatMessage(role="developer", content=system_prompt))
        if items and items[0] == expected_item:
            items = items[1:]
    return [item for item in items if item.get("type") == "message"]


def user_message(prompt: str) -> ConversationItem:
    return message_to_item(ChatMessage(role="user", content=prompt))


def tool_output_item(result: ToolExecutionResult) -> ConversationItem:
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": result.output,
    }


def message_to_item(message: ChatMessage) -> ConversationItem:
    if isinstance(message.content, str):
        content: str | list[JSONObject] = message.content
    else:
        content = [content_part_to_item(part) for part in message.content]

    return {
        "type": "message",
        "role": message.role,
        "content": content,
    }


def messages_from_items(items: Sequence[ConversationItem]) -> list[ChatMessage]:
    return [message for item in items if (message := message_from_item(item)) is not None]


def message_from_item(item: ConversationItem) -> ChatMessage | None:
    if item.get("type") != "message":
        return None

    role = item.get("role")
    if not isinstance(role, str):
        return None

    content = message_content_from_item(item.get("content", []))
    if content is None:
        return None

    return ChatMessage(role=role, content=content)


def message_content_from_item(
    content_value: object,
) -> str | list[TextPart | ImagePart | FilePart] | None:
    if isinstance(content_value, str):
        return content_value

    if not isinstance(content_value, list):
        return None

    content_parts: list[TextPart | ImagePart | FilePart] = []
    saw_rich_content = False

    for block in content_value:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type in {"input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                content_parts.append(TextPart(text))
        elif block_type == "input_image":
            image_url = block.get("image_url")
            detail = block.get("detail", "auto")
            if isinstance(image_url, str) and isinstance(detail, str):
                content_parts.append(ImagePart(image_url=image_url, detail=detail))
                saw_rich_content = True
        elif block_type == "input_file":
            try:
                file_part = FilePart.model_validate(
                    {
                        "file_url": block.get("file_url"),
                        "file_data": block.get("file_data"),
                        "filename": block.get("filename"),
                    }
                )
            except ValidationError:
                pass
            else:
                content_parts.append(file_part)
                saw_rich_content = True

    if saw_rich_content and content_parts:
        return content_parts
    text = "".join(part.text for part in content_parts if isinstance(part, TextPart))
    if text:
        return text
    return None

def content_part_to_item(part: TextPart | ImagePart | FilePart) -> JSONObject:
    if isinstance(part, TextPart):
        return {
            "type": "input_text",
            "text": part.text,
        }

    if isinstance(part, FilePart):
        item: JSONObject = {"type": "input_file"}
        if part.file_url is not None:
            item["file_url"] = part.file_url
        if part.file_data is not None:
            item["file_data"] = part.file_data
        if part.filename is not None:
            item["filename"] = part.filename
        return item

    return {
        "type": "input_image",
        "image_url": part.image_url,
        "detail": part.detail,
    }
