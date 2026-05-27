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
    cleaned = system_prompt.strip()
    if not cleaned:
        return None
    return cleaned


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
    return prepend_system_prompt(transcript, system_prompt=system_prompt)


def persist_chat_items(
    transcript: Sequence[ConversationItem],
    *,
    system_prompt: str | None,
) -> list[ConversationItem]:
    return persistable_items(strip_prepended_system_prompt(transcript, system_prompt=system_prompt))


def user_message(prompt: str) -> ConversationItem:
    return message_to_item(ChatMessage(role="user", content=prompt))


def tool_output_item(result: ToolExecutionResult) -> ConversationItem:
    return {
        "type": "function_call_output",
        "call_id": result.call_id,
        "output": result.output,
    }


def prepend_system_prompt(
    items: list[ConversationItem],
    *,
    system_prompt: str | None,
) -> list[ConversationItem]:
    if system_prompt is None:
        return list(items)

    return [
        message_to_item(ChatMessage(role="developer", content=system_prompt)),
        *items,
    ]


def strip_prepended_system_prompt(
    items: Sequence[ConversationItem],
    *,
    system_prompt: str | None,
) -> list[ConversationItem]:
    if system_prompt is None:
        return list(items)

    expected_item = message_to_item(ChatMessage(role="developer", content=system_prompt))
    result = list(items)
    if result and result[0] == expected_item:
        return result[1:]
    return result


def persistable_items(items: Sequence[ConversationItem]) -> list[ConversationItem]:
    return [item for item in items if item.get("type") == "message"]


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
    messages: list[ChatMessage] = []

    for item in items:
        message = message_from_item(item)
        if message is not None:
            messages.append(message)

    return messages


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

    return _content_from_blocks(content_value)


def _content_from_blocks(blocks: list[object]) -> str | list[TextPart | ImagePart | FilePart] | None:
    text_parts: list[str] = []
    content_parts: list[TextPart | ImagePart | FilePart] = []
    saw_rich_content = False

    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type in {"input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
                content_parts.append(TextPart(text))
        elif block_type == "input_image":
            image = _image_part_from_block(block)
            if image is not None:
                content_parts.append(image)
                saw_rich_content = True
        elif block_type == "input_file":
            file_part = _file_part_from_block(block)
            if file_part is not None:
                content_parts.append(file_part)
                saw_rich_content = True

    if saw_rich_content and content_parts:
        return content_parts
    if text_parts:
        return "".join(text_parts)
    return None


def _image_part_from_block(block: JSONObject) -> ImagePart | None:
    image_url = block.get("image_url")
    detail = block.get("detail", "auto")
    if isinstance(image_url, str) and isinstance(detail, str):
        return ImagePart(image_url=image_url, detail=detail)
    return None


def _file_part_from_block(block: JSONObject) -> FilePart | None:
    file_payload = {
        "file_url": block.get("file_url"),
        "file_data": block.get("file_data"),
        "filename": block.get("filename"),
    }
    try:
        return FilePart.model_validate(file_payload)
    except ValidationError:
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
