# Structured Output Guide

This guide explains how typed outputs work in `simple-agent-base`.

Use structured output when you want the final model response validated into a Pydantic object instead of treating it as plain text.

## Mental Model

Structured output in this package is not a separate workflow. It is the same `run(...)` or `stream(...)` API with one extra argument:

- `response_model=MyPydanticModel`

The built-in OpenAI provider passes that model to the Responses API and returns both:

- the raw final text
- the parsed Pydantic object

The parsed object appears in:

- `AgentRunResult.output_data`
- the `completed` event result in streamed requests

## Basic Example

```python
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


result = await agent.run(
    "Extract the person from: Sarah is 29 years old.",
    response_model=Person,
)

print(result.output_text)
print(result.output_data)
```

Behavior:

- `result.output_text` contains the final assistant text returned by the provider
- `result.output_data` contains the parsed `Person` object

## Where You Can Use It

Structured output works with:

- `agent.run(...)`
- `agent.stream(...)`
- `chat.run(...)`
- `chat.stream(...)`
- tool calls during a run
- multimodal input

It also works with the sync wrappers:

- `agent.run_sync(...)`
- `agent.stream_sync(...)`
- `chat.run_sync(...)`
- `chat.stream_sync(...)`

## Response Model Requirements

Pass a Pydantic model class:

```python
from pydantic import BaseModel


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str
```

Then:

```python
result = await agent.run(
    "Return the weather for San Francisco.",
    response_model=WeatherAnswer,
)
```

The package stores the resulting object in `output_data`.

## Structured Output With Tools

Structured output works across tool turns.

Example:

```python
from pydantic import BaseModel
from simple_agent_base import Agent, AgentConfig, tool


class WeatherAnswer(BaseModel):
    city: str
    temperature_f: int
    summary: str


@tool
async def get_weather(city: str) -> str:
    """Return weather information for a city as JSON text."""
    return '{"city":"San Francisco","temperature_f":65,"summary":"Foggy"}'


agent = Agent(
    config=AgentConfig(model="gpt-6-astra"),
    tools=[get_weather],
)

result = await agent.run(
    "Use the weather tool for San Francisco and return a structured weather answer.",
    response_model=WeatherAnswer,
)
```

Important behavior:

- the same `response_model` is passed on every provider turn during the run
- this includes the initial request and later turns after tool outputs are appended
- the final answer still ends up in `output_data`

## Structured Output In Streams

Streaming still emits normal text deltas.

Example:

```python
from pydantic import BaseModel


class Summary(BaseModel):
    title: str
    bullets: list[str]


async for event in agent.stream(
    "Summarize this text.",
    response_model=Summary,
):
    if event.type == "text_delta" and event.delta:
        print(event.delta, end="")
    elif event.type == "completed" and event.result is not None:
        print()
        print(event.result.output_data)
```

### Important Streaming Behavior

- partial text arrives through `text_delta`
- the parsed object is not available during deltas
- the parsed object appears only on the final `completed` event
- if the run includes tool turns, structured output still appears only on the final `completed` event

## Structured Output With Multimodal Input

You can combine typed output with images or files.

Example with an image:

```python
from pydantic import BaseModel
from simple_agent_base import ChatMessage, ImagePart, TextPart


class ColorAnswer(BaseModel):
    dominant_color: str


result = await agent.run(
    [
        ChatMessage(
            role="user",
            content=[
                TextPart("What is the dominant color in this image?"),
                ImagePart.from_file("square.png", detail="high"),
            ],
        )
    ],
    response_model=ColorAnswer,
)
```

The output is still available in:

- `result.output_text`
- `result.output_data`

## Parse Failures

If the provider cannot satisfy the structured output request, behavior depends on the API you used.

### `run()` And `run_sync()`

These raise an exception, typically `ProviderError`.

### `stream()` And `stream_sync()`

These also raise an exception, typically `ProviderError`.

## Raw Text Still Exists

Structured output does not replace the final text field.

Even when parsing succeeds:

- `output_text` still contains the assistant text
- `output_data` contains the parsed Pydantic object

This is useful when you want:

- typed data for application logic
- raw text for logs or debugging

## Chat Sessions With Structured Output

You can use structured output inside `ChatSession`.

Example:

```python
chat = agent.chat()

first = await chat.run(
    "Extract a person from: Sarah is 29 years old.",
    response_model=Person,
)

second = await chat.run(
    "Extract a person from: Ada is 35 years old.",
    response_model=Person,
)
```

The typed result applies to that one call. The chat session still stores conversation history normally.

## How The Built-In Provider Implements It

The OpenAI provider does this:

- if `response_model` is `None`, it calls `responses.create(...)`
- if `response_model` is set, it calls `responses.parse(...)`
- for streaming, it passes the same response format configuration into `responses.stream(...)`

The parsed object comes back as the provider response's `output_parsed` value and becomes `AgentRunResult.output_data`.

## Best Practices

- keep your response models small and focused
- prefer explicit field names over overly loose schemas
- use structured output for application-facing data, not every conversational response
- return plain text when the output is for humans and no downstream parsing is needed
- combine structured output with tools when the model needs fresh local data before producing a typed result

## Example Models

### Flat Record

```python
class Person(BaseModel):
    name: str
    age: int
```

### Nested Result

```python
class Citation(BaseModel):
    source: str
    quote: str


class Summary(BaseModel):
    title: str
    bullets: list[str]
    citations: list[Citation]
```

### Decision Object

```python
class Classification(BaseModel):
    label: str
    confidence: float
    rationale: str
```

## When Not To Use Structured Output

Do not force structured output for every request.

Plain text is often better when:

- the answer is purely conversational
- you do not need downstream parsing
- you want free-form writing instead of typed data

Use structured output when your application needs predictable fields.
