from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        populate_by_name=True,
    )

    model: str = Field(validation_alias="OPENAI_MODEL")
    api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    max_turns: int = Field(default=8, ge=1)
    parallel_tool_calls: bool = False
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        validation_alias="OPENAI_REASONING_EFFORT",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    timeout: float | None = Field(default=None, gt=0.0)
    tool_timeout: float | None = Field(default=None, gt=0.0)
