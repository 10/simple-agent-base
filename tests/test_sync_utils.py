from __future__ import annotations

from typing import get_type_hints

from simple_agent_base.sync_utils import SyncRuntime


def test_sync_runtime_run_type_hints_resolve() -> None:
    hints = get_type_hints(SyncRuntime.run)

    assert "awaitable_factory" in hints
