import builtins
from unittest import mock

import pytest

from distributed.cli.config_loader import detect_redis_host


@pytest.mark.parametrize(
    "head_reachable,head_local_reachable,expected",
    [
        (True, False, "tailscale"),
        (False, True, "lan"),
        (False, False, "config"),
    ],
)
def test_detect_redis_host_prefers_reachable_hosts(head_reachable, head_local_reachable, expected):
    """Ensure detect_redis_host falls back from head -> head_local -> redis.host."""
    config = {
        "head": {"host": "tailscale", "host_local": "lan"},
        "redis": {"host": "config", "port": 6379},
    }

    with mock.patch("distributed.cli.config_loader.is_host_reachable") as reachable:
        reachable.side_effect = [head_reachable, head_local_reachable]
        host = detect_redis_host(config)

    assert host == expected
