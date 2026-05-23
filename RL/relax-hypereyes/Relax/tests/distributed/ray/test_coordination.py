# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Tests for mutual exclusion, GC of terminal requests, eviction monitoring,
and engine info queries."""

from unittest.mock import MagicMock, patch

import pytest


try:
    from relax.distributed.ray.rollout import (
        ScaleInRequest,
        ScaleInStatus,
        ScaleOutRequest,
        ScaleOutStatus,
    )

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from conftest import (
    create_test_manager,
    make_engine_group,
    make_mock_engine,
    make_rollout_server,
)


pytestmark = pytest.mark.skipif(not HAS_DEPS, reason="Missing ray/sglang dependencies")


# ==================== _find_active_scale_request ===========================


class TestFindActiveScaleRequest:
    def test_no_active_requests(self):
        manager = create_test_manager()
        assert manager._find_active_scale_request() is None

    def test_finds_active_scale_out(self):
        manager = create_test_manager()
        manager._scale_out_requests["r1"] = ScaleOutRequest(
            request_id="r1",
            status=ScaleOutStatus.CREATING,
        )
        result = manager._find_active_scale_request()
        assert result is not None
        assert result["type"] == "scale_out"
        assert result["request_id"] == "r1"

    def test_finds_active_scale_in(self):
        manager = create_test_manager()
        manager._scale_in_requests["r1"] = ScaleInRequest(
            request_id="r1",
            status=ScaleInStatus.DRAINING,
        )
        result = manager._find_active_scale_request()
        assert result is not None
        assert result["type"] == "scale_in"

    def test_ignores_terminal_scale_out(self):
        manager = create_test_manager()
        for status in (ScaleOutStatus.ACTIVE, ScaleOutStatus.FAILED, ScaleOutStatus.CANCELLED, ScaleOutStatus.PARTIAL):
            manager._scale_out_requests[status.value] = ScaleOutRequest(
                request_id=status.value,
                status=status,
            )
        assert manager._find_active_scale_request() is None

    def test_ignores_terminal_scale_in(self):
        manager = create_test_manager()
        for status in (ScaleInStatus.COMPLETED, ScaleInStatus.FAILED):
            manager._scale_in_requests[status.value] = ScaleInRequest(
                request_id=status.value,
                status=status,
            )
        assert manager._find_active_scale_request() is None

    def test_scale_out_checked_before_scale_in(self):
        """When both are active, scale_out is returned first."""
        manager = create_test_manager()
        manager._scale_out_requests["so"] = ScaleOutRequest(
            request_id="so",
            status=ScaleOutStatus.PENDING,
        )
        manager._scale_in_requests["si"] = ScaleInRequest(
            request_id="si",
            status=ScaleInStatus.PENDING,
        )
        result = manager._find_active_scale_request()
        assert result["type"] == "scale_out"

    @pytest.mark.parametrize(
        "status_str",
        ["PENDING", "CREATING", "CONNECTING", "HEALTH_CHECKING", "WEIGHT_SYNCING", "READY", "REMOVING"],
    )
    def test_all_non_terminal_scale_out_detected(self, status_str):
        status = ScaleOutStatus(status_str)
        manager = create_test_manager()
        manager._scale_out_requests["r"] = ScaleOutRequest(
            request_id="r",
            status=status,
        )
        assert manager._find_active_scale_request() is not None

    @pytest.mark.parametrize(
        "status_str",
        ["PENDING", "DRAINING", "REMOVING"],
    )
    def test_all_non_terminal_scale_in_detected(self, status_str):
        status = ScaleInStatus(status_str)
        manager = create_test_manager()
        manager._scale_in_requests["r"] = ScaleInRequest(
            request_id="r",
            status=status,
        )
        assert manager._find_active_scale_request() is not None


# ===================== _gc_terminal_requests ===============================


class TestGCTerminalRequests:
    def test_no_gc_under_limit(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 10
        for i in range(5):
            manager._scale_out_requests[f"r{i}"] = ScaleOutRequest(
                request_id=f"r{i}",
                status=ScaleOutStatus.ACTIVE,
            )
        manager._gc_terminal_requests()
        assert len(manager._scale_out_requests) == 5

    def test_gc_evicts_oldest(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 3

        for i in range(5):
            req = ScaleOutRequest(
                request_id=f"r{i}",
                status=ScaleOutStatus.ACTIVE,
            )
            req.updated_at = 1000 + i  # r0 is oldest
            manager._scale_out_requests[f"r{i}"] = req

        manager._gc_terminal_requests()
        assert len(manager._scale_out_requests) == 3
        # r0, r1 should be evicted (oldest)
        assert "r0" not in manager._scale_out_requests
        assert "r1" not in manager._scale_out_requests
        assert "r4" in manager._scale_out_requests

    def test_gc_preserves_non_terminal(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 1

        # 3 terminal
        for i in range(3):
            req = ScaleOutRequest(
                request_id=f"t{i}",
                status=ScaleOutStatus.ACTIVE,
            )
            req.updated_at = 1000 + i
            manager._scale_out_requests[f"t{i}"] = req

        # 1 non-terminal (should never be evicted)
        nt = ScaleOutRequest(request_id="nt", status=ScaleOutStatus.CREATING)
        nt.updated_at = 0  # oldest timestamp
        manager._scale_out_requests["nt"] = nt

        manager._gc_terminal_requests()
        # Only 1 terminal kept, but non-terminal is preserved
        assert "nt" in manager._scale_out_requests
        terminal_count = sum(1 for r in manager._scale_out_requests.values() if r.is_terminal())
        assert terminal_count == 1

    def test_gc_scale_in_requests(self):
        manager = create_test_manager()
        manager._max_terminal_requests = 2

        for i in range(4):
            req = ScaleInRequest(
                request_id=f"si{i}",
                status=ScaleInStatus.COMPLETED,
            )
            req.updated_at = 2000 + i
            manager._scale_in_requests[f"si{i}"] = req

        manager._gc_terminal_requests()
        assert len(manager._scale_in_requests) == 2


# =================== Eviction monitoring ==================================


class TestCheckAndHandleEvictions:
    def test_detects_evicted_engines(self, patch_ray_get):
        e_normal = make_mock_engine(evicted=False)
        e_evicted = make_mock_engine(evicted=True)
        g = make_engine_group(engines=[e_normal, e_evicted], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch.object(manager, "_handle_single_eviction") as mock_handle:
            manager._check_and_handle_evictions()
            mock_handle.assert_called_once()
            call_args = mock_handle.call_args
            assert call_args[0][0] == "default"  # srv_name
            assert call_args[0][2] == 1  # node0_idx of evicted engine

    def test_no_evictions(self, patch_ray_get):
        e = make_mock_engine(evicted=False)
        g = make_engine_group(engines=[e])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch.object(manager, "_handle_single_eviction") as mock_handle:
            manager._check_and_handle_evictions()
            mock_handle.assert_not_called()

    def test_all_dead_engines_skipped(self, patch_ray_get):
        g = make_engine_group(engines=[None, None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch.object(manager, "_handle_single_eviction") as mock_handle:
            manager._check_and_handle_evictions()
            mock_handle.assert_not_called()


class TestHandleSingleEviction:
    def test_marks_intentionally_removed(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        mock_monitor = MagicMock()
        mock_monitor._engine_group = g
        manager._health_monitors.append(mock_monitor)

        with patch("ray.kill"):
            manager._handle_single_eviction("default", g, 0)

        mock_monitor.mark_intentionally_removed.assert_called_with(0)

    def test_sets_engine_to_none(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch("ray.kill"):
            manager._handle_single_eviction("default", g, 0)

        assert g.all_engines[0] is None

    def test_cleans_up_empty_groups(self, patch_ray_get):
        e = make_mock_engine()
        g = make_engine_group(engines=[e], is_scaled_out=True)
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        with patch("ray.kill"):
            manager._handle_single_eviction("default", g, 0)

        # Group should be cleaned up since it's now empty
        assert len(srv.engine_groups) == 0


# ======================== get_engines_info =================================


class TestGetEnginesInfo:
    def test_basic_info(self, patch_ray_get):
        e1 = make_mock_engine(url="http://a:1")
        e2 = make_mock_engine(url="http://b:2")
        g = make_engine_group(engines=[e1, e2], num_gpus_per_engine=2)
        srv = make_rollout_server(
            engine_groups=[g],
            router_ip="10.0.0.1",
            router_port=3000,
        )
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        assert info["total_engines"] == 2
        assert "default" in info["models"]
        model = info["models"]["default"]
        assert model["router_ip"] == "10.0.0.1"
        assert len(model["engine_groups"]) == 1

    def test_dead_engines_show_dead_status(self, patch_ray_get):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1, None])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        engines = info["models"]["default"]["engine_groups"][0]["engines"]
        assert engines[0]["status"] == "active"
        assert engines[1]["status"] == "dead"

    def test_filter_by_model(self, patch_ray_get):
        e1 = make_mock_engine()
        g1 = make_engine_group(engines=[e1])
        srv1 = make_rollout_server(engine_groups=[g1])
        g2 = make_engine_group(engines=[make_mock_engine()])
        srv2 = make_rollout_server(engine_groups=[g2])
        manager = create_test_manager(servers={"actor": srv1, "reward": srv2})

        info = manager.get_engines_info(model_name="actor")
        assert "actor" in info["models"]
        assert "reward" not in info["models"]

    def test_multiple_groups(self, patch_ray_get):
        g1 = make_engine_group(
            engines=[make_mock_engine()],
            worker_type="prefill",
        )
        g2 = make_engine_group(
            engines=[make_mock_engine(), make_mock_engine()],
            worker_type="decode",
            rank_offset=1,
        )
        srv = make_rollout_server(engine_groups=[g1, g2])
        manager = create_test_manager(servers={"default": srv})

        info = manager.get_engines_info()
        groups = info["models"]["default"]["engine_groups"]
        assert len(groups) == 2
        assert groups[0]["worker_type"] == "prefill"
        assert groups[1]["worker_type"] == "decode"
        assert info["total_engines"] == 3


# ==================== set_weight_updating ==================================


class TestSetWeightUpdating:
    def test_sets_flag_and_mirrors_to_engines(self, patch_ray_get):
        e1 = make_mock_engine()
        e2 = make_mock_engine()
        g = make_engine_group(engines=[e1, e2])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager.set_weight_updating(True)
        assert manager._is_weight_updating is True
        e1.set_weight_updating.remote.assert_called_with(True)
        e2.set_weight_updating.remote.assert_called_with(True)

    def test_unsets_flag(self, patch_ray_get):
        e1 = make_mock_engine()
        g = make_engine_group(engines=[e1])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        manager.set_weight_updating(True)
        manager.set_weight_updating(False)
        assert manager._is_weight_updating is False

    def test_skips_dead_engines(self, patch_ray_get):
        g = make_engine_group(engines=[None, make_mock_engine()])
        srv = make_rollout_server(engine_groups=[g])
        manager = create_test_manager(servers={"default": srv})

        # Should not raise
        manager.set_weight_updating(True)
