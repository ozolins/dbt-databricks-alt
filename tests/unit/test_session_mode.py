from multiprocessing import get_context
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, PropertyMock, patch

import pytest
from dbt.adapters.exceptions import FailedToConnectError
from dbt_common.exceptions import DbtConfigError, DbtRuntimeError, DbtValidationError

from dbt.adapters.databricks import DatabricksAdapter
from dbt.adapters.databricks.connections import (
    DatabricksConnectionManager,
    DatabricksSessionConnectionManager,
    QueryContextWrapper,
)
from dbt.adapters.databricks.credentials import (
    DATABRICKS_RUNTIME_VERSION_ENV,
    DBT_DATABRICKS_SESSION_MODE_ENV,
    DatabricksCredentials,
)
from dbt.adapters.databricks.dbr_capabilities import DBRCapability
from dbt.adapters.databricks.impl import DeltaLiveTableAPIBase, RelationAPIBase
from dbt.adapters.databricks.python_models.python_submissions import SessionPythonJobHelper
from dbt.adapters.databricks.relation import DatabricksRelation
from tests.unit.utils import config_from_parts_or_dicts

PROJECT = {
    "name": "X",
    "version": "0.1",
    "profile": "test",
    "project-root": "/tmp/dbt/does-not-exist",
    "config-version": 2,
}


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    monkeypatch.delenv(DBT_DATABRICKS_SESSION_MODE_ENV, raising=False)
    monkeypatch.delenv(DATABRICKS_RUNTIME_VERSION_ENV, raising=False)
    DatabricksConnectionManager._dbr_capabilities_cache.pop(
        DatabricksSessionConnectionManager.CAPABILITIES_CACHE_KEY, None
    )
    yield
    DatabricksConnectionManager._dbr_capabilities_cache.pop(
        DatabricksSessionConnectionManager.CAPABILITIES_CACHE_KEY, None
    )


def _credentials(**kwargs: Any) -> DatabricksCredentials:
    return DatabricksCredentials.from_dict({"schema": "analytics", **kwargs})


def _adapter(**target: Any) -> DatabricksAdapter:
    profile = {
        "outputs": {"test": {"type": "databricks", "schema": "analytics", **target}},
        "target": "test",
    }
    return DatabricksAdapter(config_from_parts_or_dicts(PROJECT, profile), get_context("spawn"))


class TestSessionCredentials:
    def test_session_method_needs_no_connection_details(self):
        creds = _credentials(method="session", catalog="main")

        creds.validate_creds()

        assert creds.is_session_mode
        assert not creds.has_api_access
        assert creds._credentials_manager is None
        assert creds.unique_field == "session"
        assert creds.cluster_id is None

    def test_method_is_case_insensitive(self):
        assert _credentials(method=" Session ").method == "session"

    def test_rejects_unknown_method(self):
        with pytest.raises(DbtValidationError, match="Invalid connection method: `thrift`"):
            _credentials(method="thrift")

    def test_defaults_to_dbsql(self):
        creds = _credentials(host="h", http_path="p", token="t")

        assert creds.method == "dbsql"
        assert not creds.is_session_mode

    def test_env_var_selects_session_even_with_a_host(self, monkeypatch):
        monkeypatch.setenv(DBT_DATABRICKS_SESSION_MODE_ENV, "TRUE")

        creds = _credentials(host="h", http_path="p", token="t")

        assert creds.is_session_mode
        assert creds.has_api_access

    def test_explicit_method_wins_over_env_var(self, monkeypatch):
        monkeypatch.setenv(DBT_DATABRICKS_SESSION_MODE_ENV, "true")

        assert not _credentials(method="dbsql", host="h", http_path="p").is_session_mode

    @pytest.mark.parametrize("host, expected", [(None, "session"), ("h", "dbsql")])
    def test_databricks_runtime_without_host_selects_session(self, monkeypatch, host, expected):
        monkeypatch.setenv(DATABRICKS_RUNTIME_VERSION_ENV, "15.4")

        assert _credentials(host=host).method == expected

    def test_host_enables_api_access_and_requires_auth(self):
        creds = _credentials(method="session", host="h", token="t")
        creds.validate_creds()
        assert creds.has_api_access
        assert creds._credentials_manager is not None

        with pytest.raises(DbtConfigError, match="auth_type"):
            _credentials(method="session", host="h").validate_creds()

    def test_accepts_a_profile_with_named_compute(self):
        _credentials(method="session", compute={"alt": {"http_path": "p"}}).validate_creds()

    def test_connection_info_shows_method_instead_of_endpoint(self):
        creds = _credentials(method="session", catalog="main", session_properties={"a": "b"})

        assert [key for key, _ in creds.connection_info()] == [
            "method",
            "catalog",
            "schema",
            "session_properties",
        ]
        assert [key for key, _ in creds.connection_info(with_aliases=True)] == [
            "method",
            "database",
            "schema",
            "session_properties",
            "catalog",
        ]

    def test_dbsql_connection_info_is_unchanged(self):
        creds = _credentials(host="h", http_path="p", catalog="main")

        assert [key for key, _ in creds.connection_info()] == [
            "host",
            "http_path",
            "catalog",
            "schema",
        ]


class TestSessionConnectionManager:
    @pytest.fixture
    def spark(self):
        root = Mock()
        with (
            patch("dbt.adapters.databricks.connections.get_root_spark_session", return_value=root),
            patch("dbt.adapters.databricks.connections.detect_dbr_version", return_value=(15, 4)),
        ):
            yield root

    def test_adapter_selects_connection_manager_by_method(self):
        assert type(_adapter(method="session").connections) is DatabricksSessionConnectionManager
        assert type(_adapter(host="h", http_path="p", token="t").connections) is (
            DatabricksConnectionManager
        )

    def test_opens_a_spark_session_without_the_sql_connector(self, spark):
        adapter = _adapter(method="session", catalog="main")
        handle = Mock(session_id="spark-1")

        with (
            patch(
                "dbt.adapters.databricks.connections.DatabricksSessionHandle.create",
                return_value=handle,
            ) as create,
            patch("dbt.adapters.databricks.connections.check_spog_preconditions") as spog,
            patch("dbt.adapters.databricks.handle.dbsql.connect") as dbsql_connect,
        ):
            connection = adapter.acquire_connection("model")
            assert connection.handle is handle

        create.assert_called_once_with(adapter.config.credentials)
        spog.assert_not_called()
        dbsql_connect.assert_not_called()
        assert connection.session_id == "spark-1"
        assert connection.capabilities.dbr_version == (15, 4)
        assert not connection.capabilities.is_sql_warehouse
        assert adapter.has_capability(DBRCapability.INSERT_BY_NAME)
        assert adapter.is_cluster()

    def test_failure_to_open_is_a_connection_error(self, spark):
        adapter = _adapter(method="session")

        with patch(
            "dbt.adapters.databricks.connections.DatabricksSessionHandle.create",
            side_effect=Exception("[CATALOG_NOT_FOUND]"),
        ):
            connection = adapter.acquire_connection("model")
            with pytest.raises(FailedToConnectError, match="CATALOG_NOT_FOUND"):
                connection.handle

    def test_rejects_databricks_compute(self, spark):
        connections = _adapter(method="session").connections
        context = QueryContextWrapper(compute_name="alt", relation_name="my_model")

        with pytest.raises(DbtRuntimeError, match="`databricks_compute` is not supported"):
            connections._create_fresh_connection("model", context)

    def test_api_client_needs_a_host(self):
        with pytest.raises(DbtRuntimeError, match="needs `host`"):
            _adapter(method="session").connections.api_client

    def test_cancel_open_interrupts_python_models_without_the_api(self):
        connections = _adapter(method="session").connections

        with (
            patch("dbt.adapters.databricks.connections.SessionRunTracker") as tracker,
            patch("dbt.adapters.databricks.connections.PythonRunTracker") as run_tracker,
        ):
            connections.cancel_open()

        tracker.interrupt_all.assert_called_once()
        run_tracker.cancel_runs.assert_not_called()

    def test_cancel_open_also_cancels_api_runs_when_a_host_is_set(self):
        connections = _adapter(method="session", host="h", token="t").connections

        with (
            patch("dbt.adapters.databricks.connections.SessionRunTracker") as tracker,
            patch("dbt.adapters.databricks.connections.PythonRunTracker") as run_tracker,
            patch("dbt.adapters.databricks.connections.DatabricksApiClient") as api_client,
        ):
            connections.cancel_open()

        tracker.interrupt_all.assert_called_once()
        run_tracker.cancel_runs.assert_called_once_with(api_client.return_value)


class TestSessionAdapter:
    def test_parse_time_capabilities_are_conservative(self):
        adapter = _adapter(method="session")

        assert adapter._has_dbr_capability_parse("insert_by_name") is False

    def test_python_models_default_to_session(self):
        assert _adapter(method="session").default_python_submission_method == "session"
        assert _adapter(host="h", http_path="p", token="t").default_python_submission_method == (
            "all_purpose_cluster"
        )

    def test_api_submission_methods_need_a_host(self):
        adapter = _adapter(method="session")
        parsed_model = {"config": {"submission_method": "job_cluster"}}

        with (
            patch.object(
                adapter.connections,
                "get_thread_connection",
                return_value=SimpleNamespace(name="my_model"),
            ),
            pytest.raises(DbtConfigError, match="`job_cluster` submission method"),
        ):
            adapter.submit_python_job(parsed_model, "")

    @staticmethod
    def _adapter_without_api() -> Mock:
        adapter = Mock()
        adapter.config.credentials.has_api_access = False
        type(adapter.connections).api_client = PropertyMock(
            side_effect=AssertionError("the REST API must not be used")
        )
        return adapter

    def test_table_tags_are_read_with_sql_without_the_api(self):
        adapter = self._adapter_without_api()
        adapter.execute_macro.return_value.rows = [("team", "data"), ("pii", None)]
        relation = DatabricksRelation.create(database="main", schema="s", identifier="t")

        tags = RelationAPIBase._get_table_tags(adapter, relation)

        assert tags == {"team": "data", "pii": ""}
        adapter.execute_macro.assert_called_once_with("fetch_tags", kwargs={"relation": relation})

    def test_pipeline_refreshes_are_not_awaited_without_the_api(self):
        adapter = self._adapter_without_api()
        relation_config = Mock()
        relation_config.config = {"tblproperties": Mock(pipeline_id="pipeline-1")}

        with (
            patch.object(RelationAPIBase, "get_from_relation", return_value=relation_config),
            patch("dbt.adapters.databricks.impl.logger") as logger,
        ):
            result = DeltaLiveTableAPIBase.get_from_relation(adapter, Mock())

        assert result is relation_config
        logger.warning.assert_called_once()


class TestSessionPythonJobHelper:
    MODEL = {"database": "main", "schema": "analytics", "alias": "my_model"}

    def test_runs_model_on_a_configured_spark_session(self):
        creds = _credentials(method="session")

        with (
            patch(
                "dbt.adapters.databricks.python_models.python_submissions.create_spark_session"
            ) as create,
            patch(
                "dbt.adapters.databricks.python_models.python_submissions.run_python_model"
            ) as run,
        ):
            helper = SessionPythonJobHelper({**self.MODEL, "config": {"timeout": 30}}, creds)
            helper.submit("code")

        create.assert_called_once_with(creds)
        run.assert_called_once_with(create.return_value, "code", timeout=30, name="my_model")

    def test_warns_that_configured_libraries_are_not_installed(self):
        with (
            patch("dbt.adapters.databricks.python_models.python_submissions.create_spark_session"),
            patch("dbt.adapters.databricks.python_models.python_submissions.logger") as logger,
        ):
            SessionPythonJobHelper(
                {**self.MODEL, "config": {"packages": ["requests"]}},
                _credentials(method="session"),
            )

        assert "Ignoring the libraries" in logger.warning.call_args.args[0]

    def test_requires_a_session_mode_profile(self):
        creds = _credentials(host="h", http_path="p", token="t")

        with pytest.raises(DbtConfigError, match="requires `method: session`"):
            SessionPythonJobHelper({**self.MODEL, "config": {}}, creds)
