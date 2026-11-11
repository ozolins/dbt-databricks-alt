import datetime
import decimal
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from dbt_common.exceptions import DbtRuntimeError

from dbt.adapters.databricks.credentials import DatabricksCredentials
from dbt.adapters.databricks.session import (
    DatabricksSessionHandle,
    SessionCursorWrapper,
    SessionRunTracker,
    SparkJobTag,
    create_spark_session,
    detect_dbr_version,
    get_root_spark_session,
    render_bindings,
    run_python_model,
)


def _field(name, simple_string, nullable=True, **type_attrs):
    data_type = SimpleNamespace(simpleString=lambda: simple_string, **type_attrs)
    return SimpleNamespace(name=name, dataType=data_type, nullable=nullable)


def _dataframe(rows, fields=()):
    df = Mock()
    df.collect.return_value = rows
    df.schema.fields = list(fields)
    return df


def _classic_spark():
    """A SparkSession double with a SparkContext, i.e. classic (non-Connect) Spark."""
    spark = Mock()
    local_properties = {}
    context = spark.sparkContext
    context.getLocalProperty.side_effect = local_properties.get
    context.setLocalProperty.side_effect = local_properties.__setitem__

    def set_job_group(group_id, description, interruptOnCancel=False):
        local_properties["spark.jobGroup.id"] = group_id

    context.setJobGroup.side_effect = set_job_group
    spark.local_properties = local_properties
    return spark


def _connect_spark():
    """A SparkSession double without a SparkContext, i.e. Spark Connect."""
    spark = Mock()
    type(spark).sparkContext = property(Mock(side_effect=RuntimeError("no SparkContext")))
    return spark


def _session_credentials(**kwargs):
    return DatabricksCredentials(method="session", schema="analytics", **kwargs)


class TestRenderBindings:
    def test_escapes_like_the_sql_connector(self):
        sql = render_bindings("select %s, %s, %s", ["it's", "back\\slash", None])
        assert sql == "select 'it\\'s', 'back\\\\slash', NULL"

    def test_quotes_dates_and_timestamps(self):
        sql = render_bindings(
            "select %s, %s",
            [datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 2, 3, 4, 5, 6)],
        )
        assert sql == "select '2020-01-01', '2020-01-02 03:04:05.000006'"

    def test_renders_numbers_and_booleans_unquoted(self):
        assert render_bindings("values (%s, %s, %s)", [True, 7, 1.5]) == "values (True, 7, 1.5)"


class TestSessionCursorWrapper:
    def test_collects_eagerly_and_describes_like_the_sql_connector(self):
        df = _dataframe(
            [(decimal.Decimal("1.25"), [1])],
            [
                _field("amount", "decimal(10,2)", precision=10, scale=2),
                _field("ids", "array<int>", nullable=False),
            ],
        )

        cursor = SessionCursorWrapper.from_dataframe(df)

        df.collect.assert_called_once()
        assert cursor.description == [
            ("amount", "decimal", None, None, 10, 2, True),
            ("ids", "array", None, None, None, None, False),
        ]

    def test_fetches_advance_through_rows(self):
        cursor = SessionCursorWrapper([(1,), (2,), (3,), (4,)], [])

        assert cursor.fetchone() == (1,)
        assert cursor.fetchmany(2) == [(2,), (3,)]
        assert cursor.fetchall() == [(4,)]
        assert cursor.fetchone() is None

    def test_fetch_after_close_raises(self):
        cursor = SessionCursorWrapper([(1,)], [])
        cursor.close()

        with pytest.raises(DbtRuntimeError, match="closed cursor"):
            cursor.fetchall()

    def test_response_has_no_query_id(self):
        response = SessionCursorWrapper([], []).get_response()

        assert response.query_id == "N/A"
        assert response._message == "OK"


class TestDatabricksSessionHandle:
    def test_execute_cleans_sql_and_inlines_translated_bindings(self):
        spark = _classic_spark()
        spark.sql.return_value = _dataframe([])
        handle = DatabricksSessionHandle(spark)

        handle.execute("insert into t values (%s, %s);", [decimal.Decimal("1.50"), "x"])

        spark.sql.assert_called_once_with("insert into t values (1.5, 'x')")

    def test_execute_without_bindings_leaves_percent_signs_alone(self):
        spark = _classic_spark()
        spark.sql.return_value = _dataframe([])

        DatabricksSessionHandle(spark).execute("select * from t where c like '%x%'")

        spark.sql.assert_called_once_with("select * from t where c like '%x%'")

    def test_execute_closes_previous_cursor(self):
        spark = _classic_spark()
        spark.sql.return_value = _dataframe([(1,)])
        handle = DatabricksSessionHandle(spark)

        first = handle.execute("select 1")
        handle.execute("select 1")

        assert not first.open

    def test_execute_runs_in_a_job_group_and_restores_the_callers(self):
        spark = _classic_spark()
        spark.local_properties["spark.jobGroup.id"] = "notebook-cell"
        groups_during_sql = []

        def sql(statement):
            groups_during_sql.append(spark.local_properties["spark.jobGroup.id"])
            return _dataframe([])

        spark.sql.side_effect = sql
        handle = DatabricksSessionHandle(spark)

        handle.execute("select 1")

        assert groups_during_sql == [handle._job_tag.tag]
        assert spark.local_properties["spark.jobGroup.id"] == "notebook-cell"

    def test_execute_on_closed_handle_raises(self):
        handle = DatabricksSessionHandle(_classic_spark())
        handle.close()

        with pytest.raises(DbtRuntimeError, match="closed connection"):
            handle.execute("select 1")

    def test_list_tables_excludes_temporary_views_and_leaves_type_unresolved(self):
        spark = _classic_spark()
        spark.sql.return_value.collect.return_value = [
            {"tableName": "orders", "isTemporary": False},
            {"tableName": "tmp_view", "isTemporary": True},
        ]
        cursor = DatabricksSessionHandle(spark).list_tables("hive_metastore", "my`schema")

        spark.sql.assert_called_once_with("SHOW TABLES IN `hive_metastore`.`my``schema`")
        assert [column[0] for column in cursor.description] == [
            "TABLE_CAT",
            "TABLE_SCHEM",
            "TABLE_NAME",
            "TABLE_TYPE",
        ]
        assert cursor.fetchall() == [("hive_metastore", "my`schema", "orders", None)]

    def test_list_schemas_matches_name_exactly_ignoring_case(self):
        spark = _classic_spark()
        spark.sql.return_value.collect.return_value = [("analytics",), ("analytics_2",)]

        cursor = DatabricksSessionHandle(spark).list_schemas("main", "Analytics")

        spark.sql.assert_called_once_with("SHOW SCHEMAS IN `main`")
        assert cursor.fetchall() == [("analytics", "main")]

    def test_cancel_interrupts_the_job_group_without_stopping_spark(self):
        spark = _classic_spark()
        spark.sql.return_value = _dataframe([])
        handle = DatabricksSessionHandle(spark)
        handle.execute("select 1")

        handle.cancel()

        spark.sparkContext.cancelJobGroup.assert_called_once_with(handle._job_tag.tag)
        spark.stop.assert_not_called()
        assert not handle.open

    def test_close_does_not_stop_spark(self):
        spark = _classic_spark()
        handle = DatabricksSessionHandle(spark)

        handle.close()

        spark.stop.assert_not_called()
        assert not handle.open

    def test_dbr_version_is_read_once(self):
        spark = _classic_spark()
        spark.conf.get.return_value = "15.4.x-scala2.12"
        handle = DatabricksSessionHandle(spark)

        assert handle.dbr_version == (15, 4)
        assert handle.dbr_version == (15, 4)
        spark.conf.get.assert_called_once()


class TestDetectDbrVersion:
    @pytest.fixture(autouse=True)
    def _no_runtime_version(self, monkeypatch):
        monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)

    @pytest.mark.parametrize(
        "runtime_version, expected",
        [("14.3", (14, 3)), ("client.2.5", (sys.maxsize, sys.maxsize))],
    )
    def test_falls_back_to_runtime_version_env(self, monkeypatch, runtime_version, expected):
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", runtime_version)
        spark = Mock()
        spark.conf.get.side_effect = Exception("[CONFIG_NOT_AVAILABLE]")

        assert detect_dbr_version(spark) == expected

    def test_reads_cluster_spark_version(self):
        spark = Mock()
        spark.conf.get.return_value = "16.4.x-photon-scala2.13"

        assert detect_dbr_version(spark) == (16, 4)

    @pytest.mark.parametrize("conf_value", [None, ""])
    def test_assumes_latest_without_cluster_version(self, conf_value):
        spark = Mock()
        spark.conf.get.return_value = conf_value

        assert detect_dbr_version(spark) == (sys.maxsize, sys.maxsize)

    def test_assumes_latest_when_conf_is_not_readable(self):
        spark = Mock()
        spark.conf.get.side_effect = Exception("[CONFIG_NOT_AVAILABLE]")

        assert detect_dbr_version(spark) == (sys.maxsize, sys.maxsize)


class TestCreateSparkSession:
    @pytest.fixture
    def root(self):
        root = Mock()
        with patch("dbt.adapters.databricks.session.get_root_spark_session", return_value=root):
            yield root

    def test_isolates_each_connection_in_a_new_session(self, root):
        spark = create_spark_session(_session_credentials(database="main"))

        assert spark is root.newSession.return_value
        spark.catalog.setCurrentCatalog.assert_called_once_with("main")
        spark.catalog.setCurrentDatabase.assert_called_once_with("analytics")
        root.catalog.setCurrentCatalog.assert_not_called()

    def test_shares_the_cluster_session_when_it_cannot_create_one(self, root):
        root.newSession.side_effect = Exception("not supported in Spark Connect")

        assert create_spark_session(_session_credentials()) is root

    def test_applies_session_properties(self, root):
        create_spark_session(
            _session_credentials(session_properties={"spark.sql.ansi.enabled": False, "spark.x": 4})
        )

        conf = root.newSession.return_value.conf
        conf.set.assert_any_call("spark.sql.ansi.enabled", "false")
        conf.set.assert_any_call("spark.x", "4")

    def test_tolerates_no_catalog_switch_for_hive_metastore(self, root):
        spark = root.newSession.return_value
        spark.catalog.setCurrentCatalog.side_effect = Exception("[CATALOG_NOT_FOUND]")

        create_spark_session(_session_credentials())

        spark.catalog.setCurrentDatabase.assert_called_once_with("analytics")

    def test_raises_when_another_catalog_cannot_be_used(self, root):
        spark = root.newSession.return_value
        spark.catalog.setCurrentCatalog.side_effect = Exception("[CATALOG_NOT_FOUND]")

        with pytest.raises(Exception, match="CATALOG_NOT_FOUND"):
            create_spark_session(_session_credentials(database="main"))

    def test_tolerates_a_target_schema_that_does_not_exist_yet(self, root):
        spark = root.newSession.return_value
        spark.catalog.setCurrentDatabase.side_effect = Exception("[SCHEMA_NOT_FOUND] x")

        assert create_spark_session(_session_credentials(database="main")) is spark

    def test_raises_other_schema_errors(self, root):
        spark = root.newSession.return_value
        spark.catalog.setCurrentDatabase.side_effect = Exception("[INSUFFICIENT_PERMISSIONS]")

        with pytest.raises(Exception, match="INSUFFICIENT_PERMISSIONS"):
            create_spark_session(_session_credentials(database="main"))


class TestGetRootSparkSession:
    def test_requires_pyspark(self):
        with patch("dbt.adapters.databricks.session.SparkSession", None):
            with pytest.raises(DbtRuntimeError, match="requires PySpark"):
                get_root_spark_session()

    def test_uses_the_existing_session(self):
        with patch("dbt.adapters.databricks.session.SparkSession") as spark_session:
            assert get_root_spark_session() is spark_session.active.return_value

        spark_session.builder.getOrCreate.assert_not_called()

    def test_never_creates_a_session(self):
        with patch("dbt.adapters.databricks.session.SparkSession") as spark_session:
            spark_session.active.side_effect = Exception("[NO_ACTIVE_OR_DEFAULT_SESSION]")
            with pytest.raises(DbtRuntimeError, match="found no SparkSession"):
                get_root_spark_session()

        spark_session.builder.getOrCreate.assert_not_called()

    def test_finds_the_session_on_pyspark_without_active(self):
        spark_session = Mock(spec=["getActiveSession", "_instantiatedSession"])
        spark_session.getActiveSession.return_value = None

        with patch("dbt.adapters.databricks.session.SparkSession", spark_session):
            assert get_root_spark_session() is spark_session._instantiatedSession


class TestSparkJobTag:
    def test_uses_job_group_on_classic_spark(self):
        spark = _classic_spark()
        job_tag = SparkJobTag(spark)

        with job_tag.applied():
            assert spark.local_properties["spark.jobGroup.id"] == job_tag.tag
        job_tag.interrupt()

        assert spark.local_properties["spark.jobGroup.id"] is None
        spark.sparkContext.cancelJobGroup.assert_called_once_with(job_tag.tag)
        spark.addTag.assert_not_called()

    def test_uses_operation_tag_on_spark_connect(self):
        spark = _connect_spark()
        job_tag = SparkJobTag(spark)

        with job_tag.applied():
            spark.addTag.assert_called_once_with(job_tag.tag)
            spark.removeTag.assert_not_called()
        job_tag.interrupt()

        spark.removeTag.assert_called_once_with(job_tag.tag)
        spark.interruptTag.assert_called_once_with(job_tag.tag)

    def test_runs_untagged_when_spark_supports_neither(self):
        spark = _connect_spark()
        spark.addTag.side_effect = Exception("unsupported")
        job_tag = SparkJobTag(spark)

        with job_tag.applied():
            pass
        job_tag.interrupt()

        spark.removeTag.assert_not_called()

    def test_errors_in_the_body_are_not_chained_to_fallbacks(self):
        spark = _connect_spark()

        with pytest.raises(ValueError) as exc_info:
            with SparkJobTag(spark).applied():
                raise ValueError("query failed")

        assert exc_info.value.__context__ is None


class TestRunPythonModel:
    def test_runs_model_code_with_spark_in_scope(self):
        spark = _classic_spark()

        run_python_model(spark, "spark.sql('select 1')", timeout=10, name="my_model")

        spark.sql.assert_called_once_with("select 1")

    def test_failure_reports_the_models_traceback(self):
        with pytest.raises(DbtRuntimeError) as exc_info:
            run_python_model(
                _classic_spark(), "x = 1\nraise ValueError('boom')", timeout=10, name="my_model"
            )

        message = str(exc_info.value)
        assert '<python model my_model>", line 2' in message
        assert "ValueError: boom" in message
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_timeout_interrupts_spark_jobs(self):
        spark = _classic_spark()
        release = threading.Event()
        code = "release.wait(5)"

        with patch(
            "dbt.adapters.databricks.session._python_model_globals",
            return_value={"spark": spark, "release": release},
        ):
            started = time.time()
            with pytest.raises(DbtRuntimeError, match="did not finish within 0.2 seconds"):
                run_python_model(spark, code, timeout=0.2, name="slow")
        release.set()

        assert time.time() - started < 5
        spark.sparkContext.cancelJobGroup.assert_called_once()

    def test_running_models_are_interrupted_by_cancel(self):
        spark = _classic_spark()
        running = threading.Event()
        release = threading.Event()
        code = "running.set()\nrelease.wait(5)"
        model_globals = {"spark": spark, "running": running, "release": release}

        with patch(
            "dbt.adapters.databricks.session._python_model_globals", return_value=model_globals
        ):
            runner = threading.Thread(
                target=run_python_model, args=(spark, code, 10, "cancelled"), daemon=True
            )
            runner.start()
            assert running.wait(5)
            SessionRunTracker.interrupt_all()
            release.set()
            runner.join(5)

        spark.sparkContext.cancelJobGroup.assert_called_once()
        assert not SessionRunTracker._tags
