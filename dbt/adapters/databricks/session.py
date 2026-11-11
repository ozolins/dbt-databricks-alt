"""Run dbt on the SparkSession of the Databricks cluster that dbt itself is running on.

Used with `method: session`, so a whole project (SQL and Python models) runs on a job cluster
without the Databricks SQL connector.
"""

import os
import re
import sys
import threading
import traceback
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from types import TracebackType
from typing import TYPE_CHECKING, Any, Optional

from dbt.adapters.contracts.connection import AdapterResponse
from dbt_common.exceptions import DbtRuntimeError

from databricks.sql.utils import ParamEscaper, inject_parameters
from dbt.adapters.databricks.credentials import (
    DATABRICKS_RUNTIME_VERSION_ENV,
    DatabricksCredentials,
)
from dbt.adapters.databricks.handle import DatabricksAdapterResponse, SqlUtils
from dbt.adapters.databricks.logging import logger

# PySpark ships with Databricks Runtime rather than as a dependency of this package.
try:
    from pyspark.sql import SparkSession
except ImportError:
    SparkSession = None  # type: ignore[assignment,misc]

try:
    from pyspark.dbutils import DBUtils
except ImportError:
    DBUtils = None

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructField

SPARK_VERSION_CONF = "spark.databricks.clusterUsageTags.sparkVersion"
HIVE_METASTORE_CATALOG = "hive_metastore"
JOB_GROUP_PROPERTIES = ("spark.jobGroup.id", "spark.job.description", "spark.job.interruptOnCancel")
TYPE_NAME_SEPARATOR_REGEX = re.compile(r"[(<]")
RUNTIME_VERSION_REGEX = re.compile(r"\d+\.\d+")


def get_root_spark_session() -> "SparkSession":
    """Return the cluster's existing SparkSession.

    Never creates one: outside the driver's Python process that would start a local Spark with
    its own metastore.
    """
    if SparkSession is None:
        raise DbtRuntimeError(
            "`method: session` requires PySpark. Run dbt on a Databricks cluster, or use "
            "`method: dbsql` to connect through a SQL warehouse or cluster."
        )
    spark = _existing_spark_session()
    if spark is None:
        raise DbtRuntimeError(
            "`method: session` found no SparkSession to use. Run dbt in the Python process of "
            "the cluster's driver, e.g. with `dbtRunner` from a notebook or Python task, rather "
            "than as a subprocess."
        )
    return spark


def _existing_spark_session() -> Optional["SparkSession"]:
    active = getattr(SparkSession, "active", None)
    try:
        if active is not None:
            return active()
        # PySpark before 3.5 has no `active`.
        return SparkSession.getActiveSession() or getattr(
            SparkSession, "_instantiatedSession", None
        )
    except Exception as e:
        logger.debug(f"No existing SparkSession: {e}")
        return None


def create_spark_session(credentials: DatabricksCredentials) -> "SparkSession":
    """Return a SparkSession for one dbt connection, configured from the profile.

    Each connection gets its own session so `USE`, `SET` and temporary views stay isolated, as
    they do between SQL connector sessions. Spark Connect cannot create sessions, so there all
    connections share the cluster session.
    """
    root = get_root_spark_session()
    try:
        spark = root.newSession()
    except Exception as e:
        logger.debug(f"Cannot isolate a SparkSession, sharing the cluster session: {e}")
        spark = root

    for key, value in (credentials.session_properties or {}).items():
        spark.conf.set(key, _conf_value(value))
    _use_namespace(spark, credentials.database, credentials.schema)
    return spark


def _conf_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _use_namespace(spark: "SparkSession", catalog: Optional[str], schema: Optional[str]) -> None:
    if catalog:
        try:
            spark.catalog.setCurrentCatalog(catalog)
        except Exception as e:
            # Without Unity Catalog there is no catalog to switch to, only hive_metastore.
            if catalog.lower() != HIVE_METASTORE_CATALOG:
                raise
            logger.debug(f"Keeping the default catalog, cannot switch to {catalog}: {e}")
    if schema:
        try:
            spark.catalog.setCurrentDatabase(schema)
        except Exception as e:
            # dbt creates the target schema later in the run, so it may not exist yet.
            if "SCHEMA_NOT_FOUND" not in str(e):
                raise
            logger.debug(f"Schema {schema} does not exist yet, keeping the default schema")


def quote_identifier(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def detect_dbr_version(spark: "SparkSession") -> tuple[int, int]:
    """DBR version of the cluster; assumes the latest when it cannot tell, as for warehouses."""
    try:
        version = spark.conf.get(SPARK_VERSION_CONF, None)
        if version:
            return SqlUtils.extract_dbr_version(version)
    except Exception as e:
        logger.debug(f"Failed to read {SPARK_VERSION_CONF}: {e}")
    # Serverless reports a client version such as `client.2.5` here, which is not a DBR version.
    runtime_version = os.getenv(DATABRICKS_RUNTIME_VERSION_ENV, "")
    if RUNTIME_VERSION_REGEX.fullmatch(runtime_version):
        return SqlUtils.extract_dbr_version(runtime_version)
    return (sys.maxsize, sys.maxsize)


def render_bindings(sql: str, bindings: Sequence[Any]) -> str:
    """Inline bindings exactly as the SQL connector's `use_inline_params` does for DBSQL."""
    return inject_parameters(sql, ParamEscaper().escape_args(bindings))


class SparkJobTag:
    """Labels the Spark jobs a thread starts so that another thread can interrupt them.

    Classic Spark uses a job group: its operation tags can only be interrupted from the thread
    that added them. Spark Connect has no SparkContext, so it uses an operation tag.
    """

    def __init__(self, spark: "SparkSession"):
        self._spark = spark
        self.tag = f"dbt-{uuid.uuid4().hex}"
        self._uses_job_group = False

    @contextmanager
    def applied(self) -> Iterator[None]:
        previous = self._set_job_group()
        if previous is not None:
            try:
                yield
            finally:
                # Restore the caller's group, e.g. the one a notebook uses to cancel its cell.
                context = self._spark.sparkContext
                for key, value in previous.items():
                    context.setLocalProperty(key, value)
            return

        tagged = self._add_tag()
        try:
            yield
        finally:
            if tagged:
                self._spark.removeTag(self.tag)

    def _set_job_group(self) -> Optional[dict[str, Optional[str]]]:
        try:
            context = self._spark.sparkContext
            previous = {key: context.getLocalProperty(key) for key in JOB_GROUP_PROPERTIES}
            context.setJobGroup(self.tag, "dbt", interruptOnCancel=True)
        except Exception:
            return None
        self._uses_job_group = True
        return previous

    def _add_tag(self) -> bool:
        try:
            self._spark.addTag(self.tag)
            return True
        except Exception as e:
            logger.debug(f"Spark jobs started by dbt cannot be interrupted: {e}")
            return False

    def interrupt(self) -> None:
        try:
            if self._uses_job_group:
                self._spark.sparkContext.cancelJobGroup(self.tag)
            else:
                self._spark.interruptTag(self.tag)
        except Exception as e:
            logger.debug(f"Failed to interrupt Spark jobs tagged {self.tag}: {e}")


class SessionRunTracker:
    """Python models running in session mode, so that `cancel_open` can interrupt them."""

    _tags: set[SparkJobTag] = set()
    _lock = threading.Lock()

    @classmethod
    @contextmanager
    def tracking(cls, tag: SparkJobTag) -> Iterator[None]:
        with cls._lock:
            cls._tags.add(tag)
        try:
            yield
        finally:
            with cls._lock:
                cls._tags.discard(tag)

    @classmethod
    def interrupt_all(cls) -> None:
        with cls._lock:
            tags = list(cls._tags)
        for tag in tags:
            tag.interrupt()


def run_python_model(spark: "SparkSession", compiled_code: str, timeout: float, name: str) -> None:
    """Execute a compiled Python model in-process, like a notebook cell on this cluster."""
    job_tag = SparkJobTag(spark)
    model_globals = _python_model_globals(spark)
    errors: list[BaseException] = []

    def run() -> None:
        try:
            with job_tag.applied():
                exec(compile(compiled_code, f"<python model {name}>", "exec"), model_globals)
        except BaseException as e:
            errors.append(e)

    # A separate thread lets the model time out like the other submission methods do.
    thread = threading.Thread(target=run, name=f"dbt-python-model-{name}", daemon=True)
    with SessionRunTracker.tracking(job_tag):
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            job_tag.interrupt()
            raise DbtRuntimeError(
                f"Python model {name} did not finish within {timeout} seconds; its Spark jobs "
                "were cancelled"
            )

    if errors:
        error = errors[0]
        trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        raise DbtRuntimeError(f"Python model {name} failed:\n{trace}") from error


def _python_model_globals(spark: "SparkSession") -> dict[str, Any]:
    model_globals: dict[str, Any] = {"__name__": "__main__", "spark": spark}
    # `pyspark.dbutils` is only present on Databricks Runtime.
    if DBUtils is not None:
        try:
            model_globals["dbutils"] = DBUtils(spark)
        except Exception as e:
            logger.debug(f"dbutils is not available to Python models: {e}")
    return model_globals


def _describe_field(field: "StructField") -> tuple:
    # Matches the SQL connector: base type name, with precision and scale only for decimals.
    type_name = TYPE_NAME_SEPARATOR_REGEX.split(field.dataType.simpleString(), maxsplit=1)[0]
    precision = getattr(field.dataType, "precision", None)
    scale = getattr(field.dataType, "scale", None)
    return (field.name, type_name, None, None, precision, scale, field.nullable)


def _string_columns(*names: str) -> list[tuple]:
    return [(name, "string", None, None, None, None, True) for name in names]


class SessionCursorWrapper:
    """Cursor over the result of a statement run on a SparkSession."""

    def __init__(self, rows: list[tuple], description: Optional[list[tuple]]):
        self._rows = rows
        self._description = description
        self._position = 0
        self.open = True
        self.query_id: Optional[str] = None

    @classmethod
    def from_dataframe(cls, df: "DataFrame") -> "SessionCursorWrapper":
        # Collect now: Spark plans queries lazily, and a statement must run even when dbt
        # never fetches its result, as it does with the SQL connector.
        rows = [tuple(row) for row in df.collect()]
        return cls(rows, [_describe_field(field) for field in df.schema.fields])

    @property
    def description(self) -> Optional[list[tuple]]:
        return self._description

    def fetchall(self) -> Sequence[tuple]:
        return self._fetch(len(self._rows))

    def fetchone(self) -> Optional[tuple]:
        rows = self._fetch(1)
        return rows[0] if rows else None

    def fetchmany(self, size: int) -> Sequence[tuple]:
        return self._fetch(size)

    def _fetch(self, size: int) -> list[tuple]:
        if not self.open:
            raise DbtRuntimeError("Attempting to execute on a closed cursor")
        rows = self._rows[self._position : self._position + size]
        self._position += len(rows)
        return rows

    def get_response(self) -> AdapterResponse:
        return DatabricksAdapterResponse.from_cursor(self)

    def cancel(self) -> None:
        self.close()

    def close(self) -> None:
        self.open = False
        self._rows = []

    def __str__(self) -> str:
        return f"SessionCursor(rows={len(self._rows)})"

    def __enter__(self) -> "SessionCursorWrapper":
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self.close()
        return exc_val is None


class DatabricksSessionHandle:
    """Connection handle that runs SQL on a SparkSession, with the DatabricksHandle interface."""

    def __init__(self, spark: "SparkSession"):
        self._spark = spark
        self.open = True
        self.session_id = f"spark-{uuid.uuid4().hex}"
        self._cursor: Optional[SessionCursorWrapper] = None
        self._dbr_version: Optional[tuple[int, int]] = None
        self._job_tag = SparkJobTag(spark)

    @staticmethod
    def create(credentials: DatabricksCredentials) -> "DatabricksSessionHandle":
        handle = DatabricksSessionHandle(create_spark_session(credentials))
        logger.debug(f"{handle} - Created")
        return handle

    @property
    def dbr_version(self) -> tuple[int, int]:
        if self._dbr_version is None:
            self._dbr_version = detect_dbr_version(self._spark)
        return self._dbr_version

    def execute(self, sql: str, bindings: Optional[Sequence[Any]] = None) -> SessionCursorWrapper:
        sql = SqlUtils.clean_sql(sql)
        params = SqlUtils.translate_bindings(bindings)
        if params:
            sql = render_bindings(sql, params)
        return self._run(lambda: SessionCursorWrapper.from_dataframe(self._spark.sql(sql)))

    def list_schemas(self, database: str, schema: Optional[str] = None) -> SessionCursorWrapper:
        def list_schemas() -> SessionCursorWrapper:
            rows = self._spark.sql(f"SHOW SCHEMAS IN {quote_identifier(database)}").collect()
            names = [row[0] for row in rows]
            if schema is not None:
                names = [name for name in names if name.lower() == schema.lower()]
            return SessionCursorWrapper(
                [(name, database) for name in names],
                _string_columns("TABLE_SCHEM", "TABLE_CATALOG"),
            )

        return self._run(list_schemas)

    def list_tables(self, database: str, schema: str) -> SessionCursorWrapper:
        def list_tables() -> SessionCursorWrapper:
            namespace = f"{quote_identifier(database)}.{quote_identifier(schema)}"
            rows = self._spark.sql(f"SHOW TABLES IN {namespace}").collect()
            # SHOW TABLES also lists the session's temporary views. It reports no table type,
            # so the adapter resolves views itself.
            return SessionCursorWrapper(
                [
                    (database, schema, row["tableName"], None)
                    for row in rows
                    if not row["isTemporary"]
                ],
                _string_columns("TABLE_CAT", "TABLE_SCHEM", "TABLE_NAME", "TABLE_TYPE"),
            )

        return self._run(list_tables)

    def _run(self, op: Callable[[], SessionCursorWrapper]) -> SessionCursorWrapper:
        if not self.open:
            raise DbtRuntimeError("Attempting to execute on a closed connection")
        if self._cursor:
            self._cursor.close()
        with self._job_tag.applied():
            self._cursor = op()
        return self._cursor

    def cancel(self) -> None:
        if self.open:
            self.open = False
            logger.debug(f"{self} - Cancelling")
            self._job_tag.interrupt()

    def close(self) -> None:
        # The SparkSession belongs to the cluster, so closing only releases this handle.
        if self.open:
            self.open = False
            logger.debug(f"{self} - Closing")
            if self._cursor:
                self._cursor.close()

    def rollback(self) -> None:
        logger.debug("NotImplemented: rollback")

    def __str__(self) -> str:
        return f"SparkSession(session-id={self.session_id})"
