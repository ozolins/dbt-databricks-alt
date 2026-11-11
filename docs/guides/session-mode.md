# Running dbt on a job cluster with session mode

_Last updated: 2026-11-11_

With `method: session`, dbt-databricks runs SQL and Python models on the SparkSession of the
Databricks cluster that dbt itself is running on, instead of connecting to a SQL warehouse or
all-purpose cluster through the Databricks SQL connector. A whole project can then run inside a
single job on job compute.

Use the default `method: dbsql` unless you specifically need this: SQL warehouses remain the
recommended compute for dbt SQL models.

## Requirements

- dbt must run **in the Python process of the cluster's driver**, so that it can use the cluster's
  SparkSession: a notebook task, a Python script task, or a Python wheel task that calls dbt
  through `dbtRunner` (see [Running dbt](#running-dbt)). The `dbt` task type does not use session
  mode; it runs the dbt CLI against a SQL warehouse.
- `dbt-databricks` (and therefore `dbt-core`) installed as a library of that cluster.
- Dedicated (single user) access mode is recommended. On standard (shared) access mode and
  serverless compute, Python runs on Spark Connect, where dbt connections cannot be isolated from
  each other (see [Session isolation](#session-isolation)).

## Configuration

```yaml
my_project:
  target: job
  outputs:
    job:
      type: databricks
      method: session
      catalog: main # optional, defaults to hive_metastore as with method: dbsql
      schema: analytics
      threads: 4
      session_properties: # optional, applied with spark.conf.set
        spark.sql.ansi.enabled: true
```

`host`, `http_path` and `token` are not needed. Session mode can also be selected without changing
the profile:

- setting the environment variable `DBT_DATABRICKS_SESSION_MODE=true` switches a profile without
  `method` to session mode, so one profile can serve both local development and job runs. Its
  `host` and credentials then only serve [REST API access](#rest-api-access);
- a profile without `method` and without `host` uses session mode automatically when dbt runs on
  Databricks Runtime.

An explicit `method` in the profile always wins.

## Running dbt

Call dbt in-process, for example from a notebook task:

```python
from dbt.cli.main import dbtRunner

result = dbtRunner().invoke(
    ["build", "--project-dir", "/Workspace/Users/me/my_project", "--profiles-dir", "/Workspace/Users/me/my_project"]
)
if not result.success:
    raise Exception(f"dbt failed: {result.exception}")
```

A Python script or wheel task can run the same code. Running the `dbt` CLI as a subprocess does
not give dbt access to the cluster's SparkSession.

## Python models

Python models default to the `session` submission method, which runs the compiled model in the
same Python process, with `spark` (and `dbutils` where Databricks Runtime provides it) in scope.

- `packages`, `additional_libs` and `environment_dependencies` are ignored with a warning: install
  libraries on the cluster running dbt instead.
- `timeout` is honoured. When it expires, or the run is cancelled, the model's Spark jobs are
  cancelled. Python code that is not running a Spark job cannot be interrupted and keeps running
  in the background until it finishes.
- Other submission methods (`job_cluster`, `all_purpose_cluster`, `serverless_cluster`,
  `workflow_job`) submit through the Databricks REST API and need `host` and credentials in the
  profile (see [REST API access](#rest-api-access)).

`submission_method: session` is only available with `method: session`.

## How session mode differs from `method: dbsql`

### Session isolation

Each dbt connection, and each Python model run, gets its own SparkSession created with
`newSession()`. `USE`, `SET` and temporary views stay isolated between threads, as they do between
SQL connector sessions, and dbt does not change the configuration of the notebook that called it.
Configuration set on the calling notebook's `spark` is therefore **not** visible to dbt: use
`session_properties` instead.

Spark Connect (standard access mode and serverless) cannot create new sessions, so there all
connections share the cluster's session. Avoid relying on session state (`USE`, `SET`, temporary
views) across models in that case.

### REST API access

Some features call the Databricks REST API, which needs a workspace `host` and credentials. In
session mode they are only available if the profile also sets `host` and `token` (or another
supported `auth_type`):

| Feature | Without `host` |
| --- | --- |
| Reading Unity Catalog table tags | Read from `information_schema` instead |
| Waiting for in-progress materialized view and streaming table refreshes | Skipped with a warning |
| Python submission methods other than `session` | Fail with a configuration error |

### Not supported

- The `databricks_compute` model config: every model runs on the cluster running dbt, so a model
  that sets it fails. A `compute` block in the profile is accepted and ignored.
- `query_tags`: they are a SQL warehouse feature and are not applied.
- Connection retries (`connect_retries`, `connect_timeout`, `retry_all`) have no effect.

### DBR capabilities

Version-dependent features are enabled from the cluster's Databricks Runtime version, read from
`spark.databricks.clusterUsageTags.sparkVersion`. When that is unavailable (for example on
serverless compute), the latest version is assumed, as for SQL warehouses.
