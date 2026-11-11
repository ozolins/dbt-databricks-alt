# Notice

`dbt-databricks-alt` is a modified version of
[`dbt-databricks`](https://github.com/databricks/dbt-databricks), copyright Databricks, Inc. and
contributors, licensed under the Apache License 2.0 (see [License.md](License.md)).

It is not affiliated with, endorsed by, or supported by Databricks or dbt Labs.

## Changes from upstream

Version 1.13.100 is based on `dbt-databricks` 1.13.0 (`main` at commit `31edf7fc`), with these
changes:

- Added session mode (`method: session`): `dbt/adapters/databricks/session.py`, and changes to
  `connections.py`, `credentials.py`, `impl.py` and `python_models/python_submissions.py` in the
  same directory.
- Renamed the distribution to `dbt-databricks-alt` (`pyproject.toml`) and the user agent to
  `dbt-databricks-alt/<version>` (`api_client.py`, `global_state.py`, `handle.py`).
- Added `docs/guides/session-mode.md`, and tests in `tests/unit/test_session.py` and
  `tests/unit/test_session_mode.py`.
