"""tscompare: unified harness for comparing LLM-era time-series forecasters.

Core idea: every method is a `Forecaster`; every benchmark task is a `Task`;
`runner.run` produces one JSON per (model, task); `aggregate` turns those into
the leaderboard-style tables used in the paper.
"""
__version__ = "0.1.0"
