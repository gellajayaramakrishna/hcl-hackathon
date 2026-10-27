# Backend package. macOS fix: torch (embeddings) and LightGBM (router) each ship their own
# OpenMP library and segfault together unless LightGBM is loaded first and OpenMP uses 1 thread.
# This file runs before any backend module, so every entry point gets the fix.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import lightgbm  # noqa: E402,F401  (must be imported before torch)
