# test_runner/config.py

import os
from pathlib import Path
from dotenv import load_dotenv

# Load envs from .env
load_dotenv()

# Alquimia
ALQUIMIA_API_KEY = os.environ.get("ALQUIMIA_API_KEY")
ALQUIMIA_URL = os.environ.get("ALQUIMIA_URL")
ALQUIMIA_VERSION = os.environ.get("ALQUIMIA_VERSION", "")
AGENT_ID = os.environ.get("AGENT_ID")
CHANNEL_ID = os.environ.get("CHANNEL_ID")

# Storage Configuration
TEST_STORAGE_BACKEND = os.getenv("TEST_STORAGE_BACKEND", "local")  # lakefs|local
TEST_RESULT_LANGUAGE = os.getenv("TEST_RESULT_LANGUAGE", "english")

# LakeFS (optional - only required when TEST_STORAGE_BACKEND=lakefs)
LAKEFS_HOST = os.getenv("LAKEFS_HOST", "")
LAKEFS_USERNAME = os.getenv("LAKEFS_USERNAME", "")
LAKEFS_PASSWORD = os.getenv("LAKEFS_PASSWORD", "")
LAKEFS_REPO_ID = os.getenv("LAKEFS_REPO_ID", "")

# Test Configuration
# Comma-separated list of test suites to run (e.g., "prompt_injection,toxicity,custom")
# If empty or not set, all test suites will be run
TEST_SUITES_TO_RUN = os.getenv("TEST_SUITES_TO_RUN", "").strip()
ENABLED_TEST_SUITES = [s.strip() for s in TEST_SUITES_TO_RUN.split(",") if s.strip()] if TEST_SUITES_TO_RUN else []

# Paths
TESTS_DIR = Path("./tests")
RESULTS_DIR = Path("./results")