"""
Submit insurance-optimise pytest suite to Databricks serverless compute.

Handles the full package layout including the demand/ subdirectory.
"""

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

# ---------------------------------------------------------------------------
# Load credentials
# ---------------------------------------------------------------------------
env_path = os.path.expanduser("~/.config/burning-cost/databricks.env")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

api_base = os.environ["DATABRICKS_HOST"].rstrip("/")
token = os.environ["DATABRICKS_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

RUN_ID = uuid.uuid4().hex[:8]
WORKSPACE_FOLDER = "/Workspace/insurance-optimise-v2"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_pytest"

BASE = "/home/ralph/burning-cost/repos/insurance-optimise"


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Collect all source and test files
# ---------------------------------------------------------------------------
src_pkg = f"{BASE}/src/insurance_optimise"
tests_dir = f"{BASE}/tests"

# Top-level package files
src_files = {}
for fname in os.listdir(src_pkg):
    if fname.endswith(".py") and not fname.startswith("__pycache__"):
        src_files[fname] = read_file(f"{src_pkg}/{fname}")

# demand/ subdirectory
demand_src_files = {}
demand_src_dir = f"{src_pkg}/demand"
for fname in os.listdir(demand_src_dir):
    if fname.endswith(".py") and not fname.startswith("__pycache__"):
        demand_src_files[fname] = read_file(f"{demand_src_dir}/{fname}")

# Top-level tests
test_files = {}
for fname in os.listdir(tests_dir):
    if fname.endswith(".py") and not fname.startswith("__pycache__"):
        test_files[fname] = read_file(f"{tests_dir}/{fname}")

# tests/demand/ subdirectory
demand_test_files = {}
demand_test_dir = f"{tests_dir}/demand"
if os.path.isdir(demand_test_dir):
    for fname in os.listdir(demand_test_dir):
        if fname.endswith(".py") and not fname.startswith("__pycache__"):
            demand_test_files[fname] = read_file(f"{demand_test_dir}/{fname}")

print(f"Collected: {len(src_files)} src, {len(demand_src_files)} demand-src, "
      f"{len(test_files)} tests, {len(demand_test_files)} demand-tests")

all_data = {
    "src_files": src_files,
    "demand_src_files": demand_src_files,
    "test_files": test_files,
    "demand_test_files": demand_test_files,
}
all_data_json = json.dumps(all_data)

# ---------------------------------------------------------------------------
# Build notebook source
# ---------------------------------------------------------------------------
NOTEBOOK_SOURCE = f"""# Databricks notebook source
# MAGIC %pip install polars>=1.0 numpy>=1.24 scipy>=1.10 pytest>=7.0 hatchling statsmodels>=0.14 scikit-learn>=1.3 --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/ins_opt_{{pkg_id}}"
src_dir = f"{{pkg_dir}}/src/insurance_optimise"
demand_src_dir = f"{{src_dir}}/demand"
tests_base = f"{{pkg_dir}}/tests"
demand_tests_dir = f"{{tests_base}}/demand"

for d in [src_dir, demand_src_dir, tests_base, demand_tests_dir]:
    os.makedirs(d, exist_ok=True)

DATA = json.loads({all_data_json!r})

for fname, content in DATA["src_files"].items():
    with open(f"{{src_dir}}/{{fname}}", "w") as fh:
        fh.write(content)

for fname, content in DATA["demand_src_files"].items():
    with open(f"{{demand_src_dir}}/{{fname}}", "w") as fh:
        fh.write(content)

for fname, content in DATA["test_files"].items():
    with open(f"{{tests_base}}/{{fname}}", "w") as fh:
        fh.write(content)

for fname, content in DATA["demand_test_files"].items():
    with open(f"{{demand_tests_dir}}/{{fname}}", "w") as fh:
        fh.write(content)

pyproject = \"\"\"[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-optimise"
version = "0.3.3"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "polars>=1.0",
    "scipy>=1.10",
    "statsmodels>=0.14",
    "scikit-learn>=1.3",
]

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_optimise"]
\"\"\"
with open(f"{{pkg_dir}}/pyproject.toml", "w") as fh:
    fh.write(pyproject)

print(f"Written package to {{pkg_dir}}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install error:", r.stderr[:2000])
    raise RuntimeError("pip install failed")
else:
    print("insurance-optimise installed OK")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_base, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider", "-x"],
    capture_output=True, text=True, cwd=pkg_dir
)

print(r.stdout[-12000:])
if r.stderr:
    print("STDERR:", r.stderr[-2000:])

if r.returncode == 0:
    print("\\n=== ALL TESTS PASSED ===")
    try:
        dbutils.notebook.exit("ALL TESTS PASSED")
    except NameError:
        pass
else:
    msg = f"TESTS FAILED (exit {{r.returncode}})"
    print(f"\\n=== {{msg}} ===")
    try:
        dbutils.notebook.exit(msg)
    except NameError:
        pass
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def api_call(method: str, endpoint: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(
        f"{api_base}/{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode()
        raise RuntimeError(f"API {method} {endpoint} failed {e.code}: {err_body}")


# ---------------------------------------------------------------------------
# Ensure workspace folder exists
# ---------------------------------------------------------------------------
print(f"Creating workspace folder {WORKSPACE_FOLDER} ...")
try:
    api_call("POST", "api/2.0/workspace/mkdirs", {"path": WORKSPACE_FOLDER})
    print("Folder ready.")
except RuntimeError as exc:
    print(f"mkdirs note: {exc}")

# ---------------------------------------------------------------------------
# Upload notebook
# ---------------------------------------------------------------------------
print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
notebook_b64 = base64.b64encode(NOTEBOOK_SOURCE.encode("utf-8")).decode("ascii")

api_call("POST", "api/2.0/workspace/import", {
    "path": NOTEBOOK_PATH,
    "format": "SOURCE",
    "language": "PYTHON",
    "content": notebook_b64,
    "overwrite": True,
})
print("Upload OK")

# ---------------------------------------------------------------------------
# Submit serverless run
# ---------------------------------------------------------------------------
print(f"Submitting serverless run {RUN_ID} ...")
submit_body = {
    "run_name": f"insurance-optimise-pytest-{RUN_ID}",
    "tasks": [
        {
            "task_key": "pytest",
            "notebook_task": {
                "notebook_path": NOTEBOOK_PATH,
                "source": "WORKSPACE",
            },
        }
    ],
}
result = api_call("POST", "api/2.1/jobs/runs/submit", submit_body)
run_id = result["run_id"]
print(f"Run submitted: run_id={run_id}")

# ---------------------------------------------------------------------------
# Poll
# ---------------------------------------------------------------------------
print("Polling ...")
lc, rs = "PENDING", "-"
for i in range(120):
    time.sleep(15)
    run_state = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    lc = run_state.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = run_state.get("state", {}).get("result_state", "-")
    print(f"  [{i * 15}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {lc} / {rs}")

# ---------------------------------------------------------------------------
# Fetch output
# ---------------------------------------------------------------------------
try:
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    notebook_result = output.get("notebook_output", {}).get("result", "")
    error = output.get("error", "")
    error_trace = output.get("error_trace", "")
    logs = output.get("logs", "")

    if notebook_result:
        print(f"\nExit value: {notebook_result}")
    if error:
        print(f"Error: {error}")
    if error_trace:
        print(f"Trace:\n{error_trace[:3000]}")
    if logs:
        print(f"\nLogs:\n{logs[-10000:]}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if rs == "SUCCESS":
    print("\n=== PASS: All tests completed on Databricks. ===")
    sys.exit(0)
else:
    print(f"\n=== FAIL: Run ended with state {rs}. ===")
    sys.exit(1)
