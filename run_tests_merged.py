"""
Submit insurance-optimise pytest suite to Databricks serverless compute.
Includes the merged stochastic and plotting modules from rate-optimiser.
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
WORKSPACE_FOLDER = "/Workspace/insurance-optimise-merged"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_pytest"
BASE = "/home/ralph/insurance-optimise"


def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# Source files
src_files = {
    "src/insurance_optimise/__init__.py":      read_file(f"{BASE}/src/insurance_optimise/__init__.py"),
    "src/insurance_optimise/result.py":        read_file(f"{BASE}/src/insurance_optimise/result.py"),
    "src/insurance_optimise/constraints.py":   read_file(f"{BASE}/src/insurance_optimise/constraints.py"),
    "src/insurance_optimise/audit.py":         read_file(f"{BASE}/src/insurance_optimise/audit.py"),
    "src/insurance_optimise/optimiser.py":     read_file(f"{BASE}/src/insurance_optimise/optimiser.py"),
    "src/insurance_optimise/frontier.py":      read_file(f"{BASE}/src/insurance_optimise/frontier.py"),
    "src/insurance_optimise/scenarios.py":     read_file(f"{BASE}/src/insurance_optimise/scenarios.py"),
    "src/insurance_optimise/_demand_model.py": read_file(f"{BASE}/src/insurance_optimise/_demand_model.py"),
    "src/insurance_optimise/stochastic.py":    read_file(f"{BASE}/src/insurance_optimise/stochastic.py"),
    "src/insurance_optimise/plotting.py":      read_file(f"{BASE}/src/insurance_optimise/plotting.py"),
    "src/insurance_optimise/demand/__init__.py":      read_file(f"{BASE}/src/insurance_optimise/demand/__init__.py"),
    "src/insurance_optimise/demand/_types.py":        read_file(f"{BASE}/src/insurance_optimise/demand/_types.py"),
    "src/insurance_optimise/demand/compliance.py":    read_file(f"{BASE}/src/insurance_optimise/demand/compliance.py"),
    "src/insurance_optimise/demand/conversion.py":    read_file(f"{BASE}/src/insurance_optimise/demand/conversion.py"),
    "src/insurance_optimise/demand/datasets.py":      read_file(f"{BASE}/src/insurance_optimise/demand/datasets.py"),
    "src/insurance_optimise/demand/demand_curve.py":  read_file(f"{BASE}/src/insurance_optimise/demand/demand_curve.py"),
    "src/insurance_optimise/demand/elasticity.py":    read_file(f"{BASE}/src/insurance_optimise/demand/elasticity.py"),
    "src/insurance_optimise/demand/optimiser.py":     read_file(f"{BASE}/src/insurance_optimise/demand/optimiser.py"),
    "src/insurance_optimise/demand/retention.py":     read_file(f"{BASE}/src/insurance_optimise/demand/retention.py"),
}

# Test files
test_files = {
    "tests/conftest.py":          read_file(f"{BASE}/tests/conftest.py"),
    "tests/test_constraints.py":  read_file(f"{BASE}/tests/test_constraints.py"),
    "tests/test_optimiser.py":    read_file(f"{BASE}/tests/test_optimiser.py"),
    "tests/test_result.py":       read_file(f"{BASE}/tests/test_result.py"),
    "tests/test_scenarios.py":    read_file(f"{BASE}/tests/test_scenarios.py"),
    "tests/test_frontier.py":     read_file(f"{BASE}/tests/test_frontier.py"),
    "tests/test_integration.py":  read_file(f"{BASE}/tests/test_integration.py"),
    "tests/test_audit.py":        read_file(f"{BASE}/tests/test_audit.py"),
    "tests/test_demand.py":       read_file(f"{BASE}/tests/test_demand.py"),
    "tests/test_stochastic.py":   read_file(f"{BASE}/tests/test_stochastic.py"),
    "tests/test_plotting.py":     read_file(f"{BASE}/tests/test_plotting.py"),
    "tests/demand/__init__.py":           read_file(f"{BASE}/tests/demand/__init__.py"),
    "tests/demand/test_compliance.py":    read_file(f"{BASE}/tests/demand/test_compliance.py"),
    "tests/demand/test_conversion.py":    read_file(f"{BASE}/tests/demand/test_conversion.py"),
    "tests/demand/test_datasets.py":      read_file(f"{BASE}/tests/demand/test_datasets.py"),
    "tests/demand/test_demand_curve.py":  read_file(f"{BASE}/tests/demand/test_demand_curve.py"),
    "tests/demand/test_optimiser.py":     read_file(f"{BASE}/tests/demand/test_optimiser.py"),
    "tests/demand/test_retention.py":     read_file(f"{BASE}/tests/demand/test_retention.py"),
}

all_files = {**src_files, **test_files}
files_json = json.dumps(all_files)

PYPROJECT = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "insurance-optimise"
version = "0.3.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "polars>=1.0",
    "scipy>=1.10",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
]

[tool.hatch.build.targets.wheel]
packages = ["src/insurance_optimise"]
"""
pyproject_json = json.dumps(PYPROJECT)

NOTEBOOK_SOURCE = f"""# Databricks notebook source
# MAGIC %pip install polars>=1.0 numpy>=1.24 scipy>=1.10 pytest>=7.0 hatchling pandas>=2.0 scikit-learn>=1.3 statsmodels>=0.14 pyarrow>=14.0 matplotlib>=3.6 --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
pkg_dir = f"/tmp/insurance_optimise_{{pkg_id}}"
os.makedirs(pkg_dir, exist_ok=True)

FILES_JSON = {files_json!r}
PYPROJECT_CONTENT = {pyproject_json!r}

files_map = json.loads(FILES_JSON)
pyproject_src = json.loads(PYPROJECT_CONTENT)

# Write each file preserving directory structure
for rel_path, content in files_map.items():
    full_path = os.path.join(pkg_dir, rel_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)

with open(os.path.join(pkg_dir, "pyproject.toml"), "w") as f:
    f.write(pyproject_src)

print(f"Written {{len(files_map) + 1}} files to {{pkg_dir}}")
# Show structure
for dirpath, dirnames, filenames in os.walk(pkg_dir):
    level = dirpath.replace(pkg_dir, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{{indent}}{{os.path.basename(dirpath)}}/")
    subindent = ' ' * 2 * (level + 1)
    for f2 in filenames:
        print(f"{{subindent}}{{f2}}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", pkg_dir, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install STDERR:", r.stderr[:2000])
    raise RuntimeError("pip install failed")
print("insurance-optimise installed OK")

# Quick import check
import importlib
spec = importlib.util.find_spec("insurance_optimise")
print(f"Module found at: {{spec.origin if spec else 'NOT FOUND'}}")

# COMMAND ----------

tests_dir = os.path.join(pkg_dir, "tests")
r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider", "--collect-only"],
    capture_output=True, text=True, cwd=pkg_dir
)
print("=== COLLECTION CHECK ===")
print(r.stdout[-5000:])
if r.stderr:
    print("STDERR:", r.stderr[-1000:])

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=pkg_dir
)

print(r.stdout[-15000:])
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


# Ensure workspace folder
print(f"Creating workspace folder {WORKSPACE_FOLDER} ...")
try:
    api_call("POST", "api/2.0/workspace/mkdirs", {"path": WORKSPACE_FOLDER})
    print("Folder ready.")
except RuntimeError as exc:
    print(f"mkdirs note: {exc}")

# Upload notebook
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

# Submit serverless run
print(f"Submitting serverless run {RUN_ID} ...")
submit_body = {
    "run_name": f"insurance-optimise-merged-pytest-{RUN_ID}",
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

# Poll
print("Polling ...")
lc, rs = "PENDING", "-"
for i in range(120):
    time.sleep(20)
    run_state = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    lc = run_state.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = run_state.get("state", {}).get("result_state", "-")
    print(f"  [{i * 20}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal state: {lc} / {rs}")

# Fetch task output
try:
    run_info = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    tasks = run_info.get("tasks", [])
    for t in tasks:
        task_run_id = t.get("run_id")
        if task_run_id:
            out = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={task_run_id}")
            nb_result = out.get("notebook_output", {}).get("result", "")
            logs = out.get("logs", "")
            error = out.get("error", "")
            if nb_result:
                print(f"\nNotebook exit: {nb_result}")
            if error:
                print(f"Error: {error}")
            if logs:
                print(f"\nLogs:\n{logs[-15000:]}")
            # Check notebook exit value, not just job state
            if "ALL TESTS PASSED" in nb_result:
                print("\n=== PASS: All tests passed on Databricks. ===")
                sys.exit(0)
            else:
                print(f"\n=== FAIL: Notebook exit='{nb_result}' ===")
                sys.exit(1)
except Exception as e:
    print(f"Could not fetch output: {e}")

# Fallback to job state
if rs == "SUCCESS":
    print("\n=== Job state SUCCESS (could not verify test results) ===")
    sys.exit(0)
else:
    print(f"\n=== FAIL: Run ended with state {rs}. ===")
    sys.exit(1)
