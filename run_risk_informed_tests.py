"""
Submit the RiskInformedRetentionModel tests to Databricks serverless compute.

Uploads the full demand subpackage plus the new test file and runs pytest.
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
# Credentials
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
WORKSPACE_FOLDER = "/Workspace/insurance-optimise-ri"
NOTEBOOK_PATH = f"{WORKSPACE_FOLDER}/run_pytest_ri"

BASE = "/home/ralph/burning-cost/repos/insurance-optimise"


def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Source files — full demand subpackage + top-level modules
# ---------------------------------------------------------------------------
demand_src_dir = f"{BASE}/src/insurance_optimise/demand"
demand_src_files = {}
for fname in os.listdir(demand_src_dir):
    if fname.endswith(".py"):
        demand_src_files[f"demand/{fname}"] = read_file(f"{demand_src_dir}/{fname}")

top_src_files = {
    "top/__init__.py": read_file(f"{BASE}/src/insurance_optimise/__init__.py"),
    "top/constraints.py": read_file(f"{BASE}/src/insurance_optimise/constraints.py"),
    "top/result.py": read_file(f"{BASE}/src/insurance_optimise/result.py"),
    "top/optimiser.py": read_file(f"{BASE}/src/insurance_optimise/optimiser.py"),
    "top/frontier.py": read_file(f"{BASE}/src/insurance_optimise/frontier.py"),
    "top/scenarios.py": read_file(f"{BASE}/src/insurance_optimise/scenarios.py"),
    "top/_demand_model.py": read_file(f"{BASE}/src/insurance_optimise/_demand_model.py"),
    "top/pareto.py": read_file(f"{BASE}/src/insurance_optimise/pareto.py"),
    "top/pareto_front.py": read_file(f"{BASE}/src/insurance_optimise/pareto_front.py"),
    "top/stochastic.py": read_file(f"{BASE}/src/insurance_optimise/stochastic.py"),
    "top/model_quality.py": read_file(f"{BASE}/src/insurance_optimise/model_quality.py"),
    "top/audit.py": read_file(f"{BASE}/src/insurance_optimise/audit.py"),
    "top/plotting.py": read_file(f"{BASE}/src/insurance_optimise/plotting.py"),
}

test_files = {
    "tests/conftest.py": read_file(f"{BASE}/tests/conftest.py"),
    "tests/test_risk_informed.py": read_file(
        f"{BASE}/tests/demand/test_risk_informed_retention.py"
    ),
}

# Include pyproject.toml as a special key so the notebook can write it
pyproject_content = (
    "[build-system]\n"
    'requires = ["hatchling"]\n'
    'build-backend = "hatchling.build"\n'
    "\n"
    "[project]\n"
    'name = "insurance-optimise"\n'
    'version = "0.0.1"\n'
    'requires-python = ">=3.10"\n'
    "dependencies = [\n"
    '    "numpy>=1.24",\n'
    '    "polars>=0.20",\n'
    '    "scipy>=1.10",\n'
    '    "scikit-learn",\n'
    '    "statsmodels",\n'
    '    "pandas",\n'
    "]\n"
    "\n"
    "[tool.hatch.build.targets.wheel]\n"
    'packages = ["src/insurance_optimise"]\n'
)

all_files = {**demand_src_files, **top_src_files, **test_files}
# Embed pyproject separately so it doesn't need a prefix
all_files["_pyproject_/pyproject.toml"] = pyproject_content

files_json = json.dumps(all_files)

# ---------------------------------------------------------------------------
# Notebook source
# ---------------------------------------------------------------------------
NOTEBOOK_SOURCE = f"""# Databricks notebook source
# MAGIC %pip install polars>=0.20 numpy>=1.24 scipy>=1.10 scikit-learn pytest statsmodels --quiet

# COMMAND ----------

import json, os, sys, uuid, subprocess

pkg_id = uuid.uuid4().hex[:8]
root = f"/tmp/ins_opt_{{pkg_id}}"
src_top = f"{{root}}/src/insurance_optimise"
src_demand = f"{{src_top}}/demand"
tests_dir = f"{{root}}/tests"
os.makedirs(src_demand, exist_ok=True)
os.makedirs(tests_dir, exist_ok=True)

FILES_JSON = {files_json!r}
files_map = json.loads(FILES_JSON)

for key, content in files_map.items():
    prefix, fname = key.split("/", 1)
    if prefix == "demand":
        path = f"{{src_demand}}/{{fname}}"
    elif prefix == "top":
        path = f"{{src_top}}/{{fname}}"
    elif prefix == "_pyproject_":
        path = f"{{root}}/{{fname}}"
    else:
        path = f"{{tests_dir}}/{{fname}}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

print(f"Written {{len(files_map)}} files to {{root}}")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-e", root, "--quiet"],
    capture_output=True, text=True
)
if r.returncode != 0:
    print("Install error:", r.stderr[:2000])
    raise RuntimeError("Package install failed")
else:
    print("insurance-optimise installed OK")

# COMMAND ----------

r = subprocess.run(
    [sys.executable, "-m", "pytest", tests_dir, "-v", "--tb=short",
     "--no-header", "-p", "no:cacheprovider"],
    capture_output=True, text=True, cwd=root
)

print(r.stdout[-10000:])
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
# REST helpers
# ---------------------------------------------------------------------------

def api_call(method: str, endpoint: str, body=None):
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        f"{api_base}/{endpoint}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"API {method} {endpoint} failed {e.code}: {e.read().decode()}")


# ---------------------------------------------------------------------------
# Upload and run
# ---------------------------------------------------------------------------
print(f"Creating folder {WORKSPACE_FOLDER} ...")
try:
    api_call("POST", "api/2.0/workspace/mkdirs", {"path": WORKSPACE_FOLDER})
except RuntimeError as e:
    print(f"mkdirs note: {e}")

print(f"Uploading notebook to {NOTEBOOK_PATH} ...")
api_call("POST", "api/2.0/workspace/import", {
    "path": NOTEBOOK_PATH,
    "format": "SOURCE",
    "language": "PYTHON",
    "content": base64.b64encode(NOTEBOOK_SOURCE.encode()).decode(),
    "overwrite": True,
})
print("Upload OK")

print(f"Submitting run {RUN_ID} ...")
result = api_call("POST", "api/2.1/jobs/runs/submit", {
    "run_name": f"ri-retention-pytest-{RUN_ID}",
    "tasks": [{
        "task_key": "pytest",
        "notebook_task": {"notebook_path": NOTEBOOK_PATH, "source": "WORKSPACE"},
    }],
})
run_id = result["run_id"]
print(f"Run submitted: run_id={run_id}")

print("Polling ...")
lc, rs = "PENDING", "-"
for i in range(120):
    time.sleep(15)
    state = api_call("GET", f"api/2.1/jobs/runs/get?run_id={run_id}")
    lc = state.get("state", {}).get("life_cycle_state", "UNKNOWN")
    rs = state.get("state", {}).get("result_state", "-")
    print(f"  [{i * 15}s] {lc} / {rs}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        break

print(f"\nFinal: {lc} / {rs}")

try:
    output = api_call("GET", f"api/2.1/jobs/runs/get-output?run_id={run_id}")
    nb_result = output.get("notebook_output", {}).get("result", "")
    error = output.get("error", "")
    trace = output.get("error_trace", "")
    logs = output.get("logs", "")
    if nb_result:
        print(f"\nExit value: {nb_result}")
    if error:
        print(f"Error: {error}")
    if trace:
        print(f"Trace:\n{trace[:3000]}")
    if logs:
        print(f"\nLogs:\n{logs[-10000:]}")
except Exception as e:
    print(f"Could not fetch output: {e}")

if rs == "SUCCESS":
    print("\n=== PASS ===")
    sys.exit(0)
else:
    print(f"\n=== FAIL: {rs} ===")
    sys.exit(1)
