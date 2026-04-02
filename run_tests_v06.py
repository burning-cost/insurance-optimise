"""
Upload insurance-optimise v0.6.0 to Databricks and run full test suite.
"""
import os
import time
import base64
from pathlib import Path

env_path = Path.home() / ".config" / "burning-cost" / "databricks.env"
for line in env_path.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace, jobs

w = WorkspaceClient()

WORKSPACE_DIR = "/Workspace/insurance-optimise-v06"
REPO_ROOT = Path("/home/ralph/burning-cost/repos/insurance-optimise")


def upload_file(local_path: Path, remote_path: str):
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()
    w.workspace.import_(
        path=remote_path,
        content=encoded,
        format=workspace.ImportFormat.AUTO,
        overwrite=True,
    )


def upload_dir(local_dir: Path, remote_dir: str, extensions=(".py", ".toml")):
    for f in sorted(local_dir.rglob("*")):
        if f.is_file() and f.suffix in extensions and "__pycache__" not in str(f):
            rel = f.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel}".replace("\\", "/")
            parent = "/".join(remote_path.split("/")[:-1])
            try:
                w.workspace.mkdirs(path=parent)
            except Exception:
                pass
            upload_file(f, remote_path)
            print(f"  Uploaded: {rel}")


print("Uploading source files...")
try:
    w.workspace.mkdirs(path=WORKSPACE_DIR)
except Exception:
    pass

upload_dir(REPO_ROOT / "src", f"{WORKSPACE_DIR}/src")
upload_dir(REPO_ROOT / "tests", f"{WORKSPACE_DIR}/tests")
upload_file(REPO_ROOT / "pyproject.toml", f"{WORKSPACE_DIR}/pyproject.toml")

print("\nCreating test notebook...")

nb = """\
# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-optimise v0.6.0 -- pytest run

# COMMAND ----------
import subprocess, sys, os
deps = [
    'pandas>=2.0', 'scikit-learn>=1.3', 'statsmodels>=0.14',
    'pyarrow>=12.0', 'polars>=1.0', 'numpy>=1.25', 'scipy>=1.12',
    'pytest>=7.0',
]
r1 = subprocess.run(
    [sys.executable, '-m', 'pip', 'install'] + deps + ['--quiet'],
    capture_output=True, text=True
)
print('deps rc=', r1.returncode)
if r1.returncode != 0:
    print(r1.stderr[-2000:])

# COMMAND ----------
import subprocess, sys, os
env = dict(os.environ)
env['PYTHONDONTWRITEBYTECODE'] = '1'
existing_path = env.get('PYTHONPATH', '')
env['PYTHONPATH'] = '/Workspace/insurance-optimise-v06/src' + (':' + existing_path if existing_path else '')
r3 = subprocess.run(
    [
        sys.executable, '-m', 'pytest',
        '/Workspace/insurance-optimise-v06/tests',
        '-v', '--tb=short', '--no-header', '-p', 'no:cacheprovider',
    ],
    capture_output=True, text=True,
    cwd='/Workspace/insurance-optimise-v06',
    env=env,
)
out = r3.stdout
if len(out) > 10000:
    out = out[-10000:]
if r3.stderr:
    out += '\\nSTDERR: ' + r3.stderr[-1000:]
out += '\\npytest rc=' + str(r3.returncode)
print(out)
dbutils.notebook.exit(out)
"""

notebook_b64 = base64.b64encode(nb.encode()).decode()
notebook_path = f"{WORKSPACE_DIR}/run_tests"
w.workspace.import_(
    path=notebook_path,
    content=notebook_b64,
    format=workspace.ImportFormat.SOURCE,
    language=workspace.Language.PYTHON,
    overwrite=True,
)
print(f"Notebook created at {notebook_path}")

print("\nSubmitting job...")

submit_resp = w.jobs.submit(
    run_name="insurance-optimise-v0.6.0-pytest",
    tasks=[
        jobs.SubmitTask(
            task_key="pytest",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
            ),
        )
    ],
)
run_id = submit_resp.run_id
print(f"Run ID: {run_id}")
print("Polling...")

while True:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    print(f"  {lc}")
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        result_state = state.result_state.value if state.result_state else "UNKNOWN"
        print(f"  Result: {result_state}")
        break
    time.sleep(15)

tasks = run_state.tasks or []
for task in tasks:
    task_run_id = task.run_id
    print(f"\nFetching output for task run {task_run_id}...")
    try:
        output = w.jobs.get_run_output(run_id=task_run_id)
        if output.notebook_output:
            nb_out = output.notebook_output
            print(f"  truncated={nb_out.truncated}")
            if nb_out.result:
                print("\n=== RESULT ===")
                print(nb_out.result)
        if output.error:
            print("\n=== ERROR ===")
            print(output.error)
        if output.error_trace:
            print("\n=== TRACE ===")
            print(output.error_trace[:5000])
    except Exception as e:
        print(f"Could not get output: {e}")

print("\nDone.")
