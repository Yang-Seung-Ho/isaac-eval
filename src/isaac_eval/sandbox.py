from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_NETWORK_BLOCK_SNIPPET = """
import socket as _socket
def _blocked(*a, **k):
    raise OSError("network disabled in sandbox")
_socket.socket = _blocked
_socket.create_connection = _blocked
"""

_RUNNER_TEMPLATE = '''
import sys, json, resource
try:
    resource.setrlimit(resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes}))
except (ValueError, OSError):
    pass  # some platforms (e.g. macOS) reject RLIMIT_AS after the interpreter is already loaded
{network_block}
import pandas as pd

df = pd.read_excel({dataset_path!r}, sheet_name={sheet!r})
{rename_line}
USER_CODE = {code!r}
exec(compile(USER_CODE, "<user_code>", "exec"), {{"pd": pd, "json": json, "df": df}})
'''


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    value: object
    error: str | None
    traceback: str | None
    elapsed_ms: int
    attempt: int


def run_code(code: str, dataset_path: str, sheet: str, cfg: dict, attempt: int = 1) -> SandboxResult:
    sb_cfg = cfg["sandbox"]
    mem_bytes = sb_cfg["memory_mb"] * 1024 * 1024
    network_block = "" if sb_cfg.get("network", False) else _NETWORK_BLOCK_SNIPPET
    # 첫 열 이름 정규화는 데이터셋별 옵션이다. 키가 없으면 기존 제주 동작("Month")을
    # 그대로 유지하고(하위호환), 값이 None/빈값이면 rename 을 건너뛴다(ASV 등 타 도메인).
    rename_to = cfg["paths"].get("rename_first_col_to", "Month")
    rename_line = (
        f'df = df.rename(columns={{df.columns[0]: {rename_to!r}}})' if rename_to else ""
    )
    script = _RUNNER_TEMPLATE.format(
        mem_bytes=mem_bytes,
        network_block=network_block,
        dataset_path=os.path.abspath(dataset_path),
        sheet=sheet,
        rename_line=rename_line,
        code=code,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "run.py"
        script_path.write_text(script, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", "")}
        start = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=sb_cfg["timeout_s"],
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            return SandboxResult(False, "", None, "timeout exceeded", None, elapsed_ms, attempt)
        elapsed_ms = int((time.monotonic() - start) * 1000)
        if proc.returncode != 0:
            return SandboxResult(False, proc.stdout, None, "execution failed", proc.stderr, elapsed_ms, attempt)
        last_line = next((l for l in reversed(proc.stdout.splitlines()) if l.strip()), "")
        try:
            value = json.loads(last_line)
        except json.JSONDecodeError as e:
            return SandboxResult(False, proc.stdout, None, f"invalid json output: {e}", proc.stdout, elapsed_ms, attempt)
        return SandboxResult(True, proc.stdout, value, None, None, elapsed_ms, attempt)


def run_with_retry(code_generator: Callable[[str | None], str], dataset_path: str, sheet: str, cfg: dict) -> list[SandboxResult]:
    max_attempts = cfg["sandbox"]["max_attempts"]
    results: list[SandboxResult] = []
    prev_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        code = code_generator(prev_error)
        result = run_code(code, dataset_path, sheet, cfg, attempt=attempt)
        results.append(result)
        if result.ok:
            break
        prev_error = result.traceback or result.error
    return results
