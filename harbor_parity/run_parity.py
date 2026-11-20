"""Original-side parity runner for the Harbor ether0-benchmark adapter.

Runs a CLI coding agent (Codex) on ether0-benchmark problems and scores the
answers with ether0's own reward code. The goal is an apples-to-apples
comparison with the Harbor adapter (harbor-index/adapters, src/ether0-benchmark).
Every step mirrors what Harbor does for one trial:

1. Data: the pinned test split, filtered to the 250 problems the adapter covers
   (reward functions whose reference answer ships with the dataset).
2. Prompt: ether0's LOOSE_XML_ANSWER_USER_PROMPT + problem text, followed by the
   same output-file sentence the adapter's instruction.md uses.
3. Environment: a fresh container per problem from the adapter's task image
   (environment/Dockerfile in this directory).
4. Agent: the same install sequence and `codex exec` invocation as Harbor's
   installed Codex agent (pinned version, same flags and config).
5. Scoring: the agent's /app/answer.txt is copied out and scored on the host with
   ether0.model_prompts.extract_answer_loose and
   ether0.rewards.EVAL_FUNCTIONS[fxn](yhat, y=answer_info, test=True), exactly
   as in ether0's published baseline.

Credentials are read from the environment (OPENAI_API_KEY, OPENAI_BASE_URL) and
are never written to the results directory.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shlex
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from ether0.model_prompts import LOOSE_XML_ANSWER_USER_PROMPT, extract_answer_loose
from ether0.models import RewardFunctionInfo
from ether0.rewards import EVAL_FUNCTIONS

HF_REPO_ID = "futurehouse/ether0-benchmark"
HF_REVISION = "c7d5e59960087f360bc32a5006bb994324b38c35"
HF_PARQUET = "data/test-00000-of-00001.parquet"
ADAPTED_REWARD_FUNCTIONS = {"str_eval", "product_eval", "formula_eval", "functional_group_eval"}

# Identical to the closing lines of the adapter's task-template/instruction.md.
OUTPUT_FILE_INSTRUCTION = (
    "Write your complete final response to the file `/app/answer.txt`. The file must "
    "contain your final answer inside `<answer></answer>` XML tags, for example "
    "`<answer>CCO</answer>`."
)

CODEX_HOME = "/installed-agent/codex-home"
ANSWER_PATH = "/app/answer.txt"


def build_instruction(problem: str) -> str:
    """Byte-identical to the adapter's rendered instruction.md."""
    return (
        f"{LOOSE_XML_ANSWER_USER_PROMPT}\n\n{problem.strip()}\n\n---\n\n"
        f"{OUTPUT_FILE_INSTRUCTION}\n"
    )


def load_problems(parquet_path: Path | None) -> list[dict]:
    if parquet_path is None:
        parquet_path = Path(
            hf_hub_download(HF_REPO_ID, HF_PARQUET, repo_type="dataset", revision=HF_REVISION)
        )
    rows = pq.read_table(parquet_path).to_pylist()
    keep = [r for r in rows if r["solution"].split("!:!")[0] in ADAPTED_REWARD_FUNCTIONS]
    keep.sort(key=lambda r: r["id"].lower())
    return keep


@dataclass
class TrialResult:
    problem_id: str
    problem_type: str
    reward: float | None
    error: str | None
    agent_exit_code: int | None
    duration_sec: float
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rt = args.runtime
        self.container_env = list(args.container_env or [])

    def sh(self, cmd: list[str], timeout: float | None = None, check: bool = True,
           input_text: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, input=input_text, capture_output=True, text=True,
                              timeout=timeout, check=check)

    def exec_in(self, name: str, script: str, env: dict[str, str] | None = None,
                timeout: float | None = None, check: bool = True) -> subprocess.CompletedProcess:
        env_args: list[str] = []
        for k, v in (env or {}).items():
            env_args += ["-e", f"{k}={v}"]
        return self.sh([self.rt, "exec", *env_args, name, "bash", "-c", script],
                       timeout=timeout, check=check)

    def install_codex(self, name: str) -> None:
        """Mirror Harbor's Codex agent install (system deps, nvm + Node 22, codex).

        Like Harbor's is_requested_version_installed(), skip everything when the
        requested Codex version is already present (e.g. the Dockerfile.codex image).
        """
        probe = self.exec_in(
            name, "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex --version",
            check=False)
        if probe.returncode == 0 and self.args.codex_version in probe.stdout:
            return
        missing = []
        for pkg, cmd in (("curl", "curl"), ("bash", "bash"), ("nodejs", "node"),
                         ("npm", "npm"), ("ripgrep", "rg")):
            if self.exec_in(name, f"command -v {cmd} >/dev/null 2>&1", check=False).returncode:
                missing.append(pkg)
        if missing:
            self.exec_in(name, f"apt-get update && apt-get install -y {shlex.join(missing)}",
                         env={"DEBIAN_FRONTEND": "noninteractive"}, timeout=1200)
        v = self.args.codex_version
        self.exec_in(
            name,
            "set -euo pipefail; "
            "curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash && "
            'export NVM_DIR="$HOME/.nvm" && '
            '\\. "$NVM_DIR/nvm.sh" || true && '
            "command -v nvm &>/dev/null || { echo 'Error: NVM failed to load' >&2; exit 1; } && "
            "nvm install 22 && nvm alias default 22 && npm -v && "
            f"npm install -g @openai/codex@{v} && codex --version",
            env={"NVM_NODEJS_ORG_MIRROR": "https://nodejs.org/dist"}, timeout=1200)
        self.exec_in(
            name,
            "for bin in node codex; do"
            '  BIN_PATH="$(which "$bin" 2>/dev/null || true)";'
            '  if [ -n "$BIN_PATH" ] && [ "$BIN_PATH" != "/usr/local/bin/$bin" ]; then'
            '    ln -sf "$BIN_PATH" "/usr/local/bin/$bin";'
            "  fi;"
            " done")

    def run_one(self, row: dict, out_dir: Path) -> TrialResult:
        fxn, answer_info, problem_type = row["solution"].split("!:!")
        name = f"ether0-parity-{row['id'][:8]}-{uuid.uuid4().hex[:6]}"
        trial_dir = out_dir / row["id"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        reward, error, agent_rc = None, None, None
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
        env_args: list[str] = []
        for kv in self.container_env:
            env_args += ["-e", kv]
        try:
            self.sh([self.rt, "run", "-d", "--name", name, *env_args, self.args.image,
                     "sleep", "infinity"], timeout=300)
            self.install_codex(name)
            base_url = os.environ.get("OPENAI_BASE_URL", "")
            config_toml = f'openai_base_url = "{base_url}"\n' if base_url else ""
            self.exec_in(
                name,
                f"mkdir -p {CODEX_HOME} && cat > {CODEX_HOME}/config.toml <<'EOF'\n{config_toml}EOF\n"
                f"cat > {CODEX_HOME}/auth.json <<EOF\n{{\n  \"OPENAI_API_KEY\": \"${{OPENAI_API_KEY}}\"\n}}\nEOF\n"
                f"chmod 600 {CODEX_HOME}/auth.json",
                env={"OPENAI_API_KEY": os.environ["OPENAI_API_KEY"]})
            instruction = build_instruction(row["problem"])
            (trial_dir / "instruction.md").write_text(instruction)
            cmd = (
                "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; "
                "CODEX_API_KEY= codex exec "
                "--dangerously-bypass-approvals-and-sandbox "
                "--skip-git-repo-check "
                f"--model {shlex.quote(self.args.model)} "
                "--json "
                "--enable unified_exec "
                f"-c model_reasoning_effort={self.args.reasoning_effort} "
                "-- "
                f"{shlex.quote(instruction)} "
                "2>&1 </dev/null"
            )
            agent_env = {"CODEX_HOME": CODEX_HOME,
                         "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
                         "OPENAI_BASE_URL": base_url}
            try:
                proc = self.exec_in(name, cmd, env=agent_env,
                                    timeout=self.args.agent_timeout, check=False)
                agent_rc = proc.returncode
                (trial_dir / "codex.jsonl").write_text(proc.stdout)
                for line in proc.stdout.splitlines():
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "turn.completed":
                        u = ev.get("usage") or {}
                        for k in usage:
                            usage[k] += int(u.get(k) or 0)
            except subprocess.TimeoutExpired:
                error = "AgentTimeoutError"
            # Score whatever the agent left in /app/answer.txt (missing file -> reward 0).
            got = self.sh([self.rt, "exec", name, "cat", ANSWER_PATH], check=False)
            response = got.stdout if got.returncode == 0 else ""
            (trial_dir / "answer.txt").write_text(response)
            yhat = extract_answer_loose(response)
            info = RewardFunctionInfo.model_validate(row["solution"])
            try:
                reward = float(EVAL_FUNCTIONS[info.fxn_name](yhat=yhat, y=info.answer_info,
                                                            test=True)) if yhat else 0.0
            except Exception as exc:  # same policy as the adapter's verifier
                reward = 0.0
                error = error or f"RewardFunctionError: {type(exc).__name__}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
        finally:
            self.sh([self.rt, "rm", "-f", name], check=False, timeout=120)
        result = TrialResult(row["id"], problem_type, reward, error, agent_rc,
                             round(time.time() - t0, 1), **usage)
        (trial_dir / "result.json").write_text(json.dumps(asdict(result), indent=2))
        return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run-name", required=True)
    p.add_argument("--results-dir", type=Path, default=Path("harbor_parity/results"))
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument("--codex-version", default="0.166.0")
    p.add_argument("--reasoning-effort", default="medium")
    p.add_argument("--image", default="ether0-parity-codex:0.166.0",
                   help="Image built from harbor_parity/environment/Dockerfile.codex "
                        "(the adapter's task image with Codex preinstalled)")
    p.add_argument("--runtime", default="docker", help="docker or podman")
    p.add_argument("--container-env", action="append",
                   help="KEY=VALUE passed to every container (e.g. proxy settings)")
    p.add_argument("--n-concurrent", type=int, default=4)
    p.add_argument("--agent-timeout", type=float, default=1800.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task-ids", nargs="+", default=None)
    p.add_argument("--parquet-path", type=Path, default=None)
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set (and OPENAI_BASE_URL for a proxy).")
    rows = load_problems(args.parquet_path)
    if args.task_ids:
        wanted = {t.lower() for t in args.task_ids}
        rows = [r for r in rows if r["id"].lower() in wanted]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = args.results_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_name": args.run_name, "started_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "agent": f"codex@{args.codex_version}",
        "reasoning_effort": args.reasoning_effort, "image": args.image,
        "agent_timeout_sec": args.agent_timeout, "n_problems": len(rows),
        "dataset": f"{HF_REPO_ID}@{HF_REVISION}",
    }
    runner = Runner(args)
    results: list[TrialResult] = []
    with cf.ThreadPoolExecutor(args.n_concurrent) as ex:
        futs = {ex.submit(runner.run_one, r, out_dir): r["id"] for r in rows}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(f"[{i}/{len(rows)}] {res.problem_id} reward={res.reward} "
                  f"error={res.error} {res.duration_sec:.0f}s", flush=True)
    # Infrastructure failures (reward None) are reported separately and must be rerun;
    # they are not counted as zeros in the accuracy denominator.
    scored = [r.reward for r in results if r.reward is not None]
    meta.update({
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "n_scored": len(scored),
        "n_infra_failures": sum(1 for r in results if r.reward is None),
        "infra_failed_ids": sorted(r.problem_id for r in results if r.reward is None),
        "n_with_error_note": sum(1 for r in results if r.error),
        "accuracy": sum(scored) / len(scored) if scored else None,
        "total_input_tokens": sum(r.input_tokens for r in results),
        "total_cached_input_tokens": sum(r.cached_input_tokens for r in results),
        "total_output_tokens": sum(r.output_tokens for r in results),
    })
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
