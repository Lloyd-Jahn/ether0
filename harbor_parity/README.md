# Harbor parity runner (original-benchmark side)

This directory adds the **original-benchmark side** of the parity experiment for the
Harbor adapter of ether0-benchmark
([harbor-index/adapters PR #13](https://github.com/harbor-index/adapters/pull/13),
`src/ether0-benchmark`). It runs the same CLI agent the Harbor side runs, on the same
problems, and scores the answers with **this repository's own reward code**
(`ether0.model_prompts.extract_answer_loose` +
`ether0.rewards.EVAL_FUNCTIONS[fxn](yhat, y=answer_info, test=True)`), exactly as in
ether0's published baseline.

The branch is based on commit `b77e0d8eb06f62c8528da72bff22d4c7fee11371`, the commit the
Harbor verifier pins.

## What is held identical to the Harbor side

| Aspect | Setting |
|--------|---------|
| Problems | `futurehouse/ether0-benchmark` test split @ `c7d5e59960087f360bc32a5006bb994324b38c35`, the 250 problems graded by `str_eval`, `product_eval`, `formula_eval`, `functional_group_eval` |
| Prompt | `LOOSE_XML_ANSWER_USER_PROMPT` + problem + the adapter's output-file sentence. `check_instructions.py` verifies the prompts are byte-identical to every generated Harbor `instruction.md` |
| Container | One fresh container per problem from `environment/Dockerfile` (identical to the adapter's task Dockerfile). Agent runs use `environment/Dockerfile.codex`, which is that image plus Codex preinstalled with Harbor's own install steps. Harbor skips its install when the requested Codex version is present, and so does this runner, so both sides use the same image. Two differences from Harbor's runtime install do not affect the agent: apt uses `--no-install-recommends`, which skips e.g. `dbus`, whose post-install fails under rootless runtimes; and an optional `APT_MIRROR` is used only during the build, after which the original apt sources are restored |
| Agent | `codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --model <model> --json --enable unified_exec -c model_reasoning_effort=<effort> -- <prompt>`, with `$CODEX_HOME/config.toml` = `openai_base_url` and `auth.json` = `OPENAI_API_KEY`, as Harbor's Codex agent configures it |
| Timeout | 1800 s agent timeout (the adapter's `[agent].timeout_sec`) |
| Scoring | `/app/answer.txt` scored with ether0's reward functions (`test=True`); a missing answer or a reward-function exception scores 0 |

## Reproduce

```bash
# 1. Environment: this repository's locked dependencies
uv sync

# 2. Images (docker or podman)
docker build -t ether0-parity-env:latest harbor_parity/environment
docker build --build-arg BASE_IMAGE=ether0-parity-env:latest --build-arg CODEX_VERSION=0.166.0 \
  -f harbor_parity/environment/Dockerfile.codex -t ether0-parity-codex:0.166.0 harbor_parity/environment
# (optional) add --build-arg APT_MIRROR=https://<debian-mirror> if deb.debian.org is unreliable

# 3. Credentials (OpenAI-compatible endpoint used for the parity runs)
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...   # e.g. https://app.ppapi.ai/v1

# 4. One full run (repeat with run-2, run-3, ...)
cd harbor_parity
uv run python run_parity.py --run-name original-run-1 --runtime docker \
  --image ether0-parity-codex:0.166.0 --model gpt-5-mini --codex-version 0.166.0 \
  --reasoning-effort medium --n-concurrent 6 --results-dir results

# Optional: confirm prompts match a generated Harbor dataset
uv run python check_instructions.py /path/to/harbor/datasets/ether0-benchmark
```

Each run writes `results/<run-name>/<problem-id>/{instruction.md,codex.jsonl,answer.txt,result.json}`
and `results/<run-name>/summary.json`, which holds the accuracy over scored problems, the
infrastructure-failure count and the token totals. Infrastructure failures (no reward)
are rerun rather than counted as zeros.

### Harbor side (for comparison)

```bash
# In harbor-index/adapters, generate the tasks; --docker-image points at the same codex image
cd src/ether0-benchmark
uv run ether0-benchmark --output-dir ../../datasets/ether0-benchmark --docker-image ether0-parity-codex:0.166.0
cd ../..
harbor run -p datasets/ether0-benchmark -a codex --ak version=0.166.0 --ak reasoning_effort=medium \
  -m openai/gpt-5-mini -n 6
```
