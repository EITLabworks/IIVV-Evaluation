# Server execution

Follow [todo.md](todo.md) for the complete sequence, tmux commands, transfers,
resume rules, notebooks and manuscript checks. The scientific design, class
composition, metrics and candidate budget are in [PROTOCOL.md](PROTOCOL.md).

```bash
source ./setup_environment.sh
CPUSET=224-255 ./run_all.sh --data-quality-only --output results/revision-quality-v3
CPUSET=224-255 ./run_all.sh --quality-results results/revision-quality-v3 --output results/revision-three-tasks-v3
```

Inspect quality before starting the second command. Replace CPUSET with your
allocated logical CPUs, for example `0-23`. A completed run ends with
`REVISION RESULTS READY`. Follow `logs/revision-latest.log` during execution.

No previous task checkpoints or selected configurations are reused. Use new
output paths and preserve previous scientific results as provenance.

Worker processes start with `spawn` to avoid inheriting native thread-pool locks
after serial numerical work. Each pool shares one temporary, read-only memory-mapped
dataset. This adds startup I/O and requires temporary space for the uncompressed
input arrays; `TMPDIR` can point to a fast scratch directory. Checkpoints, seeds and
training/test separation are unchanged. Regression logs show each test name and
automatically dump Python stacks after a test exceeds five minutes.
