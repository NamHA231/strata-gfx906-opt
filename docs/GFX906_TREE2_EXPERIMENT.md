# gfx906 two-branch decode experiment

This is a local experiment on one MI50 32 GB with ROCm 6.3.3. It does not change the default decode path or the server. The CLI switch is `STRATA_GFX906_TREE2=1` in a gfx906 build.

The MTP drafter records the runner-up for its first proposed token. If the target rejects the first proposal and chooses that runner-up, the CLI verifier runs a second window from the uncommitted root with the runner-up as its child. It commits the second window's accepted prefix. The two branches run **sequentially**. This tests whether a branch can be traversed with the current verifier; it does not share the target's dense-weight reads between branches.

## Measurement, 2026-11-09

- Model: Qwen3.8-Flash-Next abliterated IQ3_XXS, pack `sc117-iq3xxs`.
- Prompt: 32,768 public tokens from this repository's docs and code. Continuation: the same 256 token IDs in both arms. `--spec 3 --spec-min-p 0.10`.
- Baseline: 109 rounds, 147/217 MTP drafts accepted, 37.17 decode tokens/s.
- Two-branch experiment: seven alternate branches, 106 rounds, 150/204 drafts accepted, 36.30 and 35.72 decode tokens/s in two runs.
- The forced-continuation comparison counted 2/256 and 1/256 target tokens different from the baseline's argmax. In the second run, all seven alternate windows produced the expected next token. The remaining differences are not yet explained. The experiment is not parity validated.

The second verification pass costs more than the saved rounds on this prompt. The CLI also reported that `mlock` of about 20 GiB of host cache was refused, so these times do not represent the production service. Raw outputs are in `/tmp/strata-gfx906-ab-baseline.out`, `/tmp/strata-gfx906-tree2.out`, and their matching `.err` files. The service was restored to its original binary after each run.

## Next implementation step

A faster tree needs the target to evaluate both child branches in one window. The current verifier accepts a linear sequence and commits a prefix. A shared window needs a parent index for each row, branch-local GDN and PLE state, branch-local QSA keys and values, an ancestor mask in attention, and a commit path that keeps only the selected branch. Those changes must pass greedy parity checks before a speed claim.
