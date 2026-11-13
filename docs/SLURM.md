# Slurm

```bash
cp config.example.env config.env      # set account, partition, paths
./submit.sh single                    # one worker
./submit.sh array 20                  # twenty workers on the same corpus
```

Always submit through `submit.sh`: it passes the resources from `config.env` to
`sbatch` and tells the job where the repo is.

- Workers claim PDFs one at a time through marker files on shared storage, so
  any array size is correct.
- Resubmitting is safe: finished PDFs are skipped. A worker stops taking new
  PDFs before its walltime runs out.
- Compute nodes without internet: install as in [INSTALL.md](INSTALL.md)
  (offline machines). Jobs run with `HF_HUB_OFFLINE=1`.
