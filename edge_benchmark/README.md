# Edge benchmark (Arduino UNO Q)

```bash
OMP_NUM_THREADS=1 python3 edge_benchmark.py --n 500
OMP_NUM_THREADS=1 python3 edge_benchmark.py --n 500 --weights trained_weights.npz
```

Valid board results: `machine` / uname = aarch64, peak RSS tens of MB.
See `edge_runs_summary.txt` and `edge_results_trained_board.json`.
