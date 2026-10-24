#!/bin/bash
# T3: trained-weight retime on UNO Q. Run from Mac Terminal (password prompts).
set -euo pipefail
HOST="${1:-arduino@192.168.1.26}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$ROOT/edge_results_trained_board.json"
echo "Target: $HOST"
ssh "$HOST" 'mkdir -p ~/edge_trained'
scp "$ROOT/INSTALL_TRAINED_ON_BOARD.sh" "$ROOT/edge_benchmark.py" "$ROOT/trained_weights.npz" "$HOST:~/edge_trained/"
ssh -t "$HOST" 'cd ~/edge_trained && bash INSTALL_TRAINED_ON_BOARD.sh'
# Prefer explicit trained result name if script writes it; else edge_results.json
ssh "$HOST" 'ls -la ~/edge_trained/edge_results*.json; python3 - <<"PY"
import json,glob,os,platform
paths=sorted(glob.glob(os.path.expanduser("~/edge_trained/edge_results*.json")))
assert paths, "no json"
p=paths[-1]
d=json.load(open(p))
info=d.get("info") or d
machine=info.get("machine") or platform.machine()
rss=info.get("peak_rss_MB") or d.get("peak_rss_MB")
print("CHECK machine=", machine, "rss_MB=", rss, "file=", p)
assert "aarch64" in str(machine).lower() or "arm" in str(machine).lower() and "arm64" not in str(platform.platform()).lower() or machine=="aarch64"
# Accept aarch64; reject Mac-like huge RSS
if rss is not None and float(rss) > 500:
    raise SystemExit(f"REJECT: peak_rss_MB={rss} looks like a laptop, not the board")
print("BOARD_OK", p)
PY
'
scp "$HOST:~/edge_trained/edge_results.json" "$OUT" 2>/dev/null || scp "$HOST:~/edge_trained/edge_results_trained.json" "$OUT"
python3 - << PY
import json
d=json.load(open("$OUT"))
print("Saved", "$OUT")
print(json.dumps({k:d[k] for k in list(d)[:12]}, indent=2)[:1500])
PY
