#!/usr/bin/env python3
"""
edge_benchmark.py -- latency / memory on embedded Linux (UNO Q, Pi, …).
Needs Python 3 + NumPy. Optionally loads trained_weights.npz from MATLAB export.

Usage:
  OMP_NUM_THREADS=1 python3 edge_benchmark.py --n 1000
  OMP_NUM_THREADS=1 python3 edge_benchmark.py --n 1000 --weights trained_weights.npz
"""
import argparse, json, math, os, platform, resource, sys, time
from pathlib import Path
import numpy as np

W = 30
rng = np.random.default_rng(0)

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class LSTM:
    def __init__(self, nin, H, Wx=None, Wh=None, b=None):
        s = 1.0 / math.sqrt(H)
        self.Wx = Wx if Wx is not None else rng.uniform(-s, s, (4 * H, nin)).astype(np.float32)
        self.Wh = Wh if Wh is not None else rng.uniform(-s, s, (4 * H, H)).astype(np.float32)
        self.b = b if b is not None else np.zeros(4 * H, np.float32)
        self.H = H
    def run(self, X, reverse=False):
        H = self.H; h = np.zeros(H, np.float32); c = np.zeros(H, np.float32)
        Xp = X @ self.Wx.T + self.b
        idx = range(X.shape[0] - 1, -1, -1) if reverse else range(X.shape[0])
        for t in idx:
            z = Xp[t] + self.Wh @ h
            i = sigmoid(z[:H]); f = sigmoid(z[H:2*H]); g = np.tanh(z[2*H:3*H]); o = sigmoid(z[3*H:])
            c = f * c + i * g; h = o * np.tanh(c)
        return h
    def params(self): return self.Wx.size + self.Wh.size + self.b.size

class Dense:
    def __init__(self, nin, nout, W=None, b=None):
        self.W = W if W is not None else rng.normal(0, 1/math.sqrt(nin), (nout, nin)).astype(np.float32)
        self.b = b if b is not None else np.zeros(nout, np.float32)
        if self.W.ndim == 1: self.W = self.W.reshape(nout, nin)
        self.b = np.atleast_1d(self.b).astype(np.float32)
    def __call__(self, x): return self.W @ x + self.b
    def params(self): return self.W.size + self.b.size

class BiLSTM:
    def __init__(self, weights=None):
        if weights is None:
            self.f = LSTM(8, 128); self.r = LSTM(8, 128)
            self.d1 = Dense(256, 64); self.d2 = Dense(64, 32); self.d3 = Dense(32, 1)
        else:
            self.f = LSTM(8, 128, weights['fw_Wx'], weights['fw_Wh'], weights['fw_b'])
            self.r = LSTM(8, 128, weights['bw_Wx'], weights['bw_Wh'], weights['bw_b'])
            self.d1 = Dense(256, 64, weights['W1'], weights['b1'])
            self.d2 = Dense(64, 32, weights['W2'], weights['b2'])
            w3 = np.atleast_2d(weights['W3']); 
            if w3.shape == (32,): w3 = w3.reshape(1, 32)
            if w3.shape == (32, 1): w3 = w3.T
            b3 = np.atleast_1d(weights['b3']).reshape(-1)
            self.d3 = Dense(32, 1, w3.astype(np.float32), b3.astype(np.float32))
    def __call__(self, X):
        h = np.concatenate([self.f.run(X), self.r.run(X, True)])
        h = np.maximum(self.d1(h), 0); h = np.maximum(self.d2(h), 0); return float(self.d3(h)[0])
    def params(self): return self.f.params() + self.r.params() + self.d1.params() + self.d2.params() + self.d3.params()

class PILSTM:
    def __init__(self, weights=None):
        if weights is None:
            self.l = LSTM(11, 128); self.d1 = Dense(128, 32); self.d2 = Dense(32, 1)
        else:
            self.l = LSTM(11, 128, weights['pi_Wx'], weights['pi_Wh'], weights['pi_b'])
            self.d1 = Dense(128, 32, weights['pi_W1'], weights['pi_b1'])
            w2 = np.atleast_2d(weights['pi_W2'])
            if w2.shape == (32,): w2 = w2.reshape(1, 32)
            if w2.shape == (32, 1): w2 = w2.T
            b2 = np.atleast_1d(weights['pi_b2']).reshape(-1)
            self.d2 = Dense(32, 1, w2.astype(np.float32), b2.astype(np.float32))
    def __call__(self, X):
        h = self.l.run(X); h = np.maximum(self.d1(h), 0); return float(self.d2(h)[0])
    def params(self): return self.l.params() + self.d1.params() + self.d2.params()

OCVg = np.linspace(0, 1, 101); OCVv = 3.0 + 1.2 * OCVg - 0.25 * np.exp(-12 * OCVg) + 0.1 * np.exp(-12 * (1 - OCVg))
def ocv(s): return float(np.interp(s, OCVg, OCVv))
def docv(s):
    e = 1e-3; return (ocv(min(s + e, 1)) - ocv(max(s - e, 0))) / (min(s + e, 1) - max(s - e, 0))

class Model:
    R0 = 0.020; R1 = 0.018; tau = 25.0; Q = 2.7 * 3600; dt = 1.0
    a = math.exp(-1.0 / 25.0); Qn = np.diag([1e-9, 1e-7]); Rn = 3e-4

def pred(m, x, I):
    return [x[0] - m.dt * I / m.Q, m.a * x[1] + m.R1 * (1 - m.a) * I]
def meas(m, x, I): return ocv(x[0]) - x[1] - m.R0 * I

def ekf_step(m, x, P, I, V):
    xp = pred(m, x, I); A = np.array([[1, 0], [0, m.a]]); Pp = A @ P @ A.T + m.Qn
    H = np.array([docv(xp[0]), -1.0]); S = float(H @ Pp @ H) + m.Rn; K = (Pp @ H) / S
    e = V - meas(m, xp, I); xn = [xp[0] + K[0] * e, xp[1] + K[1] * e]
    I2 = np.eye(2) - np.outer(K, H); return xn, I2 @ Pp @ I2.T + m.Rn * np.outer(K, K)

def iekf_step(m, x, P, I, V):
    xp = pred(m, x, I); A = np.array([[1, 0], [0, m.a]]); Pp = A @ P @ A.T + m.Qn; xi = list(xp)
    for _ in range(5):
        H = np.array([docv(xi[0]), -1.0]); S = float(H @ Pp @ H) + m.Rn; K = (Pp @ H) / S
        e = V - meas(m, xi, I) - float(H @ (np.array(xp) - np.array(xi)))
        xn = [xp[0] + K[0] * e, xp[1] + K[1] * e]
        if abs(xn[0] - xi[0]) < 1e-6: xi = xn; break
        xi = xn
    I2 = np.eye(2) - np.outer(K, H); return xi, I2 @ Pp @ I2.T + m.Rn * np.outer(K, K)

def shekf_step(m, x, P, I, V, st):
    xp = pred(m, x, I); A = np.array([[1, 0], [0, m.a]]); Pp = A @ P @ A.T + m.Qn
    H = np.array([docv(xp[0]), -1.0]); e = V - meas(m, xp, I)
    k = st['k'] + 1; st['k'] = k; d = (1 - 0.98) / (1 - 0.98 ** k)
    Rk = (1 - d) * st['R'] + d * (e * e - float(H @ Pp @ H)); st['R'] = min(max(Rk, 0.2 * m.Rn), 20 * m.Rn)
    S = float(H @ Pp @ H) + st['R']; K = (Pp @ H) / S
    xn = [xp[0] + K[0] * e, xp[1] + K[1] * e]; I2 = np.eye(2) - np.outer(K, H)
    return xn, I2 @ Pp @ I2.T + st['R'] * np.outer(K, K)

def spkf_step(m, x, P, I, V):
    # v3.3 defaults: alpha=1, kappa=1
    n = 2; al, be, ka = 1.0, 2.0, 1.0; lam = al * al * (n + ka) - n
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lam))); Wc = Wm.copy(); Wm[0] = lam / (n + lam); Wc[0] = Wm[0] + (1 - al * al + be)
    L = np.linalg.cholesky((n + lam) * P + 1e-12 * np.eye(2)); X = [np.array(x)] + [np.array(x) + L[:, i] for i in range(n)] + [np.array(x) - L[:, i] for i in range(n)]
    Xp = [np.array(pred(m, xi, I)) for xi in X]; xm = sum(w * xi for w, xi in zip(Wm, Xp))
    Pp = m.Qn + sum(w * np.outer(xi - xm, xi - xm) for w, xi in zip(Wc, Xp))
    Y = np.array([meas(m, xi, I) for xi in Xp]); ym = float(Wm @ Y)
    Pyy = m.Rn + float(sum(w * (y - ym) ** 2 for w, y in zip(Wc, Y)))
    Pxy = sum(w * (xi - xm) * (y - ym) for w, xi, y in zip(Wc, Xp, Y)); K = Pxy / Pyy
    xn = xm + K * (V - ym); return list(xn), Pp - np.outer(K, K) * Pyy

def timeit(fn, n):
    for _ in range(20): fn()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns(); fn(); t[i] = (time.perf_counter_ns() - t0) / 1e6
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), p99_ms=float(np.percentile(t, 99)), max_ms=float(t.max()))

def cpu_name():
    try:
        for l in open('/proc/cpuinfo'):
            if l.lower().startswith(('model name', 'hardware', 'processor\t: 0')) and ':' in l: return l.split(':', 1)[1].strip()
    except Exception: pass
    return platform.processor() or platform.machine()

def load_weights(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float32) for k in z.files}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--weights', type=str, default='', help='path to trained_weights.npz (empty = random)')
    a = ap.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    wpath = a.weights
    if not wpath:
        cand = Path(__file__).resolve().parent / 'trained_weights.npz'
        if cand.is_file(): wpath = str(cand)
    weights = None; weight_src = 'random'
    if wpath and Path(wpath).is_file():
        weights = load_weights(wpath); weight_src = str(Path(wpath).resolve())
        print(f'Loaded trained weights from {weight_src}')
    else:
        print('Using random weights (same layer sizes as trained nets)')

    m = Model(); x0 = [0.8, 0.0]; P0 = np.diag([1e-2, 1e-3]); I, V = 1.5, 3.9
    Xb = rng.normal(size=(W, 8)).astype(np.float32); Xp = rng.normal(size=(W, 11)).astype(np.float32)
    bi, pi = BiLSTM(weights), PILSTM(weights)
    st = {'k': 0, 'R': m.Rn}
    # pruned stack weights: prefer SHEKF / PI / Bi from NNLS if present (7-vector)
    if weights is not None and 'w' in weights and weights['w'].size >= 7:
        ww = weights['w'].reshape(-1)
        # indices: EKF SPKF IEKF SHEKF LSBoost PI Bi  -> use SHEKF, PI, Bi
        wstack = np.array([ww[3], ww[5], ww[6]], dtype=np.float64)
        if wstack.sum() > 0: wstack = wstack / wstack.sum()
        else: wstack = np.array([0.18, 0.17, 0.65])
    else:
        wstack = np.array([0.18, 0.17, 0.65])
    res = {}
    res['EKF'] = timeit(lambda: ekf_step(m, x0, P0, I, V), a.n)
    res['SPKF'] = timeit(lambda: spkf_step(m, x0, P0, I, V), a.n)
    res['IEKF'] = timeit(lambda: iekf_step(m, x0, P0, I, V), a.n)
    res['SHEKF'] = timeit(lambda: shekf_step(m, x0, P0, I, V, st), a.n)
    res['BiLSTM'] = timeit(lambda: bi(Xb), a.n)
    res['PI-LSTM'] = timeit(lambda: pi(Xp), a.n)
    def stack():
        s = shekf_step(m, x0, P0, I, V, st)[0][0]; b = bi(Xb); p = pi(Xp)
        return float(wstack @ np.array([s, p, b]))
    res['Pruned stack'] = timeit(stack, max(a.n // 2, 100))
    info = dict(cpu=cpu_name(), machine=platform.machine(), python=sys.version.split()[0], numpy=np.__version__,
                threads=os.environ.get('OMP_NUM_THREADS'), window=W, n=a.n, weights=weight_src,
                peak_rss_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                params=dict(BiLSTM=bi.params(), PILSTM=pi.params(), weights_MB_fp32=(bi.params() + pi.params()) * 4 / 1e6))
    out = dict(info=info, results=res); json.dump(out, open('edge_results.json', 'w'), indent=1)
    print(f"\nDevice: {info['cpu']} ({info['machine']}), Python {info['python']}, NumPy {info['numpy']}, 1 thread")
    print(f"Weights: {weight_src}")
    print(f"{'Estimator':<14}{'median ms':>11}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>9}")
    for k, v in res.items(): print(f"{k:<14}{v['median_ms']:>11.3f}{v['p95_ms']:>9.3f}{v['p99_ms']:>9.3f}{v['max_ms']:>9.3f}")
    print(f"\nPeak RSS {info['peak_rss_MB']:.1f} MB | parameters BiLSTM {bi.params():,}, PI-LSTM {pi.params():,} | fp32 weights {info['params']['weights_MB_fp32']:.2f} MB")
    print("Wrote edge_results.json")

if __name__ == '__main__': main()
