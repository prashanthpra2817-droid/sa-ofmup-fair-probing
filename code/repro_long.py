"""Figure 1 setting (a) at the paper's horizon T = 3000 (resumable, one file per seed)."""
import os, numpy as np
from sa_ofmup import bernoulli_env, make_alpha, offline_opt, run
T, M, A, I, P = 3000, 12, 8, 4, 3
METHODS = ["Random P+A", "Non-Probing", "Greedy P+A", "OFMUP"]
os.makedirs("results/long", exist_ok=True)
for seed in range(3):
    path = f"results/long/repro_a_T3000_seed{seed}.npz"
    if os.path.exists(path): continue
    env = bernoulli_env(M, A, np.random.default_rng(1000 + seed), 0.0)
    opt = offline_opt(env, make_alpha(I, P), I, cap=M / A)
    data = {}
    for m in METHODS:
        r = run(env, m, T=T, I=I, seed=seed, opt=opt, alpha_p=P)
        data[m + "|regret"] = r["regret"]
        print(f"seed={seed} {m:14s} regret={r['regret'][-1]:.1f}", flush=True)
    np.savez(path, **data)
