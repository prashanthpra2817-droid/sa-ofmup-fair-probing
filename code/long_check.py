"""Setting (a) at T = 3000 and rho = 0.8 (resumable, one file per seed)."""
import os, sys, numpy as np
from sa_ofmup import bernoulli_env, make_alpha, offline_opt, run
T, RHO, M, A, I, P = 3000, 0.8, 12, 8, 4, 3
METHODS = ["Non-Probing", "OFMUP", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho, LCB)"]
os.makedirs("results/long", exist_ok=True)
for seed in range(3):
    path = f"results/long/rho0.8_T3000_seed{seed}.npz"
    if os.path.exists(path): continue
    env = bernoulli_env(M, A, np.random.default_rng(1000 + seed), RHO)
    opt = offline_opt(env, make_alpha(I, P), I, cap=M / A)
    data = {}
    for m in METHODS:
        r = run(env, m, T=T, I=I, seed=seed, opt=opt, alpha_p=P)
        for k in ["regret", "probes", "rho_hat", "rho_point"]: data[m + "|" + k] = r[k]
        print(f"seed={seed} {m:28s} regret={r['regret'][-1]:.1f} probes={r['probes'][M*A:].mean():.2f} planned={r['rho_hat'][-1]:.2f} point={r['rho_point'][-1]:.2f}", flush=True)
    np.savez(path, **data)
