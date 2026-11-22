"""Resumable version of the staleness sweep (setting a). Each (rho, seed) is saved
to results/chunks/ as soon as it finishes; rerunning skips finished chunks.
    python stale_chunks.py run        # run missing chunks
    python stale_chunks.py aggregate  # build staleness_a.csv / .png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sa_ofmup import bernoulli_env, make_alpha, offline_opt, run

T, SEEDS, RHOS, I, ALPHA_P = 1500, 3, [0.0, 0.4, 0.8], 4, 3
M, A = 12, 8
METHODS = ["Non-Probing", "OFMUP", "OFMUP+fix", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho)", "SA-OFMUP (learned rho, LCB)"]
OUT = "results"
os.makedirs(os.path.join(OUT, "chunks"), exist_ok=True)


def chunk_path(rho, seed):
    return os.path.join(OUT, "chunks", f"stale_a_rho{rho}_seed{seed}.npz")


def run_chunks():
    for rho in RHOS:
        for seed in range(SEEDS):
            data = dict(np.load(chunk_path(rho, seed))) if os.path.exists(chunk_path(rho, seed)) else {}
            todo = [m for m in METHODS if m + "|regret" not in data]
            if not todo:
                continue
            env = bernoulli_env(M, A, np.random.default_rng(1000 + seed), rho)
            opt = offline_opt(env, make_alpha(I, ALPHA_P), I, cap=M / A)
            for m in todo:
                r = run(env, m, T=T, I=I, seed=seed, opt=opt, alpha_p=ALPHA_P)
                data[m + "|regret"] = r["regret"]
                data[m + "|probes"] = r["probes"]
                data[m + "|rho_hat"] = r["rho_hat"]
                data[m + "|rho_point"] = r["rho_point"]
                data[m + "|worst"] = np.array(r["worst_avg"])
                print(f"rho={rho} seed={seed} {m:24s} regret={r['regret'][-1]:.1f}", flush=True)
            np.savez(chunk_path(rho, seed), **data)


def aggregate():
    rows, summary, traces = [], {m: {k: [] for k in ["regret", "regret_sd", "probes", "worst", "rho_hat"]} for m in METHODS}, {}
    for rho in RHOS:
        chunks = [np.load(chunk_path(rho, s)) for s in range(SEEDS)]
        for m in METHODS:
            reg = [c[m + "|regret"][-1] for c in chunks]
            prb = [c[m + "|probes"][M * A:].mean() for c in chunks]
            wst = [float(c[m + "|worst"]) for c in chunks]
            rh = [c[m + ("|rho_point" if "LCB" in m else "|rho_hat")][-1] for c in chunks]
            for key, val in zip(["regret", "probes", "worst", "rho_hat"], [reg, prb, wst, rh]):
                summary[m][key].append(np.mean(val))
            summary[m]["regret_sd"].append(np.std(reg))
            rows.append([rho, m, np.mean(reg), np.std(reg), np.mean(prb), np.mean(wst), np.mean(rh)])
        traces[rho] = np.mean([c["SA-OFMUP (learned rho, LCB)|rho_point"] for c in chunks], axis=0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for m in METHODS:
        axes[0].errorbar(RHOS, summary[m]["regret"], yerr=summary[m]["regret_sd"], marker="o", capsize=3, label=m)
        axes[1].plot(RHOS, summary[m]["probes"], marker="o", label=m)
        axes[2].plot(RHOS, summary[m]["worst"], marker="o", label=m)
    for rho, tr in traces.items():
        axes[3].plot(tr, label=f"true rho = {rho}")
        axes[3].axhline(rho, ls=":", c="gray")
    titles = ["Final cumulative regret", "Zones probed per round", "Worst-off driver: average reward", "Point estimate of rho over time (LCB method)"]
    for ax, ti, xl in zip(axes, titles, ["staleness rho"] * 3 + ["Time steps"]):
        ax.set_title(ti); ax.set_xlabel(xl); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8); axes[3].legend(fontsize=8)
    fig.suptitle(f"Staleness sweep: bernoulli, M={M}, A={A}, T={T}, seeds={SEEDS}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "staleness_a.png"), dpi=150)
    with open(os.path.join(OUT, "staleness_a.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "method", "final_regret_mean", "final_regret_std", "probes_per_round",
                    "worst_driver_avg_reward", "final_rho_hat"])
        w.writerows(rows)
    print("aggregated")


if __name__ == "__main__":
    run_chunks() if sys.argv[1] == "run" else aggregate()
