"""
run_experiments.py
==================
  python run_experiments.py repro  --T 3000 --seeds 5      # Figure 1 style (rho = 0)
  python run_experiments.py stale  --T 3000 --seeds 5      # novelty: sweep over rho
  python run_experiments.py repro  --quick                 # fast smoke test

Results (plots + CSV) go to ./results/.
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sa_ofmup import (bernoulli_env, discrete_env, taxi_env, make_alpha,
                      offline_opt, run)

PAPER_METHODS = ["Random P+A", "Non-Probing", "Greedy P+A", "OFMUP"]
COLORS = {"Random P+A": "tab:blue", "Non-Probing": "tab:orange", "Greedy P+A": "tab:green",
          "OFMUP": "tab:red", "OFMUP+fix": "tab:pink", "SA-OFMUP (known rho)": "tab:purple",
          "SA-OFMUP (learned rho)": "tab:gray", "SA-OFMUP (learned rho, LCB)": "black"}
STALE_METHODS = ["Non-Probing", "OFMUP", "OFMUP+fix", "SA-OFMUP (learned rho, LCB)",
                 "SA-OFMUP (known rho)", "SA-OFMUP (learned rho)"]
SETTINGS = {                                   # paper's Figure 1 panels
    "a": ("bernoulli", 12, 8), "b": ("bernoulli", 20, 10),
    "c": ("discrete", 12, 8),  "d": ("discrete", 20, 10),
}


def make_env(kind, M, A, seed, rho, taxi_csv=None):
    rng = np.random.default_rng(1000 + seed)     # a new problem instance per seed
    if kind == "bernoulli":
        return bernoulli_env(M, A, rng, rho)
    if kind == "discrete":
        return discrete_env(M, A, rng, rho)
    return taxi_env(taxi_csv, M, A, rng, rho)


def run_block(kind, M, A, rho, methods, args):
    """All methods x seeds on one setting. Same instance and benchmark per seed."""
    out = {m: [] for m in methods}
    for seed in range(args.seeds):
        env = make_env(kind, M, A, seed, rho, args.taxi_csv)
        opt = offline_opt(env, make_alpha(args.I, args.alpha_p), args.I,
                          cap=M / A if args.model == "matrix" else None)
        for m in methods:
            t0 = time.time()
            res = run(env, m, T=args.T, I=args.I, seed=seed, opt=opt,
                      model=args.model, alpha_p=args.alpha_p)
            out[m].append(res)
            print(f"  {kind} M={M} A={A} rho={rho:.1f} seed={seed} {m:24s} "
                  f"regret={res['regret'][-1]:7.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def repro(args):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    rows = []
    for ax, (panel, (kind, M, A)) in zip(axes, SETTINGS.items()):
        if args.taxi_csv:
            kind = "taxi"
        out = run_block(kind, M, A, 0.0, PAPER_METHODS, args)
        final = {}
        for m in PAPER_METHODS:
            R = np.array([r["regret"] for r in out[m]])
            mean, std = R.mean(0), R.std(0)
            ax.plot(mean, label=m, color=COLORS[m])
            ax.fill_between(np.arange(args.T), mean - std, mean + std, alpha=0.2, color=COLORS[m])
            final[m] = mean[-1]
            rows.append([panel, kind, M, A, m, mean[-1], std[-1]])
        ax.set_title(f"({panel}) {kind}, M={M}, A={A}")
        ax.set_xlabel("Time steps")
        ax.set_ylabel("Cumulative regret")
        ax.grid(alpha=0.3)
        o = final["OFMUP"]
        print(f"({panel}) OFMUP vs Random P+A: {100 * (1 - o / final['Random P+A']):.0f}% lower | "
              f"vs Greedy P+A: {100 * (1 - o / final['Greedy P+A']):.0f}% | "
              f"vs Non-Probing: {100 * (1 - o / final['Non-Probing']):.0f}%")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig1_reproduction.png"), dpi=150)
    save_csv(os.path.join(args.out, "fig1_reproduction.csv"),
             ["panel", "env", "M", "A", "method", "final_regret_mean", "final_regret_std"], rows)


def stale(args):
    kind, M, A = SETTINGS[args.setting]
    if args.taxi_csv:
        kind = "taxi"
    rhos = [float(x) for x in args.rhos.split(",")]
    summary = {m: {"regret": [], "regret_sd": [], "probes": [], "worst": [], "rho_hat": []}
               for m in STALE_METHODS}
    traces = {}
    rows = []
    for rho in rhos:
        out = run_block(kind, M, A, rho, STALE_METHODS, args)
        for m in STALE_METHODS:
            reg = [r["regret"][-1] for r in out[m]]
            prb = [r["probes"][M * A:].mean() for r in out[m]]
            wst = [r["worst_avg"] for r in out[m]]
            rh = [r["rho_hat"][-1] for r in out[m]]
            for key, val in zip(["regret", "probes", "worst", "rho_hat"], [reg, prb, wst, rh]):
                summary[m][key].append(np.mean(val))
            summary[m]["regret_sd"].append(np.std(reg))
            rows.append([rho, m, np.mean(reg), np.std(reg), np.mean(prb), np.mean(wst), np.mean(rh)])
        traces[rho] = np.mean([r["rho_hat"] for r in out["SA-OFMUP (learned rho)"]], axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for m in STALE_METHODS:
        axes[0].errorbar(rhos, summary[m]["regret"], yerr=summary[m]["regret_sd"],
                         marker="o", capsize=3, label=m, color=COLORS[m])
        axes[1].plot(rhos, summary[m]["probes"], marker="o", label=m, color=COLORS[m])
        axes[2].plot(rhos, summary[m]["worst"], marker="o", label=m, color=COLORS[m])
    for rho, tr in traces.items():
        axes[3].plot(tr, label=f"true rho = {rho}")
        axes[3].axhline(rho, ls=":", c="gray")
    titles = ["Final cumulative regret", "Zones probed per round",
              "Worst-off driver: average reward", "Learned rho over time"]
    xlabels = ["staleness rho"] * 3 + ["Time steps"]
    for ax, ti, xl in zip(axes, titles, xlabels):
        ax.set_title(ti)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[3].legend(fontsize=8)
    fig.suptitle(f"Staleness sweep: {kind}, M={M}, A={A}, T={args.T}, seeds={args.seeds}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, f"staleness_{args.setting}.png"), dpi=150)
    save_csv(os.path.join(args.out, f"staleness_{args.setting}.csv"),
             ["rho", "method", "final_regret_mean", "final_regret_std",
              "probes_per_round", "worst_driver_avg_reward", "final_rho_hat"], rows)


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("saved", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp", choices=["repro", "stale"])
    p.add_argument("--T", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--I", type=int, default=4, help="probing budget")
    p.add_argument("--alpha-p", type=float, default=3, help="alpha(k) = (k/I)^p")
    p.add_argument("--model", choices=["matrix", "shared"], default="matrix")
    p.add_argument("--setting", choices=list(SETTINGS), default="a")
    p.add_argument("--rhos", default="0,0.2,0.4,0.6,0.8")
    p.add_argument("--taxi-csv", default=None, help="NYC Yellow Taxi 2016 CSV")
    p.add_argument("--out", default="results")
    p.add_argument("--quick", action="store_true", help="T=300, 2 seeds, 3 rho values")
    args = p.parse_args()
    if args.quick:
        args.T, args.seeds, args.rhos = 300, 2, "0,0.4,0.8"
    os.makedirs(args.out, exist_ok=True)
    {"repro": repro, "stale": stale}[args.exp](args)
