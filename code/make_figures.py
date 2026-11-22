"""Build the report figures from saved runs.
  staleness_a.png : rho sweep (setting a, T=1500) from results/chunks
  long_T3000.png  : regret curves at the paper's horizon from results/long
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

M, A, SEEDS, RHOS = 12, 8, 3, [0.0, 0.4, 0.8]
LABEL = {"Non-Probing": "Non-Probing", "OFMUP": "OFMUP", "OFMUP+fix": "OFMUP+fix",
         "SA-OFMUP (known rho)": r"SA-OFMUP (known $\rho$)",
         "SA-OFMUP (learned rho)": "SA-OFMUP (point estimate)",
         "SA-OFMUP (learned rho, LCB)": "SA-OFMUP (LCB, final)"}
# One colour per method in every figure; SA-OFMUP (LCB, final) is black.
STYLE = {"Random P+A": dict(c="tab:blue"), "Non-Probing": dict(c="tab:orange"),
         "Greedy P+A": dict(c="tab:green"), "OFMUP": dict(c="tab:red"),
         "OFMUP+fix": dict(c="tab:red", ls=":"),
         "SA-OFMUP (known rho)": dict(c="tab:purple", ls="--"),
         "SA-OFMUP (learned rho)": dict(c="tab:gray"),
         "SA-OFMUP (learned rho, LCB)": dict(c="black", lw=2.2)}


def chunks(rho):
    return [np.load(f"results/chunks/stale_a_rho{rho}_seed{s}.npz") for s in range(SEEDS)]


def staleness_figure():
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    methods = ["Non-Probing", "OFMUP", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho)",
               "SA-OFMUP (learned rho, LCB)"]           # OFMUP+fix is indistinguishable from OFMUP
    for m in methods:
        reg = np.array([[c[m + "|regret"][-1] for c in chunks(r)] for r in RHOS])
        prb = [np.mean([c[m + "|probes"][M * A:].mean() for c in chunks(r)]) for r in RHOS]
        wst = [np.mean([float(c[m + "|worst"]) for c in chunks(r)]) for r in RHOS]
        ax[0].errorbar(RHOS, reg.mean(1), yerr=reg.std(1), marker="o", capsize=3, label=LABEL[m], **STYLE[m])
        ax[1].plot(RHOS, prb, marker="o", **STYLE[m])
        ax[2].plot(RHOS, wst, marker="o", **STYLE[m])
    lcb = "SA-OFMUP (learned rho, LCB)"
    for r, ls in zip(RHOS[1:], ["--", "-"]):
        tr = np.mean([ch[lcb + "|rho_point"] for ch in chunks(r)], axis=0)
        ax[3].plot(tr, c="black", ls=ls, label=rf"point estimate, true $\rho$={r}")
        ax[3].axhline(r, ls=":", c="gray")
    ax[0].set_title("Final cumulative regret"); ax[0].set_xlabel(r"staleness $\rho$")
    ax[1].set_title("Arms probed per round"); ax[1].set_xlabel(r"staleness $\rho$")
    ax[2].set_title("Worst-off agent: average reward"); ax[2].set_xlabel(r"staleness $\rho$")
    ax[2].set_ylim(0, 0.7)
    ax[3].set_title(r"SA-OFMUP (LCB): estimate of $\rho$"); ax[3].set_xlabel("Time steps"); ax[3].set_ylim(0, 1)
    for a in ax:
        a.grid(alpha=0.3)
    ax[3].legend(loc="lower right", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=10, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig("results/staleness_a.png", dpi=150)


def long_figure():
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    for m in ["Random P+A", "Greedy P+A", "Non-Probing", "OFMUP"]:
        R = np.array([np.load(f"results/long/repro_a_T3000_seed{s}.npz")[m + "|regret"] for s in range(SEEDS)])
        ax[0].plot(R.mean(0), label=m, **STYLE[m])
        ax[0].fill_between(np.arange(R.shape[1]), R.mean(0) - R.std(0), R.mean(0) + R.std(0),
                           alpha=0.2, color=STYLE[m]["c"])
    load = lambda m: np.array([np.load(f"results/long/rho0.8_T3000_seed{s}.npz")[m + "|regret"] for s in range(SEEDS)])
    base = load("OFMUP")
    for m in ["Non-Probing", "OFMUP", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho, LCB)"]:
        R = load(m)
        ax[1].plot(R.mean(0), label=LABEL[m], **STYLE[m])
        ax[1].fill_between(np.arange(R.shape[1]), R.mean(0) - R.std(0), R.mean(0) + R.std(0),
                           alpha=0.15, color=STYLE[m]["c"])
        if m != "OFMUP":
            ax[2].plot((base - R).mean(0), label=LABEL[m], **STYLE[m])     # paired by seed
    ax[2].axhline(0, c="tab:red", lw=1)
    ax[0].set_title(r"Setting (a), $\rho=0$: reproduction at $T=3000$")
    ax[1].set_title(r"Setting (a), $\rho=0.8$: staleness at $T=3000$")
    ax[2].set_title(r"$\rho=0.8$: regret saved relative to OFMUP")
    for a in ax:
        a.set_xlabel("Time steps"); a.grid(alpha=0.3); a.legend(fontsize=9)
    ax[0].set_ylabel("Cumulative regret"); ax[1].set_ylabel("Cumulative regret")
    ax[2].set_ylabel("OFMUP regret minus method regret")
    fig.tight_layout()
    fig.savefig("results/long_T3000.png", dpi=150)


if __name__ == "__main__":
    staleness_figure()
    long_figure()
    print("figures written")
