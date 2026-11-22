QUICK      = True

RUN_REPRO  = True
RUN_STALE  = True
RUN_LONG   = True

T_MAIN     = 1500
T_LONG     = 3000
SEEDS      = 3
RHOS       = [0.0, 0.4, 0.8]
ALPHA_P    = 3


import csv
import os
import time
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np


GRID = np.round(np.arange(21) * 0.05, 2)
G = len(GRID)


def to_grid_idx(values):
    return np.clip(np.rint(np.asarray(values, dtype=float) / 0.05).astype(int), 0, G - 1)


def gm(util):
    """Geometric-mean NSW over the last axis (agents). Zero if any agent gets 0."""
    with np.errstate(divide="ignore"):
        return np.exp(np.mean(np.log(util), axis=-1))


def solve_nsw(r, iters=60, cap=None):
    """Maximise geometric-mean NSW. Always returns a matrix policy pi (..., M, n),
    where pi[j, a] = probability that agent j is sent to arm a.

    cap=None  -> shared model: one distribution over arms for every agent.
    cap=c     -> matrix model: each agent has its own row, and each arm receives
                 at most c agents in expectation (c = M / A in the experiments).

    Multiplicative (EM-style) updates; the matrix model adds capacity scaling."""
    r = np.maximum(r, 1e-9)
    M, n = r.shape[-2:]
    if cap is None:
        p = np.full(r.shape[:-2] + (n,), 1.0 / n)
        for _ in range(iters):
            u = np.einsum("...mn,...n->...m", r, p)
            p = p * np.mean(r / u[..., None], axis=-2)
            p /= p.sum(axis=-1, keepdims=True)
        return np.broadcast_to(p[..., None, :], r.shape).copy()
    pi = np.full(r.shape, 1.0 / n)
    for _ in range(iters):
        u = (pi * r).sum(axis=-1)
        pi = pi * r / u[..., None]
        for _ in range(4):
            col = pi.sum(axis=-2)
            pi = pi * np.minimum(1.0, cap / np.maximum(col, 1e-12))[..., None, :]
            pi /= pi.sum(axis=-1, keepdims=True)
    return pi


def nsw_value(r, pi):
    """Geometric-mean NSW of matrix policy pi for values r (batched)."""
    return gm((pi * r).sum(axis=-1))


def best_nsw(r, iters=60, cap=None):
    return nsw_value(r, solve_nsw(r, iters, cap))


def sample_grid(P, rng, K=None):
    """Draw rewards from per-pair distributions P (..., G) on GRID.
    Returns shape (...) or (K, ...)."""
    cdf = np.cumsum(P, axis=-1)
    shape = P.shape[:-1] if K is None else (K,) + P.shape[:-1]
    u = rng.random(shape)[..., None]
    idx = np.minimum((u > cdf).sum(axis=-1), G - 1)
    return GRID[idx]


def make_alpha(I, p=4):
    """Probing overhead alpha(k) = (k / I)^p: non-decreasing, alpha(0) = 0, alpha(I) = 1."""
    return lambda k: (k / I) ** p


class Env:
    """M agents, A arms, true distributions P (M, A, G).
    rho = probability that a probed value is stale when the agent arrives (rho = 0: OFMUP setting)."""

    def __init__(self, P, rho=0.0):
        self.P = np.asarray(P, dtype=float)
        self.rho = float(rho)
        self.M, self.A, _ = self.P.shape
        self.mu = self.P @ GRID

    def probe(self, S, rng):
        """Probe arms S: one draw for every agent on every probed arm -> (M, |S|)."""
        return sample_grid(self.P[:, S], rng)

    def pull(self, arms, S, X, rng):
        """Each agent j goes to arm arms[j]. A probed value stays valid with
        probability 1 - rho (one staleness event per arm per round);
        otherwise the agent gets a fresh draw."""
        stale = rng.random(self.A) < self.rho
        Y = sample_grid(self.P[np.arange(self.M), arms], rng)
        pos = {a: i for i, a in enumerate(S)}
        for j, a in enumerate(arms):
            if a in pos and not stale[a]:
                Y[j] = X[j, pos[a]]
        return Y


def bernoulli_env(M, A, rng, rho=0.0):
    """Bernoulli rewards with means in [0.3, 0.8]."""
    mu = rng.uniform(0.3, 0.8, (M, A))
    P = np.zeros((M, A, G))
    P[..., 0], P[..., -1] = 1 - mu, mu
    return Env(P, rho)


def discrete_env(M, A, rng, rho=0.0):
    """Rewards on {0.3, 0.4, 0.5, 0.6, 0.7, 0.8} with random probabilities per pair."""
    support = to_grid_idx([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    P = np.zeros((M, A, G))
    P[..., support] = rng.dirichlet(np.ones(6), size=(M, A))
    return Env(P, rho)


class Stats:
    def __init__(self, M, A):
        self.counts = np.zeros((M, A, G))

    def add(self, j, a, v):
        np.add.at(self.counts, (np.asarray(j), np.asarray(a), to_grid_idx(v)), 1)

    @property
    def N(self):
        return self.counts.sum(axis=-1)

    @property
    def mu_hat(self):
        N = self.N
        return np.where(N > 0, (self.counts @ GRID) / np.maximum(N, 1), 0.0)

    def P_hat(self):
        """Empirical distributions; pairs with no data get an optimistic point mass at 1."""
        N = self.N
        P = self.counts / np.maximum(N, 1)[..., None]
        P[N == 0, -1] = 1.0
        return P

    def ucb(self, L):
        """Empirical-Bernstein bonus: w = sqrt(2 (mu - mu^2) L / N) + L / (3N)."""
        N, m = self.N, self.mu_hat
        n = np.maximum(N, 1)
        w = np.sqrt(2 * np.clip(m - m ** 2, 0, None) * L / n) + L / (3 * n)
        return np.where(N > 0, np.minimum(m + w, 1.0), 1.0)


def algorithm1(P, m, alpha, I, rho, rng, K=24, iters=50, cap=None):
    """Greedy probe selection.

    P: (M, A, G) planning distributions (true offline, F_hat online).
    m: (M, A) means used for unprobed arms and for the stale part.
    rho: staleness used by the planner (rho = 0 gives Xu et al.'s Algorithm 1).

    Step 1: greedy chain S_0 c S_1 c ... c S_I, adding the arm with the largest
      gain in g(S), the best NSW using only probed arms (phi is increasing, so
      this is the same order as f_upper = phi(g)).
    Step 2: return the chain set with the largest Monte Carlo estimate of
      R(S) = (1 - alpha(|S|)) E[max_pi NSW]."""
    M, A, _ = P.shape
    chain, S = [[]], []
    for _ in range(I):
        cand = [a for a in range(A) if a not in S]
        r = np.concatenate([np.broadcast_to(m[:, S], (len(cand), M, len(S))),
                            m[:, cand].T[:, :, None]], axis=2)
        S = S + [cand[int(np.argmax(best_nsw(r, iters, cap)))]]
        chain.append(S)
    chain = [s for s in chain if alpha(len(s)) < 1]

    X = sample_grid(P, rng, K)
    Xt = (1 - rho) * X + rho * m[None]
    r = np.broadcast_to(m, (len(chain), K, M, A)).copy()
    for i, s in enumerate(chain):
        if s:
            r[i][:, :, s] = Xt[:, :, s]
    vals = best_nsw(r, iters, cap).mean(axis=1) * np.array([1 - alpha(len(s)) for s in chain])
    return chain[int(np.argmax(vals))]


def offline_opt(env, alpha, I, cap=None, K=300, iters=80, seed=12345, chunk=24):
    """Per-round value of the best (S, pi) with the true mu and rho, by exhaustive search (regret benchmark)."""
    rng = np.random.default_rng(seed)
    M, A, mu = env.M, env.A, env.mu
    X = sample_grid(env.P, rng, K)
    Xt = (1 - env.rho) * X + env.rho * mu[None]
    best = float(best_nsw(mu, iters, cap))
    for k in range(1, I + 1):
        if alpha(k) >= 1:
            continue
        subsets = list(combinations(range(A), k))
        for c in range(0, len(subsets), chunk):
            part = subsets[c:c + chunk]
            r = np.broadcast_to(mu, (len(part), K, M, A)).copy()
            for i, s in enumerate(part):
                r[i][:, :, list(s)] = Xt[:, :, list(s)]
            best = max(best, float((1 - alpha(k)) * best_nsw(r, iters, cap).mean(axis=1).max()))
    return best


METHODS = {
    "Random P+A":  dict(probe="random", assign="random"),
    "Greedy P+A":  dict(probe="greedy", assign="random"),
    "Non-Probing": dict(probe="none",   assign="nsw"),
    "OFMUP":       dict(probe="greedy", assign="nsw"),
    "OFMUP+fix":   dict(probe="greedy", assign="nsw", count_fix=True),
    "SA-OFMUP (known rho)":   dict(probe="greedy", assign="nsw", count_fix=True,
                                   stale_aware=True),
    "SA-OFMUP (learned rho)": dict(probe="greedy", assign="nsw", count_fix=True,
                                   stale_aware=True, learn_rho=True),
    "SA-OFMUP (learned rho, LCB)": dict(probe="greedy", assign="nsw", count_fix=True,
                                        stale_aware=True, learn_rho=True, lcb=True),
}


def run(env, method, T=3000, I=4, delta=0.1, seed=0, opt=None, n_random_probe=2,
        kappa=5.0, K=24, count_warmup=False, model="matrix", alpha_p=4, lcb_c=1.0):
    """One run. Returns cumulative regret, probes per round, rho_hat trace and
    the worst-off agent's average reward."""
    cfg = dict(probe="greedy", assign="nsw", count_fix=False, stale_aware=False,
               learn_rho=False, lcb=False)
    cfg.update(METHODS[method])
    rng = np.random.default_rng(seed)
    M, A, mu = env.M, env.A, env.mu
    if T <= M * A:
        raise ValueError(f"T must be larger than M*A = {M * A} (the warm-start rounds).")
    alpha = make_alpha(I, alpha_p)
    cap = M / A if model == "matrix" else None
    L = np.log(2 * M * A * T / delta)
    opt = offline_opt(env, alpha, I, cap) if opt is None else opt
    st = Stats(M, A)
    rho_hat = env.rho if (cfg["stale_aware"] and not cfg["learn_rho"]) else 0.0
    num = x_sum = 0.0
    n_pairs = 0
    sel = np.zeros((M, A))
    rho_point = 0.0
    inst = np.zeros(T)
    probes = np.zeros(T)
    rho_trace = np.zeros(T)
    point_trace = np.zeros(T)
    earned = np.zeros(M)


    for t in range(M * A):
        j, a = t % M, t // M
        st.add(j, a, env.probe([a], rng)[j, 0])
        inst[t] = opt if count_warmup else 0.0
        rho_trace[t] = rho_hat

    for t in range(M * A, T):
        rho_use = rho_hat if cfg["stale_aware"] else 0.0
        P_hat = st.P_hat()
        if cfg["probe"] == "none":
            S = []
        elif cfg["probe"] == "random":
            S = sorted(rng.choice(A, n_random_probe, replace=False).tolist())
        else:
            S = algorithm1(P_hat, P_hat @ GRID, alpha, I, rho_use, rng, K, cap=cap)

        if S:
            X = env.probe(S, rng)
            jj, aa = np.meshgrid(np.arange(M), S, indexing="ij")
            st.add(jj.ravel(), aa.ravel(), X.ravel())
        else:
            X = np.zeros((M, 0))
        U = st.ucb(L)

        if cfg["assign"] == "random":
            pi = np.full((M, A), 1.0 / A)
        else:
            r = U.copy()
            if S:
                r[:, S] = (1 - rho_use) * X + rho_use * U[:, S]
            pi = solve_nsw(r, cap=cap)

        arms = np.minimum((rng.random((M, 1)) > np.cumsum(pi, axis=1)).sum(axis=1), A - 1)
        Y = env.pull(arms, S, X, rng)
        earned += Y

        pos = {a: i for i, a in enumerate(S)}
        for j, a in enumerate(arms):
            if a in pos:
                if cfg["learn_rho"]:
                    num += X[j, pos[a]] - Y[j]
                    x_sum += X[j, pos[a]]
                    sel[j, a] += 1
                    n_pairs += 1
                if not cfg["count_fix"]:
                    st.add(j, a, Y[j])
            else:
                st.add(j, a, Y[j])
        if cfg["learn_rho"]:

            den = x_sum - float((sel * st.mu_hat).sum())
            rho_point = float(np.clip(num / den, 0.0, 1.0)) if den >= kappa else 0.0
            if cfg["lcb"]:

                rho_hat = (float(np.clip((num - lcb_c * np.sqrt(n_pairs)) / den, 0.0, 1.0))
                           if den >= kappa else 0.0)
            else:
                rho_hat = rho_point


        v = mu.copy()
        if S:
            v[:, S] = (1 - env.rho) * X + env.rho * mu[:, S]
        inst[t] = opt - (1 - alpha(len(S))) * float(nsw_value(v, pi))
        probes[t] = len(S)
        rho_trace[t] = rho_hat
        point_trace[t] = rho_point

    return dict(regret=np.cumsum(inst), probes=probes, rho_hat=rho_trace,
                rho_point=point_trace, worst_avg=float(earned.min() / (T - M * A)), opt=opt)


OUT, RUNS = "results", os.path.join("results", "runs")
os.makedirs(RUNS, exist_ok=True)
I_BUDGET = 4
SETTINGS = {"a": ("bernoulli", 12, 8), "b": ("bernoulli", 20, 10),
            "c": ("discrete", 12, 8), "d": ("discrete", 20, 10)}
PAPER = ["Random P+A", "Non-Probing", "Greedy P+A", "OFMUP"]
STALE = ["Non-Probing", "OFMUP", "OFMUP+fix", "SA-OFMUP (known rho)",
         "SA-OFMUP (learned rho)", "SA-OFMUP (learned rho, LCB)"]
LABEL = {"SA-OFMUP (known rho)": r"SA-OFMUP (known $\rho$)",
         "SA-OFMUP (learned rho)": "SA-OFMUP (point estimate)",
         "SA-OFMUP (learned rho, LCB)": "SA-OFMUP (LCB, final)"}
STYLE = {"Random P+A": dict(c="tab:blue"), "Non-Probing": dict(c="tab:orange"),
         "Greedy P+A": dict(c="tab:green"), "OFMUP": dict(c="tab:red"),
         "OFMUP+fix": dict(c="tab:red", ls=":"),
         "SA-OFMUP (known rho)": dict(c="tab:purple", ls="--"),
         "SA-OFMUP (learned rho)": dict(c="tab:gray"),
         "SA-OFMUP (learned rho, LCB)": dict(c="black", lw=2.2)}
lab = lambda m: LABEL.get(m, m)
_opt_cache = {}


def make_env(kind, M, A, seed, rho):
    rng = np.random.default_rng(1000 + seed)
    return (bernoulli_env if kind == "bernoulli" else discrete_env)(M, A, rng, rho)


def get_run(kind, M, A, rho, seed, method, T):
    """Run one (setting, rho, seed, method), or load it if it was saved before."""
    tag = f"{kind}_{M}x{A}_rho{rho}_T{T}_p{ALPHA_P}_s{seed}_{method.replace(' ', '_')}"
    path = os.path.join(RUNS, tag + ".npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    key = (kind, M, A, rho, seed)
    env = make_env(kind, M, A, seed, rho)
    if key not in _opt_cache:
        _opt_cache[key] = offline_opt(env, make_alpha(I_BUDGET, ALPHA_P), I_BUDGET, cap=M / A)
    t0 = time.time()
    r = run(env, method, T=T, I=I_BUDGET, seed=seed, opt=_opt_cache[key], alpha_p=ALPHA_P)
    out = {k: np.asarray(r[k]) for k in ["regret", "probes", "rho_hat", "rho_point", "worst_avg"]}
    np.savez(path, **out)
    print(f"  {kind} M={M} A={A} rho={rho} T={T} seed={seed} {method:28s} "
          f"regret={r['regret'][-1]:7.1f}  ({time.time() - t0:.0f}s)", flush=True)
    return out


def save_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def band(ax, R, m):
    mu, sd = R.mean(0), R.std(0)
    ax.plot(mu, label=lab(m), **STYLE[m])
    ax.fill_between(np.arange(len(mu)), mu - sd, mu + sd, alpha=0.15, color=STYLE[m]["c"])


def part1_reproduction(T, seeds):
    print(f"\n=== Part 1: reproduction of Figure 1 (rho = 0, T = {T}, {seeds} seeds) ===")
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    rows = []
    for ax, (panel, (kind, M, A)) in zip(axes, SETTINGS.items()):
        final = {}
        for m in PAPER:
            R = np.array([get_run(kind, M, A, 0.0, s, m, T)["regret"] for s in range(seeds)])
            band(ax, R, m)
            final[m] = R[:, -1].mean()
            rows.append([panel, kind, M, A, m, R[:, -1].mean(), R[:, -1].std()])
        o = final["OFMUP"]
        red = {m: 100 * (1 - o / final[m]) for m in PAPER if m != "OFMUP"}
        print(f"({panel}) OFMUP regret is lower by: " +
              ", ".join(f"{red[m]:.0f}% vs {m}" for m in red))
        ax.set_title(f"({panel}) {kind}, M={M}, A={A}")
        ax.set_xlabel("Time steps"); ax.set_ylabel("Cumulative regret"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_reproduction.png"), dpi=150); plt.show()
    save_csv("fig1_reproduction.csv",
             ["panel", "env", "M", "A", "method", "final_regret_mean", "final_regret_std"], rows)


def part2_staleness(T, seeds, rhos):
    print(f"\n=== Part 2: staleness sweep (setting a, rho = {rhos}, T = {T}, {seeds} seeds) ===")
    kind, M, A = SETTINGS["a"]
    res = {(m, r): [get_run(kind, M, A, r, s, m, T) for s in range(seeds)] for r in rhos for m in STALE}
    rows, seedrows = [], []
    for r in rhos:
        for m in STALE:
            runs = res[(m, r)]
            reg = [x["regret"][-1] for x in runs]
            prb = [x["probes"][M * A:].mean() for x in runs]
            wst = [float(x["worst_avg"]) for x in runs]
            est = [x["rho_point" if "LCB" in m else "rho_hat"][-1] for x in runs]
            rows.append([r, m, np.mean(reg), np.std(reg), np.mean(prb), np.mean(wst), np.mean(est)])
            seedrows.append([r, m] + [round(v, 1) for v in reg])
    save_csv("staleness_a.csv", ["rho", "method", "final_regret_mean", "final_regret_std",
                                 "probes_per_round", "worst_driver_avg_reward", "final_rho_estimate"], rows)
    save_csv("staleness_a_per_seed.csv", ["rho", "method"] + [f"seed{s}" for s in range(seeds)], seedrows)
    print(f"\n{'method':30s}" + "".join(f"  rho={r:<5}" for r in rhos))
    for m in STALE:
        print(f"{m:30s}" + "".join(f"  {np.mean([x['regret'][-1] for x in res[(m, r)]]):8.1f}" for r in rhos))

    fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))
    for m in [x for x in STALE if x != "OFMUP+fix"]:
        reg = np.array([[x["regret"][-1] for x in res[(m, r)]] for r in rhos])
        ax[0].errorbar(rhos, reg.mean(1), yerr=reg.std(1), marker="o", capsize=3, label=lab(m), **STYLE[m])
        ax[1].plot(rhos, [np.mean([x["probes"][M * A:].mean() for x in res[(m, r)]]) for r in rhos],
                   marker="o", **STYLE[m])
        ax[2].plot(rhos, [np.mean([float(x["worst_avg"]) for x in res[(m, r)]]) for r in rhos],
                   marker="o", **STYLE[m])
    for r, ls in zip([x for x in rhos if x > 0], ["--", "-", "-.", ":"]):
        tr = np.mean([x["rho_point"] for x in res[("SA-OFMUP (learned rho, LCB)", r)]], axis=0)
        ax[3].plot(tr, c="black", ls=ls, label=rf"point estimate, true $\rho$={r}")
        ax[3].axhline(r, ls=":", c="gray")
    titles = ["Final cumulative regret", "Arms probed per round",
              "Worst-off agent: average reward", r"SA-OFMUP (LCB): estimate of $\rho$"]
    for a, t, xl in zip(ax, titles, [r"staleness $\rho$"] * 3 + ["Time steps"]):
        a.set_title(t); a.set_xlabel(xl); a.grid(alpha=0.3)
    ax[2].set_ylim(0, 0.7); ax[3].set_ylim(0, 1); ax[3].legend(loc="lower right", fontsize=9)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(OUT, "staleness_a.png"), dpi=150); plt.show()


def part3_long(T, seeds):
    print(f"\n=== Part 3: the paper's horizon (setting a, T = {T}, {seeds} seeds) ===")
    kind, M, A = SETTINGS["a"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.2))
    rows = []
    for m in ["Random P+A", "Greedy P+A", "Non-Probing", "OFMUP"]:
        R = np.array([get_run(kind, M, A, 0.0, s, m, T)["regret"] for s in range(seeds)])
        band(ax[0], R, m)
        rows.append([0.0, m] + list(R[:, -1].round(1)) + [R[:, -1].mean().round(1)])
    o = rows[-1][-1]
    print("rho = 0: OFMUP regret is lower by " +
          ", ".join(f"{100 * (1 - o / r[-1]):.0f}% vs {r[1]}" for r in rows[:-1]))
    base = None
    for m in ["Non-Probing", "OFMUP", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho, LCB)"]:
        R = np.array([get_run(kind, M, A, 0.8, s, m, T)["regret"] for s in range(seeds)])
        band(ax[1], R, m)
        rows.append([0.8, m] + list(R[:, -1].round(1)) + [R[:, -1].mean().round(1)])
        if m == "OFMUP":
            base = R
        elif base is not None:
            ax[2].plot((base - R).mean(0), label=lab(m), **STYLE[m])
    ax[2].axhline(0, c="tab:red", lw=1)
    titles = [rf"$\rho=0$: reproduction at $T={T}$", rf"$\rho=0.8$: staleness at $T={T}$",
              r"$\rho=0.8$: regret saved relative to OFMUP"]
    for a, t in zip(ax, titles):
        a.set_title(t); a.set_xlabel("Time steps"); a.grid(alpha=0.3); a.legend(fontsize=9)
    ax[0].set_ylabel("Cumulative regret"); ax[2].set_ylabel("OFMUP regret minus method regret")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "long_T3000.png"), dpi=150); plt.show()
    save_csv("long_T3000.csv", ["rho", "method"] + [f"seed{s}" for s in range(seeds)] + ["mean"], rows)
    for r in rows:
        print(f"  rho={r[0]}  {r[1]:28s} mean regret {r[-1]}")


if QUICK:
    T_MAIN, T_LONG, SEEDS, RHOS = 300, 400, 2, [0.0, 0.8]
start = time.time()
print(f"Mode: {'QUICK test' if QUICK else 'full experiments'} | T_MAIN={T_MAIN}, "
      f"T_LONG={T_LONG}, SEEDS={SEEDS}, RHOS={RHOS}, ALPHA_P={ALPHA_P}")
if RUN_REPRO:
    part1_reproduction(T_MAIN, SEEDS)
if RUN_STALE:
    part2_staleness(T_MAIN, SEEDS, RHOS)
if RUN_LONG:
    part3_long(T_LONG, SEEDS)
print(f"\nDone in {(time.time() - start) / 60:.1f} minutes. Plots and CSV files are in '{OUT}/'.")
