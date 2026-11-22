"""
OFMUP (Xu, Liu, Mattei & Zheng, AAAI-26) and the staleness-aware extension SA-OFMUP.

Switches (all off = OFMUP):
    stale_aware : use (1 - rho) * probe + rho * mean instead of the raw probe value
    count_fix   : a pull from an arm probed in the same round is not a new sample
    learn_rho   : estimate the staleness probability rho online from probe-pull pairs
    lcb         : plan with a lower confidence bound on rho
"""
from __future__ import annotations

from itertools import combinations

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


def taxi_env(csv_path, M, A, rng, rho=0.0, n_pickups=500,
             lon_bounds=(-74.05, -73.75), lat_bounds=(40.60, 40.90)):
    """NYC Yellow Taxi 2016 setup (semi-synthetic, as in Xu et al.):
    vehicles = agents at fixed random positions inside the city bounds;
    arms = the A busiest 0.01-degree pickup cells; reward = 1 - normalised
    Manhattan distance from the vehicle to a random pickup in that cell.
    Needs a CSV with columns pickup_longitude, pickup_latitude."""
    import pandas as pd

    df = pd.read_csv(csv_path, usecols=["pickup_longitude", "pickup_latitude"]).dropna()
    df = df[df.pickup_longitude.between(*lon_bounds) & df.pickup_latitude.between(*lat_bounds)]
    cell = (np.floor(df.pickup_longitude / 0.01).astype(int).astype(str) + "_"
            + np.floor(df.pickup_latitude / 0.01).astype(int).astype(str))
    zones = cell.value_counts().index[:A]
    veh = np.column_stack([rng.uniform(*lon_bounds, M), rng.uniform(*lat_bounds, M)])
    dmax = (lon_bounds[1] - lon_bounds[0]) + (lat_bounds[1] - lat_bounds[0])
    P = np.zeros((M, A, G))
    for a, c in enumerate(zones):
        pts = df[cell == c][["pickup_longitude", "pickup_latitude"]].to_numpy()
        pts = pts[rng.choice(len(pts), size=min(n_pickups, len(pts)), replace=False)]
        d = np.abs(veh[:, None, 0] - pts[None, :, 0]) + np.abs(veh[:, None, 1] - pts[None, :, 1])
        idx = to_grid_idx(1 - d / dmax)
        for j in range(M):
            P[j, a] = np.bincount(idx[j], minlength=G) / idx.shape[1]
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
