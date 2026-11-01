"""
Simulation study of the writing sample. All figures and numbers of Sections 3 and 5 and Appendix C.

Data: X ~ N(0, I_3), p(x) = sigmoid(-2.2 + x1 + 0.5 x1 x2), c(x) = sigmoid(-1 + 1.5 x2 - 0.3 x1 + 0.5 x3),
given Y = 1 the loss is Beta(4c, 4(1 - c)), B = 1, alpha_cov = 0.10.

Commands (repo root)                                   output in results/simulation/
  python src/simulation.py fig1a                       fig_shape.png    Figure 1(a), prints S_cc, a_low, theta*, review shares
  python src/simulation.py fig1b --step 0.001          fig_unique.png   Figure 1(b), prints the uniqueness check
  python src/simulation.py fig2                        fig_theorem2.png and fig_theorem2_data.npz   Figure 2, Section 5
  python src/simulation.py fig2 --plot-only            redraw Figure 2 from the saved data
  python src/simulation.py fig2 --plot-only --with-bias  also draw a three-panel version with the bias of part b
  python src/simulation.py c0                          prints C_0 and e_max of Theorem 2(b) (Discussion)
Run times: fig1a seconds, fig1b a few minutes with --step 0.001, fig2 about 25 minutes, c0 about 8 minutes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/simulation"
A_COV, RATIO, B = 0.10, 0.5, 1.0
INK, MUTED, GRID, CLOUD = "#1f2328", "#6e7781", "#d0d7de", "#c9ccd1"
BLUE, ORANGE, GREEN, DOT = "#0b4f9c", "#c4661f", "#3a9a5b", "#d9622b"
# same colours as the LendingClub figures: theta = 0 grey, theta = 1 orange, separate cut-offs green, blend blue
COL = {"p": "#5a6068", "pc": "#c4661f", "rect": "#3a9a5b", "opt": "#0b4f9c"}


# ---------------------------------------------------------------- data
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


def draw_x(n, rng):
    return rng.standard_normal((n, 3))


def true_scores(x):
    p = sigmoid(-2.2 + x[:, 0] + 0.5 * x[:, 0] * x[:, 1])
    c = sigmoid(-1.0 + 1.5 * x[:, 1] - 0.3 * x[:, 0] + 0.5 * x[:, 2])
    return p, c


def draw_labels(p, c, rng, k=4.0):
    """Y ~ Bernoulli(p). Given Y = 1, C ~ Beta(k c, k (1 - c)), so that E[C | X, Y = 1] = c. C = 0 if Y = 0."""
    y = (rng.random(len(p)) < p).astype(int)
    cc = np.clip(c, 1e-6, 1 - 1e-6)
    loss = np.where(y == 1, rng.beta(k * cc, k * (1 - cc)), 0.0)
    return y, loss


# ---------------------------------------------------------------- weight from scores (no labels)
def passes(theta, ps, pcs, a1, a2):
    """Cases passed by the cut-off on (1 - theta) ps + theta pcs, for as long as both constraints hold."""
    n = len(ps)
    o = np.argsort((1 - theta) * ps + theta * pcs, kind="stable")
    k_cov = np.searchsorted(np.cumsum(ps[o]) / n, a1, side="right")
    k_sev = np.searchsorted(np.cumsum(pcs[o]) / n, a2, side="right")
    return int(min(k_cov, k_sev))


def best_weight(ps, pcs, a_sev):
    """Weight that passes the most cases, on a coarse grid refined twice. Uses no label.
    By linear-programming duality it equals the weight from the dual variables of the plug-in linear program."""
    a1 = A_COV * ps.mean()
    best_t, best_v = 0.0, -1
    for lo, hi, step in [(0.0, 1.0, 0.02), (None, None, 0.001), (None, None, 0.0001)]:
        if lo is None:
            lo, hi = max(0.0, best_t - 25 * step), min(1.0, best_t + 25 * step)
        for th in np.arange(lo, hi + step / 2, step):
            v = passes(th, ps, pcs, a1, a_sev)
            if v > best_v:
                best_t, best_v = float(th), v
    return best_t


def best_weight_value(ps, pcs, a_sev):
    """Same search as best_weight, returning the weight and the number of cases passed."""
    a1 = A_COV * ps.mean()
    best_t, best_v = 0.0, -1
    for lo, hi, step in [(0.0, 1.0, 0.02), (None, None, 0.001), (None, None, 0.0001)]:
        if lo is None:
            lo, hi = max(0.0, best_t - 25 * step), min(1.0, best_t + 25 * step)
        for th in np.arange(lo, hi + step / 2, step):
            v = passes(th, ps, pcs, a1, a_sev)
            if v > best_v:
                best_t, best_v = float(th), v
    return best_t, best_v


# ---------------------------------------------------------------- calibration (multi-risk CRC)
def crc_cutoff(s_cal, y_cal, loss_cal, a_sev):
    """Cut-off t_hat = min(t_cov, t_sev). The rule passes a new case with score s < t_hat."""
    n = len(s_cal)
    # coverage: #{defaults with s < t} <= alpha_cov (n1 + 1) - 1
    sd = np.sort(s_cal[y_cal == 1])
    n1 = len(sd)
    m = int(np.floor(A_COV * (n1 + 1) - 1 + 1e-12))
    t_cov = -np.inf if m < 0 else (np.inf if m >= n1 else sd[m])
    # severity: sum of losses with s < t <= alpha_sev (n + 1) - B
    o = np.argsort(s_cal, kind="stable")
    cum = np.cumsum(loss_cal[o])
    b = a_sev * (n + 1) - B
    k = int(np.searchsorted(cum, b + 1e-15, side="right"))             # largest prefix with cum <= b
    t_sev = -np.inf if b < 0 else (np.inf if k >= n else s_cal[o][k])
    return min(t_cov, t_sev)


# ---------------------------------------------------------------- shared population quantities
def population(big, seed):
    rng = np.random.default_rng(seed)
    x = draw_x(big, rng)
    p, c = true_scores(x)
    pc = p * c
    a1 = A_COV * p.mean()
    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / big, a1, side="right"))
    s_cc = np.cumsum(pc[o])[k - 1] / big
    a_sev = RATIO * s_cc
    theta_star = best_weight(p, pc, a_sev)
    return x, p, c, a_sev, theta_star


# ---------------------------------------------------------------- part (a)
def part_a(theta_star, a_sev, reps, seed):
    sizes = [1_000, 4_000, 16_000, 64_000, 256_000]
    rng = np.random.default_rng(seed + 1)
    err = np.empty((len(sizes), reps))
    for i, n_u in enumerate(sizes):
        t0 = time.time()
        for r in range(reps):
            p, c = true_scores(draw_x(n_u, rng))
            err[i, r] = abs(best_weight(p, p * c, a_sev) - theta_star)
        print(f"  (a) n_U = {n_u:7d}: mean |theta_hat - theta*| = {err[i].mean():.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
    return np.array(sizes), err


# ---------------------------------------------------------------- part (b)
def score_error(p, c, ph, ch):
    return (np.mean(np.abs(ph - p)) + np.mean(np.abs(ph * ch - p * c))
            + A_COV * abs(np.mean(ph - p)))


def part_b(x, p, c, theta_star, a_sev):
    rows = []
    for m in [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]:    # scale the expected loss given default
        ch = np.clip(m * c, 0, 1)
        rows.append(("c scaled", m, score_error(p, c, p, ch), best_weight(p, p * ch, a_sev)))
        print(f"  (b) c x {m:.2f}: e = {rows[-1][2]:.5f}, theta_tilde = {rows[-1][3]:.4f}", flush=True)
    for s in [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]:    # scale the logit of the probability
        ph = sigmoid(s * logit(p))
        rows.append(("p logit scaled", s, score_error(p, c, ph, c), best_weight(ph, ph * c, a_sev)))
        print(f"  (b) logit p x {s:.2f}: e = {rows[-1][2]:.5f}, theta_tilde = {rows[-1][3]:.4f}", flush=True)
    return rows


# ---------------------------------------------------------------- part (c)
def part_c(theta_star, a_sev, reps, n_eval, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    sizes = [2_000, 5_000, 20_000, 50_000, 200_000]
    rng = np.random.default_rng(seed + 2)

    # evaluation sample with the true scores. Shares are expectations over X, so the true p is enough.
    x_ev = draw_x(n_eval, rng)
    p_ev, c_ev = true_scores(x_ev)
    pc_ev = p_ev * c_ev
    pi1 = p_ev.mean()
    a1_ev = A_COV * pi1
    best = passes(theta_star, p_ev, pc_ev, a1_ev, a_sev)
    r_star = 1 - best / n_eval                                           # smallest share sent to review

    def evaluate(score_ev, t_hat):
        passed = score_ev < t_hat
        review = 1 - passed.mean()
        cov = (p_ev * passed).mean() / pi1                               # Pr(pass | Y = 1)
        sev = (pc_ev * passed).mean()                                    # E[C 1{Y = 1, pass}]
        return review, cov, sev

    out = {k: np.empty((len(sizes), reps)) for k in ["cal", "wt", "full", "cov", "sev"]}
    for i, n in enumerate(sizes):
        t0 = time.time()
        for r in range(reps):
            # three independent samples of size n: training, unlabeled, calibration
            x_tr, x_un, x_ca = draw_x(n, rng), draw_x(n, rng), draw_x(n, rng)
            p_tr, c_tr = true_scores(x_tr)
            y_tr, l_tr = draw_labels(p_tr, c_tr, rng)
            p_un, c_un = true_scores(x_un)
            p_ca, c_ca = true_scores(x_ca)
            y_ca, l_ca = draw_labels(p_ca, c_ca, rng)

            # estimated scores, fitted on the training sample only
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=r).fit(x_tr, y_tr)
            d = y_tr == 1
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=r).fit(x_tr[d], l_tr[d])
            ph = lambda xx: clf.predict_proba(xx)[:, 1]
            ch = lambda xx: np.clip(reg.predict(xx), 0, B)

            # run 1: true scores, theta*
            h_ca = (1 - theta_star) * p_ca + theta_star * p_ca * c_ca
            h_ev = (1 - theta_star) * p_ev + theta_star * pc_ev
            rev1, _, _ = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            # run 2: true scores, theta_hat from the unlabeled sample
            th2 = best_weight(p_un, p_un * c_un, a_sev)
            h_ca = (1 - th2) * p_ca + th2 * p_ca * c_ca
            h_ev = (1 - th2) * p_ev + th2 * pc_ev
            rev2, _, _ = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            # run 3: the practical pipeline. estimated scores, theta_hat from them, calibrated cut-off
            pu, cu = ph(x_un), ch(x_un)
            th3 = best_weight(pu, pu * cu, a_sev)
            pca, cca = ph(x_ca), ch(x_ca)
            pev, cev = ph(x_ev), ch(x_ev)
            h_ca = (1 - th3) * pca + th3 * pca * cca
            h_ev = (1 - th3) * pev + th3 * pev * cev
            rev3, cov3, sev3 = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            out["cal"][i, r] = rev1 - r_star
            out["wt"][i, r] = rev2 - rev1
            out["full"][i, r] = rev3 - r_star
            out["cov"][i, r], out["sev"][i, r] = cov3, sev3
        print(f"  (c) n = {n:7d}: extra review of the pipeline = {100 * out['full'][i].mean():.2f} pp "
              f"({time.time() - t0:.0f} s)", flush=True)
    return np.array(sizes), out, r_star


# ---------------------------------------------------------------- plotting
def bias_panel(ax, store, letter):
    """Bias |theta_tilde - theta*| against the error e of the distorted values, natural log on both axes.
    One series per distorted quantity, so that the two can be compared at the same error e."""
    kinds, dist = store["b_kind"], store["b_dist"]
    e_all, b_all = store["b_e"], store["b_bias"]
    series = [("c scaled", BLUE, "o", r"$c$ scaled"), ("p logit scaled", "#8a5cb8", "s", r"logit of $p$ scaled")]
    print("Bias panel: distortion m, error e, bias |theta_tilde - theta*|")
    for kind, col, mk, lab in series:
        sel = (kinds == kind) & (b_all > 0)
        order = np.argsort(e_all[sel])
        x, y, m = np.log(e_all[sel][order]), np.log(b_all[sel][order]), dist[sel][order]
        # markers only. The distortion m of each point is printed, not drawn, to keep the panel readable
        ax.plot(x, y, color=col, marker=mk, ms=3.5, lw=0, label=lab, zorder=3)
        slope, icpt = np.polyfit(x, y, 1)
        print(f"  {kind:15s} log-log slope {slope:.2f}")
        for mi, ei, bi in zip(m, e_all[sel][order], b_all[sel][order]):
            print(f"      m = {mi:.2f}  e = {ei:.5f}  bias = {bi:.4f}")
        store.setdefault("_fit", {})[kind] = (slope, icpt)
        dropped = int(np.sum((kinds == kind) & (b_all <= 0)))
        if dropped:
            print(f"      ({dropped} point(s) with zero bias not drawn on the log scale)")
    # fair comparison: predicted bias of each series at common errors inside both ranges
    fits = store.pop("_fit")
    lo = max(e_all[(kinds == k) & (b_all > 0)].min() for k, *_ in series)
    hi = min(e_all[(kinds == k) & (b_all > 0)].max() for k, *_ in series)
    if lo < hi:
        print("  Bias at the same error e (from the log-log fits):")
        for e0 in np.geomspace(lo, hi, 3):
            vals = {k: np.exp(fits[k][1] + fits[k][0] * np.log(e0)) for k, *_ in series}
            print(f"      e = {e0:.5f}: c scaled {vals['c scaled']:.4f}, logit p scaled {vals['p logit scaled']:.4f}, "
                  f"ratio p/c {vals['p logit scaled'] / vals['c scaled']:.2f}")
    else:
        print("  The two series do not overlap in e, so they cannot be compared at the same error.")
    # all points together: fitted slope (the 0.83 of the text) and a reference line of slope 1/2, the rate of the
    # bound C_0 sqrt(e). The reference line starts at the point with the largest e and shows the slope only, not the
    # bound itself, whose constant is far larger (see the c0 command).
    keep = b_all > 0
    xa, ya = np.log(e_all[keep]), np.log(b_all[keep])
    slope_all, icpt_all = np.polyfit(xa, ya, 1)
    xs = np.linspace(xa.min(), xa.max(), 50)
    ax.plot(xs, icpt_all + slope_all * xs, color=INK, ls="--", lw=1.0, zorder=1)
    j = int(np.argmax(xa))
    ax.plot(xs, ya[j] + 0.5 * (xs - xa[j]), color=MUTED, ls=":", lw=1.2, zorder=1)
    print(f"  all {int(keep.sum())} points: log-log slope {slope_all:.2f} (dashed). Dotted line: slope 1/2 from the "
          f"point with the largest e.")
    ax.set_xlabel(f"$\\log e$\n({letter})", fontsize=9, color=INK, linespacing=1.8)
    ax.set_ylabel(r"$\log|\tilde\theta-\theta^*|$", fontsize=9, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", handletextpad=0.2)


def plot(store, path, with_bias=False):
    """Figure 2 of the paper. Two panels: (a) sampling error of the weight, (b) extra review of the full method.
    With with_bias=True, three panels: (a) sampling error, (b) bias from distorted p or c, (c) full method."""
    k = 3 if with_bias else 2
    fig, axes = plt.subplots(1, k, figsize=(6.5, 2.5 if with_bias else 2.7))   # printed at the full text width
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8 if k == 2 else 7)
    letters = "abc"

    if "a_sizes" in store:
        sizes, err = store["a_sizes"], store["a_err"]
        x, y = np.log(sizes), np.log(err.mean(axis=1))
        ax = axes[0]
        ax.plot(x, y[0] - 0.5 * (x - x[0]), color=MUTED, ls=":", lw=1.3)        # slope -1/2, explained in caption
        ax.plot(x, y, color=BLUE, marker="o", ms=4, lw=1.4, label=r"$\log|\hat\theta-\theta^*|$")
        ax.set_xlabel(f"$\\log n_U$\n({letters[0]})", fontsize=9, color=INK, linespacing=1.8)
        ax.set_ylabel(r"$\log|\hat\theta-\theta^*|$", fontsize=9, color=INK)
        ax.legend(fontsize=8 if k == 2 else 7, frameon=False)
    else:
        axes[0].set_visible(False)

    if with_bias:
        if "b_e" in store:
            bias_panel(axes[1], store, letters[1])
        else:
            axes[1].set_visible(False)
            print("No saved part b. Run: python src/simulation.py fig2 --parts b --with-bias")

    if "c_sizes" in store:
        sizes = store["c_sizes"]
        full = 100 * store["c_full"].mean(axis=1)
        se = 100 * store["c_full"].std(axis=1, ddof=1) / np.sqrt(store["c_full"].shape[1])
        ax = axes[-1]
        ax.plot(np.log(sizes), full, color=INK, marker="o", ms=4, lw=1.4)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(f"$\\log n$\n({letters[k - 1]})", fontsize=9, color=INK, linespacing=1.8)
        ax.set_ylabel("extra share sent to review (pp)" if k == 2 else "extra review (pp)", fontsize=9, color=INK)
        print(f"Figure 2({letters[k - 1]}): n, extra review (pp), SE (pp)  [SE goes in the caption]")
        for n_, f_, s_ in zip(sizes, full, se):
            print(f"  {int(n_):7d}  {f_:6.2f}  {s_:5.2f}")
    else:
        axes[-1].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)


# ---------------------------------------------------------------- Figure 1(a) helpers
def largest_prefix(score, p, pc, a1, a2):
    """Pass the cases with the lowest score for as long as both constraints hold."""
    n = len(score)
    order = np.argsort(score, kind="stable")
    cov = np.cumsum(p[order]) / n
    sev = np.cumsum(pc[order]) / n
    ok = (cov <= a1) & (sev <= a2)
    k = n if ok.all() else int(np.argmin(ok))
    passed = np.zeros(n, bool)
    passed[order[:k]] = True
    return passed


def single_cut(score, use, level):
    """Cut-off on score that meets one constraint, given by the per-case use and its level."""
    n = len(score)
    order = np.argsort(score, kind="stable")
    k = int(np.searchsorted(np.cumsum(use[order]) / n, level, side="right"))
    return score[order[k - 1]]


# ---------------------------------------------------------------- Theorem 2(b) constant helpers
def use(lam, Z):
    """M(lambda) = E[Z 1{lambda . Z < 1}], the use of the two constraints by the rule with prices lambda."""
    m = Z @ lam < 1
    return Z[m].sum(axis=0) / len(Z)


def hessian(lam, Z, h):
    """Hessian of the dual function, G(lambda) = - dM / dlambda, by central differences."""
    H = np.empty((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        H[:, j] = -(use(lam + e, Z) - use(lam - e, Z)) / (2 * h)
    return (H + H.T) / 2


def grad_theta_norm(lam):
    """|| grad theta(lambda) || for theta = lambda_2 B / (lambda_1 + lambda_2 B)."""
    l1, l2 = lam
    S = l1 + l2 * B
    return np.hypot(-l2 * B / S ** 2, l1 * B / S ** 2)


# ---------------------------------------------------------------- Figure 1(a)
def fig1a(args):
    """Figure 1(a). Pass regions of four rules in the plane of p and pc. Also prints S_cc and a_low."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    p, c = true_scores(draw_x(args.n, rng))
    pc = p * c
    n = args.n
    pi1 = p.mean()
    a_cov = 0.10
    a1 = a_cov * pi1                              # coverage in linear form, E[phi p] <= alpha_cov pi_1

    # S_cc: uncovered severity of the coverage-only rule (cut-off on p)
    t_cc = single_cut(p, p, a1)
    S_cc = pc[p <= t_cc].sum() / n
    a2 = args.ratio * S_cc                        # alpha_sev

    # lower threshold: largest alpha_sev at which the severity-only rule also meets coverage
    order = np.argsort(pc, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[order]) / n, a1, side="right"))
    a_low = np.cumsum(pc[order])[k - 1] / n

    rules = {}
    rules["p"] = largest_prefix(p, p, pc, a1, a2)                       # theta = 0
    rules["pc"] = largest_prefix(pc, p, pc, a1, a2)                     # theta = 1
    t1 = single_cut(p, p, a1)                                           # coverage threshold on p
    t2 = single_cut(pc, pc, a2)                                         # severity threshold on pc
    rules["rect"] = (p <= t1) & (pc <= t2)                              # two cut-offs, calibrated separately
    best = None
    for th in np.round(np.arange(0, 1.0001, 0.01), 2):                 # optimal weight on a grid
        passed = largest_prefix((1 - th) * p + th * pc, p, pc, a1, a2)
        if best is None or passed.sum() > best[1].sum():
            best = (th, passed)
    th_star, rules["opt"] = best

    tot = pc.mean()
    print(f"alpha_cov = {a_cov:.2f}, pi_1 = {pi1:.4f}")
    print(f"S_cc      = {S_cc:.5f}  ({100 * S_cc / tot:.2f}% of default loss)")
    print(f"alpha_sev = {a2:.5f}  ({100 * a2 / tot:.2f}% of default loss, {args.ratio:.2f} S_cc)")
    print(f"a_low     = {a_low:.5f}  ({100 * a_low / tot:.2f}% of default loss, {a_low / S_cc:.3f} S_cc)")
    print(f"theta*    = {th_star:.2f}")
    names = {"p": "Probability only (theta=0)", "pc": "Expected loss only (theta=1)",
             "rect": "Two cut-offs on p and pc", "opt": f"Blend h_theta (theta*={th_star:.2f})"}
    print(f"\n{'rule':34s} {'review %':>9s} {'coverage used':>14s} {'severity used':>14s}")
    for key in ["p", "pc", "rect", "opt"]:
        v = rules[key]
        print(f"{names[key]:34s} {100 * (1 - v.mean()):9.1f} "
              f"{(p * v).sum() / n / pi1:14.3f} {(pc * v).sum() / n / a2:14.2f}")

    # boundaries in the (p, pc) plane
    tp = p[rules["p"]].max()
    tpc = pc[rules["pc"]].max()
    h = (1 - th_star) * p + th_star * pc
    t_opt = h[rules["opt"]].max()
    rev = {key: 100 * (1 - v.mean()) for key, v in rules.items()}

    idx = rng.choice(n, 20_000, replace=False)
    xmax, ymax = 0.2, 0.06
    fig, ax = plt.subplots(figsize=(3.4, 2.75))   # printed at half the text width
    ax.scatter(p[idx], pc[idx], s=0.8, c=CLOUD, lw=0, zorder=1)

    # diagonal pc = p: no case lies above it, since c <= B = 1
    if not args.no_diagonal:
        ax.plot([0, ymax], [0, ymax], color=MUTED, lw=0.8, ls=":", zorder=2)

    # probability only: vertical line p = tp, drawn up to the diagonal (pc <= p = tp)
    ax.plot([tp, tp], [0, min(tp, ymax)], color=COL["p"], lw=2, zorder=3,
            label=rf"$\theta=0$: {rev['p']:.1f}%")

    # expected loss only: horizontal line pc = tpc, drawn from the diagonal (p >= tpc) to the right
    ax.plot([tpc, xmax], [tpc, tpc], color=COL["pc"], lw=2, zorder=3,
            label=rf"$\theta=1$: {rev['pc']:.1f}%")

    # two cut-offs: rectangle p <= t1 and pc <= t2, top edge drawn from the diagonal (p >= t2)
    ax.plot([t1, t1, min(t2, t1)], [0, min(t2, t1), min(t2, t1)], color=COL["rect"], lw=2, zorder=3,
            label=rf"two cut-offs: {rev['rect']:.1f}%")

    # blend: line (1 - theta) p + theta pc = t_opt, from the diagonal (p = t_opt) to the axis (pc = 0)
    if 0 < th_star < 1:
        x_start = t_opt                              # meets the diagonal pc = p
        x_end = min(t_opt / (1 - th_star), xmax)     # meets the axis pc = 0
        xs = np.linspace(x_start, x_end, 200)
        ax.plot(xs, (t_opt - (1 - th_star) * xs) / th_star, color=COL["opt"], lw=3, zorder=4,
                label=rf"$\theta^*={th_star:.2f}$: {rev['opt']:.1f}%")
    elif th_star == 0:
        ax.plot([t_opt, t_opt], [0, min(t_opt, ymax)], color=COL["opt"], lw=3, zorder=4,
                label=rf"Blend $h_\theta$, $\theta^*=0$: review {rev['opt']:.1f}%")
    else:
        ax.plot([t_opt, xmax], [t_opt, t_opt], color=COL["opt"], lw=3, zorder=4,
                label=rf"Blend $h_\theta$, $\theta^*=1$: review {rev['opt']:.1f}%")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("default probability $p(x)$", fontsize=8.5, color=INK)
    ax.set_ylabel("expected loss $p(x)c(x)$", fontsize=8.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID,
              title="sent to review", title_fontsize=7, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out / "fig_shape.png", dpi=300)
    print(f"\nSaved {out / 'fig_shape.png'}")


# ---------------------------------------------------------------- Figure 1(b)
def fig1b(args):
    """Figure 1(b). Best weight against the severity level, with a uniqueness check per level."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    p, c = true_scores(draw_x(args.n, np.random.default_rng(args.seed)))
    pc = p * c
    n = args.n
    a1 = 0.10 * p.mean()                                 # coverage in linear form

    # S_cc and a_low
    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, a1, side="right"))
    S_cc = np.cumsum(pc[o])[k - 1] / n
    o = np.argsort(pc, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, a1, side="right"))
    a_low = np.cumsum(pc[o])[k - 1] / n

    levels = np.geomspace(0.5 * a_low, 1.5 * S_cc, args.levels)    # alpha_sev, log-spaced
    m = int(round(1 / args.step))
    thetas = np.linspace(0.0, 1.0, m + 1)

    # review[i, j]: share sent to review at level i with weight j
    review = np.empty((len(levels), len(thetas)))
    for j, th in enumerate(thetas):
        if j % 1000 == 0 and len(thetas) > 1000:
            print(f"  theta {j} of {len(thetas) - 1}", flush=True)
        o = np.argsort((1 - th) * p + th * pc, kind="stable")
        cov = np.cumsum(p[o]) / n
        sev = np.cumsum(pc[o]) / n
        k_cov = np.searchsorted(cov, a1, side="right")              # largest prefix meeting coverage
        k_sev = np.searchsorted(sev, levels, side="right")          # largest prefix meeting severity, per level
        review[:, j] = 1.0 - np.minimum(k_cov, k_sev) / n
    excess = 100 * (review - review.min(axis=1, keepdims=True))     # percentage points
    theta_star = thetas[np.argmin(review, axis=1)]

    # uniqueness check. The weights within tol of the minimum must form one contiguous block,
    # and the curve must rise on both sides of that block.
    tol = 100 / n                                                   # one case, in percentage points
    n_blocks, widths, lo, hi = [], [], [], []
    for row in excess:
        near = np.flatnonzero(row <= tol)
        blocks = 1 + int(np.sum(np.diff(near) > 1))
        n_blocks.append(blocks)
        widths.append(thetas[near[-1]] - thetas[near[0]])
        lo.append(thetas[near[0]])
        hi.append(thetas[near[-1]])
    lo, hi = np.array(lo), np.array(hi)
    n_blocks = np.array(n_blocks)

    tot = pc.mean()
    print(f"a_low     = {a_low:.5f}  ({100 * a_low / tot:.2f}% of default loss)")
    print(f"S_cc      = {S_cc:.5f}  ({100 * S_cc / tot:.2f}% of default loss)")
    print(f"levels    = {len(levels)} from {levels[0]:.5f} to {levels[-1]:.5f}")
    print(f"rows with a single minimizing block: {int(np.sum(n_blocks == 1))} of {len(levels)}")
    print(f"largest width of that block in theta: {max(widths):.4f}  (grid step {args.step:g})")
    bad = np.flatnonzero(n_blocks > 1)
    if bad.size:
        print("rows with more than one block (alpha_sev):", ", ".join(f"{levels[i]:.5f}" for i in bad))
    print(f"\n{'alpha_sev':>10s} {'theta*':>8s} {'review %':>9s}")
    for i in np.linspace(0, len(levels) - 1, 12).astype(int):
        print(f"{levels[i]:10.5f} {theta_star[i]:8.4f} {100 * review[i].min():9.1f}")

    # best weight against the severity level
    fig, ax = plt.subplots(figsize=(3.4, 2.75))   # printed at half the text width, next to fig_shape
    for y in (0.0, 1.0):
        ax.axhline(y, color=GRID, lw=0.8, ls=":", zorder=1)
    for x, lab in [(a_low, "a\u0332"), (S_cc, r"$S_{\mathrm{cc}}$")]:
        ax.axvline(x, color=INK, lw=0.9, ls="--", zorder=1)
        ax.text(x, 1.03, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.plot(levels, theta_star, lw=0, marker="o", ms=2.6, mfc=DOT, mec="white", mew=0.3, zorder=3)
    ax.set_xscale("log")
    ticks = [t for t in [0.0003, 0.0005, 0.001, 0.002, 0.005] if levels[0] <= t <= levels[-1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(levels[0] / 1.05, levels[-1] * 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"severity level $\alpha_{\mathrm{sev}}$", fontsize=8.5, color=INK)
    ax.set_ylabel(r"best weight $\theta^*$", fontsize=8.5, color=INK)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7.5)
    fig.tight_layout()
    path = out / "fig_unique.png"
    fig.savefig(path, dpi=300)
    print(f"\nSaved {path}")


# ---------------------------------------------------------------- Figure 2
def fig2(args):
    """Figure 2. Parts a (sampling error), b (bias, printed for the text) and c (full method, panel b)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "fig_theorem2_data.npz"
    fig_path = out / "fig_theorem2.png"

    # results saved by earlier runs, so that parts can be computed separately and redrawn
    store = dict(np.load(data_path, allow_pickle=True)) if data_path.exists() else {}
    if args.plot_only:
        if not store:
            raise SystemExit(f"No saved results at {data_path}. Run without --plot-only first.")
        plot(store, fig_path)
        print(f"Redrew {fig_path} from {data_path}")
        if args.with_bias:
            plot(store, out / "fig_theorem2_3panel.png", with_bias=True)
            print(f"Drew {out / 'fig_theorem2_3panel.png'} (three panels, with the bias)")
        return

    t0 = time.time()
    # (b) compares theta_tilde with theta* on the same large draw, so that sampling noise cancels
    x, p, c, a_sev, theta_star_b = population(args.big, args.seed)
    # (a) and (c) need theta* with little sampling noise, from a larger independent draw
    theta_star = theta_star_b
    if "a" in args.parts or "c" in args.parts:
        rng_ref = np.random.default_rng(args.seed + 10)
        p_r, c_r = true_scores(draw_x(args.ref, rng_ref))
        theta_star = best_weight(p_r, p_r * c_r, a_sev)
        del p_r, c_r
    print(f"alpha_cov = {A_COV}, alpha_sev = {a_sev:.5f} ({RATIO} S_cc), "
          f"theta* = {theta_star:.4f} (reference), {theta_star_b:.4f} (draw for (b)) "
          f"({time.time() - t0:.0f} s)\n", flush=True)

    if "a" in args.parts:
        sizes, err = part_a(theta_star, a_sev, args.reps_a, args.seed)
        store["a_sizes"], store["a_err"] = sizes, err
        mean = err.mean(axis=1)
        se = err.std(axis=1, ddof=1) / np.sqrt(err.shape[1])
        print("\n(a) n_U, mean |theta_hat - theta*|, 2 SE, mean x sqrt(n_U)")
        for n_u, m_, s_ in zip(sizes, mean, se):
            print(f"  {n_u:7d}  {m_:.5f}  {2 * s_:.5f}  {m_ * np.sqrt(n_u):.3f}")
        slope = np.polyfit(np.log(sizes), np.log(mean), 1)[0]
        print(f"  log-log slope: {slope:.2f}  (Theorem 2(a): -0.5)\n")

    if "b" in args.parts:
        rows = part_b(x, p, c, theta_star_b, a_sev)
        store["b_kind"] = np.array([r_[0] for r_ in rows])
        store["b_dist"] = np.array([r_[1] for r_ in rows])
        store["b_e"] = np.array([r_[2] for r_ in rows])
        store["b_bias"] = np.array([abs(r_[3] - theta_star_b) for r_ in rows])
        keep = store["b_bias"] > 0
        slope = np.polyfit(np.log(store["b_e"][keep]), np.log(store["b_bias"][keep]), 1)[0]
        print("\n(b) distortion, e, theta_tilde, |theta_tilde - theta*|")
        for kind, m_, e_, t_ in rows:
            print(f"  {kind:15s} {m_:.2f}  {e_:.5f}  {t_:.4f}  {abs(t_ - theta_star_b):.4f}")
        print(f"  log-log slope over all points: {slope:.2f}  (bound: 0.5)\n")

    if "c" in args.parts:
        sizes, res, r_star = part_c(theta_star, a_sev, args.reps_c, args.eval, args.seed)
        store["c_sizes"] = sizes
        for k in ["cal", "wt", "full", "cov", "sev"]:
            store["c_" + k] = res[k]
        cal = 100 * res["cal"].mean(axis=1)
        wt = 100 * res["wt"].mean(axis=1)
        full = 100 * res["full"].mean(axis=1)
        sc = full - cal - wt
        se_full = 100 * res["full"].std(axis=1, ddof=1) / np.sqrt(res["full"].shape[1])
        print(f"\n(c) optimal share sent to review R* = {100 * r_star:.2f}%")
        print("  n        calibration  weight   score    pipeline (2 SE)   coverage (<= 0.10)  severity (<= alpha_sev)")
        for i, n in enumerate(sizes):
            print(f"  {n:7d}  {cal[i]:10.3f}  {wt[i]:7.3f}  {sc[i]:7.3f}  {full[i]:8.3f} ({2 * se_full[i]:.3f})"
                  f"  {res['cov'][i].mean():12.4f}  {res['sev'][i].mean():14.6f}")
        print(f"  alpha_sev = {a_sev:.6f}. Parts can be negative in a single size because of noise.\n")

    np.savez(data_path, **store)
    plot(store, fig_path)
    if args.with_bias:
        plot(store, out / "fig_theorem2_3panel.png", with_bias=True)
    print(f"Saved {fig_path} and {data_path}  (total {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- Theorem 2(b) constant
def c0(args):
    """Constant C_0 and range e_max of Theorem 2(b), computed from the true scores (Discussion)."""
    t0 = time.time()

    rng = np.random.default_rng(args.seed)
    p, c = true_scores(draw_x(args.big, rng))
    pc = p * c
    n = args.big
    a1 = A_COV * p.mean()
    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, a1, side="right"))
    a_sev = RATIO * np.cumsum(pc[o])[k - 1] / n

    # theta*, the cut-off t*, and the prices lambda* = ((1 - theta*) / t*, theta* B / t*) up to the scale of h_theta
    theta_star, k_star = best_weight_value(p, pc, a_sev)
    h = (1 - theta_star) * p + theta_star * pc
    t_star = np.sort(h)[k_star - 1]                                      # largest score passed
    lam_star = np.array([(1 - theta_star) / t_star, theta_star / (B * t_star)])
    print(f"alpha_sev = {a_sev:.5f}, theta* = {theta_star:.4f}, t* = {t_star:.5f}, "
          f"lambda* = ({lam_star[0]:.3f}, {lam_star[1]:.3f})  ({time.time() - t0:.0f} s)", flush=True)

    # mu and L_theta on the ball of radius s_0 around lambda*
    Z = np.column_stack([p, pc])
    norm_l = np.linalg.norm(lam_star)
    s0 = args.s0 * norm_l
    hstep = 0.01 * norm_l
    pts = [lam_star]
    for r in (0.5 * s0, s0):
        for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            pts.append(lam_star + r * np.array([np.cos(ang), np.sin(ang)]))
    eig_min, eig_star = np.inf, None
    for q in pts:
        ev = np.linalg.eigvalsh(hessian(q, Z, hstep))
        eig_min = min(eig_min, ev[0])
        if eig_star is None:
            eig_star = ev
    mu = eig_min
    L_theta = max(grad_theta_norm(q) for q in pts)
    linf = np.max(np.abs(lam_star)) + s0
    C0 = 2 * L_theta * np.sqrt(linf / mu)
    e_max = mu * s0 ** 2 / (4 * linf)
    print(f"eigenvalues of G at lambda*: {eig_star[0]:.4f}, {eig_star[1]:.4f}")
    print(f"s_0 = {s0:.3f} ({args.s0:.0%} of ||lambda*||), mu = {mu:.4f}, L_theta = {L_theta:.4f}")
    print(f"C_0 = {C0:.3f}, e_max = {e_max:.6f}  ({time.time() - t0:.0f} s)\n", flush=True)

    # distorted scores, as in Figure 2(b). theta_tilde on the same draw as theta*
    rows = []
    for m in [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]:
        ch = np.clip(m * c, 0, 1)
        th, _ = best_weight_value(p, p * ch, a_sev)
        rows.append(("c", m, score_error(p, c, p, ch), abs(th - theta_star)))
    for s in [0.80, 0.85, 0.90, 0.95, 0.98, 1.02, 1.05, 1.10, 1.15, 1.20]:
        ph = sigmoid(s * logit(p))
        th, _ = best_weight_value(ph, ph * c, a_sev)
        rows.append(("p", s, score_error(p, c, ph, c), abs(th - theta_star)))
    print(f"{'score':>5s} {'distortion':>10s} {'e':>9s} {'bias':>8s} {'bound':>8s} {'e < e_max':>9s} {'below':>6s}")
    for kind, d, e, b in rows:
        bnd = C0 * np.sqrt(e)
        print(f"{kind:>5s} {d:10.2f} {e:9.5f} {b:8.4f} {bnd:8.4f} {str(e < e_max):>9s} {str(b <= bnd):>6s}")
    inside = [r_ for r_ in rows if r_[2] < e_max]
    print(f"\npoints with e < e_max: {len(inside)} of {len(rows)}, all below the bound: "
          f"{all(r_[3] <= C0 * np.sqrt(r_[2]) for r_ in inside)}")
    print(f"largest bias / bound among them: "
          f"{max((r_[3] / (C0 * np.sqrt(r_[2])) for r_ in inside if r_[3] > 0), default=float('nan')):.3f}")
    print(f"(total {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- command line
def cli():
    ap = argparse.ArgumentParser(description="Simulation study (Sections 3 and 5, Appendix C)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("fig1a", help="Figure 1(a)")
    a.add_argument("--ratio", type=float, default=0.5, help="alpha_sev as a multiple of S_cc")
    a.add_argument("--n", type=int, default=400_000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-diagonal", action="store_true", help="do not draw the diagonal pc = p")

    b = sub.add_parser("fig1b", help="Figure 1(b)")
    b.add_argument("--n", type=int, default=400_000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--levels", type=int, default=80)
    b.add_argument("--step", type=float, default=0.001, help="step of the grid of theta")

    c = sub.add_parser("fig2", help="Figure 2 and the bias numbers of Section 5")
    c.add_argument("--parts", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"])
    c.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved results")
    c.add_argument("--reps-a", type=int, default=100)
    c.add_argument("--reps-c", type=int, default=20)
    c.add_argument("--big", type=int, default=1_000_000, help="size of the large sample for S_cc and part b")
    c.add_argument("--ref", type=int, default=4_000_000, help="size of the sample for the reference theta*")
    c.add_argument("--eval", type=int, default=1_000_000, help="size of the evaluation sample in part c")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--with-bias", action="store_true",
                   help="also draw fig_theorem2_3panel.png with the bias of part b as the middle panel")

    d = sub.add_parser("c0", help="C_0 and e_max of Theorem 2(b)")
    d.add_argument("--big", type=int, default=1_000_000)
    d.add_argument("--s0", type=float, default=0.1, help="radius s_0 as a fraction of ||lambda*||")
    d.add_argument("--seed", type=int, default=0)

    for p_ in (a, b, c, d):
        p_.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"fig1a": fig1a, "fig1b": fig1b, "fig2": fig2, "c0": c0}[args.cmd](args)


if __name__ == "__main__":
    cli()