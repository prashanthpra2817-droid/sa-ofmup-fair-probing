"""Build tables/*.tex and numbers.tex from results/*.csv (run after the experiments)."""
import csv
import os
import sys

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
T = sys.argv[2] if len(sys.argv) > 2 else "1500"
SEEDS = sys.argv[3] if len(sys.argv) > 3 else "3"
os.makedirs("tables", exist_ok=True)
nums = [f"\\newcommand{{\\TT}}{{{T}}}", f"\\newcommand{{\\Seeds}}{{{SEEDS}}}"]

# ---------------- reproduction table ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "fig1_reproduction.csv"))))
panels = sorted(set(r["panel"] for r in rows))
methods = ["Random P+A", "Greedy P+A", "Non-Probing", "OFMUP"]
lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Setting & Random P+A & Greedy P+A & Non-Probing & OFMUP \\", r"\midrule"]
for p in panels:
    d = {r["method"]: (float(r["final_regret_mean"]), float(r["final_regret_std"])) for r in rows if r["panel"] == p}
    env = [r for r in rows if r["panel"] == p][0]
    cells = [f"{d[m][0]:.1f} $\\pm$ {d[m][1]:.1f}" for m in methods]
    lines.append(f"({p}) {env['env'][:4]}., $M$={env['M']}, $A$={env['A']} & " + " & ".join(cells) + r" \\")
    o = d["OFMUP"][0]
    red = {m: 100 * (1 - o / d[m][0]) for m in methods[:3]}
    lines.append(r"\quad OFMUP reduction & " + " & ".join(f"{red[m]:.0f}\\%" for m in methods[:3]) + r" & -- \\")
    if p == "a":
        nums += [f"\\newcommand{{\\RedRandA}}{{{red['Random P+A']:.0f}}}",
                 f"\\newcommand{{\\RedGPA}}{{{red['Greedy P+A']:.0f}}}",
                 f"\\newcommand{{\\RedNPA}}{{{red['Non-Probing']:.0f}}}"]
lines += [r"\bottomrule", r"\end{tabular}"]
open("tables/repro_table.tex", "w").write("\n".join(lines) + "\n")

# ---------------- staleness table ----------------
rows = list(csv.DictReader(open(os.path.join(RES, "staleness_a.csv"))))
rhos = sorted(set(float(r["rho"]) for r in rows))
methods = ["Non-Probing", "OFMUP", "OFMUP+fix", "SA-OFMUP (known rho)", "SA-OFMUP (learned rho)", "SA-OFMUP (learned rho, LCB)"]
short = {"Non-Probing": "Non-Probing", "OFMUP": "OFMUP", "OFMUP+fix": "OFMUP+fix",
         "SA-OFMUP (known rho)": r"SA-OFMUP (known $\rho$)", "SA-OFMUP (learned rho)": r"SA-OFMUP (point estimate)",
         "SA-OFMUP (learned rho, LCB)": r"\textbf{SA-OFMUP (LCB, final)}"}
lines = [r"\begin{tabular}{l" + "r" * len(rhos) + "}", r"\toprule",
         r"Method & " + " & ".join(f"$\\rho={x:g}$" for x in rhos) + r" \\", r"\midrule",
         r"\multicolumn{%d}{l}{\emph{Final cumulative regret (mean $\pm$ s.d.)}} \\" % (len(rhos) + 1)]
get = lambda m, x, k: float([r for r in rows if r["method"] == m and float(r["rho"]) == x][0][k])
for m in methods:
    lines.append(short[m] + " & " + " & ".join(f"{get(m, x, 'final_regret_mean'):.1f} $\\pm$ {get(m, x, 'final_regret_std'):.1f}" for x in rhos) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{%d}{l}{\emph{Arms probed per round}} \\" % (len(rhos) + 1))
for m in methods[1:]:
    lines.append(short[m] + " & " + " & ".join(f"{get(m, x, 'probes_per_round'):.2f}" for x in rhos) + r" \\")
lines.append(r"\midrule")
lines.append(r"\multicolumn{%d}{l}{\emph{Final estimate of $\rho$}} \\" % (len(rhos) + 1))
lines.append(short["SA-OFMUP (learned rho)"] + " & " + " & ".join(f"{get('SA-OFMUP (learned rho)', x, 'final_rho_hat'):.2f}" for x in rhos) + r" \\")
lines.append(r"SA-OFMUP (LCB), point estimate & " + " & ".join(f"{get('SA-OFMUP (learned rho, LCB)', x, 'final_rho_hat'):.2f}" for x in rhos) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open("tables/stale_table.tex", "w").write("\n".join(lines) + "\n")

hi, mid = max(rhos), sorted(rhos)[len(rhos) // 2]
nums += [f"\\newcommand{{\\RhoHatHi}}{{{get('SA-OFMUP (learned rho)', hi, 'final_rho_hat'):.2f}}}",
         f"\\newcommand{{\\RhoHatMid}}{{{get('SA-OFMUP (learned rho)', mid, 'final_rho_hat'):.2f}}}",
         f"\\newcommand{{\\ProbesOFHi}}{{{get('OFMUP', hi, 'probes_per_round'):.2f}}}",
         f"\\newcommand{{\\ProbesSALHi}}{{{get('SA-OFMUP (learned rho)', hi, 'probes_per_round'):.2f}}}",
         f"\\newcommand{{\\RegOFHi}}{{{get('OFMUP', hi, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegSAKHi}}{{{get('SA-OFMUP (known rho)', hi, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegSALHi}}{{{get('SA-OFMUP (learned rho)', hi, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegNPHi}}{{{get('Non-Probing', hi, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegOFMid}}{{{get('OFMUP', mid, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegSALMid}}{{{get('SA-OFMUP (learned rho)', mid, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegLCBMid}}{{{get('SA-OFMUP (learned rho, LCB)', mid, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\RegLCBHi}}{{{get('SA-OFMUP (learned rho, LCB)', hi, 'final_regret_mean'):.1f}}}",
         f"\\newcommand{{\\ProbesLCBHi}}{{{get('SA-OFMUP (learned rho, LCB)', hi, 'probes_per_round'):.2f}}}",
         f"\\newcommand{{\\ProbesLCBMid}}{{{get('SA-OFMUP (learned rho, LCB)', mid, 'probes_per_round'):.2f}}}",
         f"\\newcommand{{\\PointHi}}{{{get('SA-OFMUP (learned rho, LCB)', hi, 'final_rho_hat'):.2f}}}",
         f"\\newcommand{{\\PointMid}}{{{get('SA-OFMUP (learned rho, LCB)', mid, 'final_rho_hat'):.2f}}}"]
open("numbers.tex", "w").write("\n".join(nums) + "\n")
print("wrote tables/repro_table.tex, tables/stale_table.tex, numbers.tex")
