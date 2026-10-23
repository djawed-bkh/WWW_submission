# #!/usr/bin/env python3
# """Génère le tableau LaTeX RQ1 (Obs./Inf./Unk. selon tau_c et tau) à partir du JSON de résultats."""
# import json
# import re
# import sys

# INPUT = sys.argv[1] if len(sys.argv) > 1 else "RQ1.txt"
# OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "tab_rq1.tex"
# SCALE = 0.8

# # Ordre et noms des datasets dans le tableau
# DATASETS = [("Q6256", "Country"), ("Q215380", "Musical group"), ("Q82955", "Politician")]

# KEY_RE = re.compile(r"^(Q\d+)/.*?occ_([0-9p]+)_min_(\d+)_")


# def parse_occ(s):
#     return float(s.replace("p", "."))


# def fmt_pct(x):
#     if x is None:
#         return "--"
#     if x == 0:
#         return "0.00"
#     if x < 0.01:
#         return "$<$0.01"
#     return f"{x:.2f}"


# def fmt_tau_c(x):
#     return str(int(x)) if x == int(x) else f"{x:g}"


# def fmt_int(n):
#     return f"{n:,}".replace(",", "{,}")


# with open(INPUT) as f:
#     raw = json.load(f)

# # data[qid][(tau_c, tau)] = entrée ; meta[qid] = (total_domains, observed_percentage)
# data, meta = {}, {}
# for key, v in raw.items():
#     m = KEY_RE.match(key)
#     if not m:
#         print(f"Clé ignorée : {key}", file=sys.stderr)
#         continue
#     qid, occ, tau = m.group(1), parse_occ(m.group(2)), int(m.group(3))
#     data.setdefault(qid, {})[(occ, tau)] = v
#     meta[qid] = (v["total_domains"], v["observed_percentage"])

# tau_cs = sorted({tc for q in data.values() for (tc, _) in q})
# taus = {qid: sorted({t for (_, t) in data.get(qid, {})}) for qid, _ in DATASETS}
# n_rows = max(len(t) for t in taus.values())

# L = []
# L.append(r"\begin{table}[t]")
# L.append(r"\centering")
# L.append(r"\setlength{\tabcolsep}{2.5pt}")
# L.append(rf"\scalebox{{{SCALE}}}{{%")
# L.append(r"\begin{tabular}{@{}c " + " ".join(["rrr"] * len(DATASETS)) + r"@{}}")
# L.append(r"\toprule")
# L.append(" & " + "\n & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in DATASETS) + r"\\")
# L.append(" & " + "\n & ".join(rf"\multicolumn{{3}}{{c}}{{{fmt_int(meta[q][0])} dom.}}" for q, _ in DATASETS) + r"\\")
# L.append(" & " + "\n & ".join(rf"\multicolumn{{3}}{{c}}{{Obs.\ {meta[q][1]:.2f}\%}}" for q, _ in DATASETS) + r"\\")
# cmid = []
# for i in range(len(DATASETS)):
#     a, b = 2 + 3 * i, 4 + 3 * i
#     trim = "l" if i == len(DATASETS) - 1 else "lr"
#     cmid.append(rf"\cmidrule({trim}){{{a}-{b}}}")
# L.append("".join(cmid))
# L.append(r"$\tau_c$ & " + " & ".join([r"$\tau$ & Inf. & Unk."] * len(DATASETS)) + r"\\")

# for tc in tau_cs:
#     L.append(r"\midrule")
#     for r in range(n_rows):
#         first = rf"\multirow{{{n_rows}}}{{*}}{{{fmt_tau_c(tc)}}}" if r == 0 else ""
#         cells = [first]
#         for qid, _ in DATASETS:
#             ts = taus[qid]
#             if r >= len(ts):
#                 cells += ["", "", ""]
#                 continue
#             e = data[qid].get((tc, ts[r]))
#             inf = e["inferred_percentage"] if e else None
#             unk = e["unknown_percentage"] if e else None
#             cells += [str(ts[r]), fmt_pct(inf), fmt_pct(unk)]
#         L.append(" & ".join(cells) + r"\\")

# L.append(r"\bottomrule")
# L.append(r"\end{tabular}%")
# L.append("}")
# L.append(r"\caption{\small Proportion (\%) of observed (Obs.), inferred (Inf.) and unknown (Unk.), "
#          r"w.r.t.\ $\tau_c$ and $\tau$. The observed proportion (Obs.), independent of $(\tau_c,\tau)$, "
#          r"is given under each dataset name. ``--'': run not available; $<$0.01: non-zero but below 0.01\%.}")
# L.append(r"\label{tab:rq1}")
# L.append(r"\end{table}")

# with open(OUTPUT, "w") as f:
#     f.write("\n".join(L) + "\n")
# print(f"Tableau écrit dans {OUTPUT}")
#!/usr/bin/env python3
"""Génère le tableau LaTeX RQ1 (Obs./Inf./Unk. selon tau) à partir du JSON de résultats.
Seuls les runs à tau_c = 0 (occ_0) sont conservés : tau_c ne fait plus partie de la méthode."""
import json
import re
import sys

INPUT = sys.argv[1] if len(sys.argv) > 1 else "RQ1.txt"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "tab_rq1.tex"
SCALE = 0.8
TAU_C = 0.0  # seule valeur de tau_c conservée

# Ordre et noms des datasets dans le tableau
DATASETS = [("Q6256", "Country"), ("Q215380", "Musical group"), ("Q82955", "Politician")]

KEY_RE = re.compile(r"^(Q\d+)/.*?occ_([0-9p]+)_min_(\d+)_")


def parse_occ(s):
    return float(s.replace("p", "."))


def fmt_pct(x):
    if x is None:
        return "--"
    if x == 0:
        return "0.00"
    if x < 0.01:
        return "$<$0.01"
    return f"{x:.2f}"


def fmt_int(n):
    return f"{n:,}".replace(",", "{,}")


with open(INPUT) as f:
    raw = json.load(f)

# data[qid][tau] = entrée ; meta[qid] = (total_domains, observed_percentage)
data, meta = {}, {}
for key, v in raw.items():
    m = KEY_RE.match(key)
    if not m:
        print(f"Clé ignorée : {key}", file=sys.stderr)
        continue
    qid, occ, tau = m.group(1), parse_occ(m.group(2)), int(m.group(3))
    if abs(occ - TAU_C) > 1e-9:
        continue  # run avec tau_c > 0 : ignoré
    data.setdefault(qid, {})[tau] = v
    meta[qid] = (v["total_domains"], v["observed_percentage"])

missing = [q for q, _ in DATASETS if q not in data]
if missing:
    sys.exit(f"Aucun run à tau_c = {TAU_C:g} pour : {', '.join(missing)}")

taus = {qid: sorted(data[qid]) for qid, _ in DATASETS}
n_rows = max(len(t) for t in taus.values())

L = []
L.append(r"\begin{table}[t]")
L.append(r"\centering")
L.append(r"\setlength{\tabcolsep}{2.5pt}")
L.append(rf"\scalebox{{{SCALE}}}{{%")
L.append(r"\begin{tabular}{@{}" + " ".join(["rrr"] * len(DATASETS)) + r"@{}}")
L.append(r"\toprule")
L.append("\n & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in DATASETS) + r"\\")
L.append("\n & ".join(rf"\multicolumn{{3}}{{c}}{{{fmt_int(meta[q][0])} dom.}}" for q, _ in DATASETS) + r"\\")
L.append("\n & ".join(rf"\multicolumn{{3}}{{c}}{{Obs.\ {meta[q][1]:.2f}\%}}" for q, _ in DATASETS) + r"\\")
cmid = []
for i in range(len(DATASETS)):
    a, b = 1 + 3 * i, 3 + 3 * i
    if i == 0:
        trim = "r"
    elif i == len(DATASETS) - 1:
        trim = "l"
    else:
        trim = "lr"
    cmid.append(rf"\cmidrule({trim}){{{a}-{b}}}")
L.append("".join(cmid))
L.append(" & ".join([r"$\tau$ & Inf. & Unk."] * len(DATASETS)) + r"\\")
L.append(r"\midrule")

for r in range(n_rows):
    cells = []
    for qid, _ in DATASETS:
        ts = taus[qid]
        if r >= len(ts):
            cells += ["", "", ""]
            continue
        e = data[qid][ts[r]]
        cells += [str(ts[r]), fmt_pct(e["inferred_percentage"]), fmt_pct(e["unknown_percentage"])]
    L.append(" & ".join(cells) + r"\\")

L.append(r"\bottomrule")
L.append(r"\end{tabular}%")
L.append("}")
L.append(r"\caption{\small Proportion (\%) of observed (Obs.), inferred (Inf.) and unknown (Unk.) "
         r"domains w.r.t.\ $\tau$. The observed proportion (Obs.), independent of $\tau$, "
         r"is given under each dataset name. $<$0.01: non-zero but below 0.01\%.}")
L.append(r"\label{tab:rq1}")
L.append(r"\end{table}")

with open(OUTPUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Tableau écrit dans {OUTPUT}")