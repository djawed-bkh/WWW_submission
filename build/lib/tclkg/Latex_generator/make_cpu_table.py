# #!/usr/bin/env python3
# """Génère le tableau LaTeX des temps CPU (oracle / propagation) selon tau_c et tau,
# à partir des fichiers tempscpu_<QID>.txt (TSV)."""
# import csv
# import glob
# import os
# import re
# import sys

# # Usage : python3 make_cpu_table.py [dossier_des_fichiers] [sortie.tex]
# IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
# OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "tab_cpu.tex"
# SCALE = 0.8
# DECIMALS = 1

# # Ordre et noms des datasets dans le tableau
# DATASETS = [("Q6256", "Country"), ("Q215380", "Musical group"), ("Q82955", "Politician")]

# COL_ORACLE = "after_oracle_cpu_s"
# COL_PROP = "after_propagation_cpu_s"


# def fmt_time(s):
#     s = s.strip()
#     if s.upper() == "TO":
#         return "TO"
#     if s == "" or s.upper() in {"NA", "NAN"}:
#         return "--"
#     return f"{float(s):.{DECIMALS}f}"


# def fmt_tau_c(x):
#     return str(int(x)) if x == int(x) else f"{x:g}"


# # data[qid][(tau_c, tau)] = (oracle, propagation) ; timeout[qid] = timeout_s
# data, timeout = {}, {}
# for path in glob.glob(os.path.join(IN_DIR, "*tempscpu_Q*.txt")):
#     m = re.search(r"tempscpu_(Q\d+)\.txt$", path)
#     if not m:
#         continue
#     qid = m.group(1)
#     with open(path, newline="") as f:
#         for row in csv.DictReader(f, delimiter="\t"):
#             key = (float(row["tau_c"]), int(row["tau"]))
#             data.setdefault(qid, {})[key] = (fmt_time(row[COL_ORACLE]), fmt_time(row[COL_PROP]))
#             timeout[qid] = int(row["timeout_s"])

# missing = [q for q, _ in DATASETS if q not in data]
# if missing:
#     sys.exit(f"Fichiers manquants pour : {', '.join(missing)}")

# tau_cs = sorted({tc for q in data.values() for (tc, _) in q})
# taus = {qid: sorted({t for (_, t) in data[qid]}) for qid, _ in DATASETS}
# n_rows = max(len(t) for t in taus.values())
# to_hours = sorted({timeout[q] for q, _ in DATASETS})
# to_txt = " / ".join(f"{t // 3600}\\,h" if t % 3600 == 0 else f"{t}\\,s" for t in to_hours)

# L = []
# L.append(r"\begin{table}[t]")
# L.append(r"\centering")
# L.append(r"\setlength{\tabcolsep}{2.5pt}")
# L.append(rf"\scalebox{{{SCALE}}}{{%")
# L.append(r"\begin{tabular}{@{}c " + " ".join(["rrr"] * len(DATASETS)) + r"@{}}")
# L.append(r"\toprule")
# L.append(" & " + "\n & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in DATASETS) + r"\\")
# cmid = []
# for i in range(len(DATASETS)):
#     a, b = 2 + 3 * i, 4 + 3 * i
#     trim = "l" if i == len(DATASETS) - 1 else "lr"
#     cmid.append(rf"\cmidrule({trim}){{{a}-{b}}}")
# L.append("".join(cmid))
# L.append(r"$\tau_c$ & " + " & ".join([r"$\tau$ & Orac. & Prop."] * len(DATASETS)) + r"\\")

# for tc in tau_cs:
#     L.append(r"\midrule")
#     for r in range(n_rows):
#         cells = [rf"\multirow{{{n_rows}}}{{*}}{{{fmt_tau_c(tc)}}}" if r == 0 else ""]
#         for qid, _ in DATASETS:
#             ts = taus[qid]
#             if r >= len(ts):
#                 cells += ["", "", ""]
#                 continue
#             orac, prop = data[qid].get((tc, ts[r]), ("--", "--"))
#             cells += [str(ts[r]), orac, prop]
#         L.append(" & ".join(cells) + r"\\")

# L.append(r"\bottomrule")
# L.append(r"\end{tabular}%")
# L.append("}")
# L.append(r"\caption{\small CPU time (s) w.r.t.\ $\tau_c$ and $\tau$, measured after the oracle step (Orac.) "
#          r"and after propagation (Prop.). TO: timeout (" + to_txt + r"); ``--'': run not available.}")
# L.append(r"\label{tab:cpu}")
# L.append(r"\end{table}")

# with open(OUTPUT, "w") as f:
#     f.write("\n".join(L) + "\n")
# print(f"Tableau écrit dans {OUTPUT}")

#!/usr/bin/env python3
"""Génère le tableau LaTeX des temps CPU (oracle / propagation) selon tau,
à partir des fichiers tempscpu_<QID>.txt (TSV).
Seuls les runs à tau_c = 0 sont conservés : tau_c ne fait plus partie de la méthode."""
import csv
import glob
import os
import re
import sys

# Usage : python3 make_cpu_table.py [dossier_des_fichiers] [sortie.tex]
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "tab_cpu.tex"
SCALE = 0.8
DECIMALS = 1
TAU_C = 0.0  # seule valeur de tau_c conservée

# Ordre et noms des datasets dans le tableau
DATASETS = [("Q6256", "Country"), ("Q215380", "Musical group"), ("Q82955", "Politician")]

COL_ORACLE = "after_oracle_cpu_s"
COL_PROP = "after_propagation_cpu_s"


def fmt_time(s):
    s = s.strip()
    if s.upper() == "TO":
        return "TO"
    if s == "" or s.upper() in {"NA", "NAN"}:
        return "--"
    return f"{float(s):.{DECIMALS}f}"


# data[qid][tau] = (oracle, propagation) ; timeout[qid] = timeout_s
data, timeout = {}, {}
for path in glob.glob(os.path.join(IN_DIR, "*tempscpu_Q*.txt")):
    m = re.search(r"tempscpu_(Q\d+)\.txt$", path)
    if not m:
        continue
    qid = m.group(1)
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if abs(float(row["tau_c"]) - TAU_C) > 1e-9:
                continue  # run avec tau_c > 0 : ignoré
            data.setdefault(qid, {})[int(row["tau"])] = (fmt_time(row[COL_ORACLE]),
                                                         fmt_time(row[COL_PROP]))
            timeout[qid] = int(row["timeout_s"])

missing = [q for q, _ in DATASETS if q not in data]
if missing:
    sys.exit(f"Aucun run à tau_c = {TAU_C:g} pour : {', '.join(missing)}")

taus = {qid: sorted(data[qid]) for qid, _ in DATASETS}
n_rows = max(len(t) for t in taus.values())
to_hours = sorted({timeout[q] for q, _ in DATASETS})
to_txt = " / ".join(f"{t // 3600}\\,h" if t % 3600 == 0 else f"{t}\\,s" for t in to_hours)

L = []
L.append(r"\begin{table}[t]")
L.append(r"\centering")
L.append(r"\setlength{\tabcolsep}{2.5pt}")
L.append(rf"\scalebox{{{SCALE}}}{{%")
L.append(r"\begin{tabular}{@{}" + " ".join(["rrr"] * len(DATASETS)) + r"@{}}")
L.append(r"\toprule")
L.append("\n & ".join(rf"\multicolumn{{3}}{{c}}{{{name}}}" for _, name in DATASETS) + r"\\")
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
L.append(" & ".join([r"$\tau$ & Orac. & Prop."] * len(DATASETS)) + r"\\")
L.append(r"\midrule")

for r in range(n_rows):
    cells = []
    for qid, _ in DATASETS:
        ts = taus[qid]
        if r >= len(ts):
            cells += ["", "", ""]
            continue
        orac, prop = data[qid][ts[r]]
        cells += [str(ts[r]), orac, prop]
    L.append(" & ".join(cells) + r"\\")

L.append(r"\bottomrule")
L.append(r"\end{tabular}%")
L.append("}")
L.append(r"\caption{\small CPU time (s) w.r.t.\ $\tau$, measured after the oracle step (Orac.) "
         r"and after propagation (Prop.). TO: timeout (" + to_txt + r"); ``--'': run not available.}")
L.append(r"\label{tab:cpu}")
L.append(r"\end{table}")

with open(OUTPUT, "w") as f:
    f.write("\n".join(L) + "\n")
print(f"Tableau écrit dans {OUTPUT}")