# #!/usr/bin/env python3
# """
# Construit un tableau LaTeX à partir de fichiers de résultats MRR.

# Pour chaque TKG et chaque configuration (occ, min) :
#   - #domaines masqués        = n_masked
#   - #relations masquées dans les domaines atteints = n_masked_relations
#   - MRR
#   - % domaines atteints      = 100 * (n_masked - n_unreached) / n_masked
#   - % relations atteintes    = 100 * (n_masked_relations - n_missing_relations) / n_masked_relations

# Usage :
#   python mrr_to_latex.py MRR_Results_*.txt
#   python mrr_to_latex.py MRR_Results_*.txt --names Q6256=Country Q82955=Politician -o table.tex
# """

# import argparse
# import ast
# import re
# import sys
# from collections import defaultdict

# # Ligne de résultat : "<chemin>.json: {...}"
# LINE_RE = re.compile(r"^(?P<path>\S+\.json):\s*(?P<dict>\{.*\})\s*$")
# TKG_RE = re.compile(r"/(Q\d+)/")
# OCC_RE = re.compile(r"occ_([0-9p]+)")
# MIN_RE = re.compile(r"min_(\d+)")


# def parse_occ(s: str) -> float:
#     return float(s.replace("p", "."))


# def parse_file(path: str):
#     rows = []
#     with open(path, encoding="utf-8") as f:
#         for line in f:
#             m = LINE_RE.match(line.strip())
#             if not m:
#                 continue  # lignes "Processing ..." et autres
#             p = m.group("path")
#             stats = ast.literal_eval(m.group("dict"))
#             tkg = TKG_RE.search(p)
#             occ = OCC_RE.search(p)
#             mn = MIN_RE.search(p)
#             if not (tkg and occ and mn):
#                 print(f"[warn] chemin non reconnu, ignoré : {p}", file=sys.stderr)
#                 continue
#             rows.append({
#                 "tkg": tkg.group(1),
#                 "occ": parse_occ(occ.group(1)),
#                 "min": int(mn.group(1)),
#                 **stats,
#             })
#     return rows


# def pct(num, den):
#     return 100.0 * num / den if den else float("nan")


# def fmt_int(n: int) -> str:
#     return f"{n:,}".replace(",", "\\,")  # séparateur de milliers fin


# def build_table(rows, names, caption, label, mrr_digits):
#     by_tkg = defaultdict(list)
#     for r in rows:
#         by_tkg[r["tkg"]].append(r)

#     out = []
#     out.append(r"\begin{table}[t]")
#     out.append(r"\centering")
#     out.append(r"\small")
#     out.append(r"\begin{tabular}{llrrrrrr}")
#     out.append(r"\toprule")
#     out.append(r"TKG & $\theta_{occ}$ & $min$ & \#Dom. masqués & \#Rel. masquées "
#                r"& MRR & \% Dom. atteints & \% Rel. atteintes \\")
#     out.append(r"\midrule")

#     tkgs = list(by_tkg)
#     for i, tkg in enumerate(tkgs):
#         rs = sorted(by_tkg[tkg], key=lambda r: (r["occ"], r["min"]))
#         name = names.get(tkg, tkg)
#         for j, r in enumerate(rs):
#             dom = pct(r["n_masked"] - r["n_unreached"], r["n_masked"])
#             rel = pct(r["n_masked_relations"] - r["n_missing_relations"],
#                       r["n_masked_relations"])
#             first = rf"\multirow{{{len(rs)}}}{{*}}{{{name}}}" if j == 0 else ""
#             out.append(
#                 f"{first} & {r['occ']:g} & {r['min']} & {fmt_int(r['n_masked'])} & "
#                 f"{fmt_int(r['n_masked_relations'])} & {r['mrr']:.{mrr_digits}f} & "
#                 f"{dom:.2f} & {rel:.2f} \\\\"
#             )
#         if i < len(tkgs) - 1:
#             out.append(r"\midrule")

#     out.append(r"\bottomrule")
#     out.append(r"\end{tabular}")
#     out.append(rf"\caption{{{caption}}}")
#     out.append(rf"\label{{{label}}}")
#     out.append(r"\end{table}")
#     return "\n".join(out)


# def main():
#     ap = argparse.ArgumentParser(description=__doc__,
#                                  formatter_class=argparse.RawDescriptionHelpFormatter)
#     ap.add_argument("files", nargs="+", help="fichiers de résultats MRR")
#     ap.add_argument("--names", nargs="*", default=[],
#                     help="renommage des TKG, ex. Q6256=Country")
#     ap.add_argument("-o", "--output", help="fichier .tex de sortie (défaut : stdout)")
#     ap.add_argument("--caption", default="Résultats de complétion par TKG.")
#     ap.add_argument("--label", default="tab:mrr")
#     ap.add_argument("--mrr-digits", type=int, default=3)
#     args = ap.parse_args()

#     names = dict(n.split("=", 1) for n in args.names)
#     rows = [r for f in args.files for r in parse_file(f)]
#     if not rows:
#         sys.exit("Aucune ligne de résultat trouvée.")

#     table = build_table(rows, names, args.caption, args.label, args.mrr_digits)
#     header = "% Nécessite : \\usepackage{booktabs, multirow}\n"
#     if args.output:
#         with open(args.output, "w", encoding="utf-8") as f:
#             f.write(header + table + "\n")
#     else:
#         print(header + table)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Construit un tableau LaTeX à partir de fichiers de résultats MRR.

Seuls les runs à tau_c = 0 (occ_0) sont conservés : tau_c ne fait plus partie
de la méthode, seul tau (min) est un paramètre.

Pour chaque TKG et chaque valeur de tau (min) :
  - #domaines masqués        = n_masked
  - #relations masquées dans les domaines atteints = n_masked_relations
  - MRR
  - % domaines atteints      = 100 * (n_masked - n_unreached) / n_masked
  - % relations atteintes    = 100 * (n_masked_relations - n_missing_relations) / n_masked_relations

Usage :
  python mrr_to_latex.py MRR_Results_*.txt
  python mrr_to_latex.py MRR_Results_*.txt --names Q6256=Country Q82955=Politician -o table.tex
"""

import argparse
import ast
import re
import sys
from collections import defaultdict

TAU_C = 0.0  # seule valeur de tau_c conservée

# Ligne de résultat : "<chemin>.json: {...}"
LINE_RE = re.compile(r"^(?P<path>\S+\.json):\s*(?P<dict>\{.*\})\s*$")
TKG_RE = re.compile(r"/(Q\d+)/")
OCC_RE = re.compile(r"occ_([0-9p]+)")
MIN_RE = re.compile(r"min_(\d+)")


def parse_occ(s: str) -> float:
    return float(s.replace("p", "."))


def parse_file(path: str):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue  # lignes "Processing ..." et autres
            p = m.group("path")
            stats = ast.literal_eval(m.group("dict"))
            tkg = TKG_RE.search(p)
            occ = OCC_RE.search(p)
            mn = MIN_RE.search(p)
            if not (tkg and occ and mn):
                print(f"[warn] chemin non reconnu, ignoré : {p}", file=sys.stderr)
                continue
            if abs(parse_occ(occ.group(1)) - TAU_C) > 1e-9:
                continue  # run avec tau_c > 0 : ignoré
            rows.append({
                "tkg": tkg.group(1),
                "min": int(mn.group(1)),
                **stats,
            })
    return rows


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", "\\,")  # séparateur de milliers fin


def build_table(rows, names, caption, label, mrr_digits):
    by_tkg = defaultdict(list)
    for r in rows:
        by_tkg[r["tkg"]].append(r)

    out = []
    out.append(r"\begin{table}[t]")
    out.append(r"\centering")
    out.append(r"\small")
    out.append(r"\begin{tabular}{lrrrrrr}")
    out.append(r"\toprule")
    out.append(r"TKG & $\tau$ & \#Masked domains & \#Masked relations "
               r"& MRR & \% Reached domains & \% Reached relations \\")
    out.append(r"\midrule")

    tkgs = list(by_tkg)
    for i, tkg in enumerate(tkgs):
        rs = sorted(by_tkg[tkg], key=lambda r: r["min"])
        name = names.get(tkg, tkg)
        for j, r in enumerate(rs):
            dom = pct(r["n_masked"] - r["n_unreached"], r["n_masked"])
            rel = pct(r["n_masked_relations"] - r["n_missing_relations"],
                      r["n_masked_relations"])
            first = rf"\multirow{{{len(rs)}}}{{*}}{{{name}}}" if j == 0 else ""
            out.append(
                f"{first} & {r['min']} & {fmt_int(r['n_masked'])} & "
                f"{fmt_int(r['n_masked_relations'])} & {r['mrr']:.{mrr_digits}f} & "
                f"{dom:.2f} & {rel:.2f} \\\\"
            )
        if i < len(tkgs) - 1:
            out.append(r"\midrule")

    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(rf"\caption{{{caption}}}")
    out.append(rf"\label{{{label}}}")
    out.append(r"\end{table}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="fichiers de résultats MRR")
    ap.add_argument("--names", nargs="*", default=[],
                    help="renommage des TKG, ex. Q6256=Country")
    ap.add_argument("-o", "--output", help="fichier .tex de sortie (défaut : stdout)")
    ap.add_argument("--caption", default="Completion results by TKG.")
    ap.add_argument("--label", default="tab:mrr")
    ap.add_argument("--mrr-digits", type=int, default=3)
    args = ap.parse_args()

    names = dict(n.split("=", 1) for n in args.names)
    rows = [r for f in args.files for r in parse_file(f)]
    if not rows:
        sys.exit(f"Aucune ligne de résultat trouvée pour tau_c = {TAU_C:g}.")

    table = build_table(rows, names, args.caption, args.label, args.mrr_digits)
    header = "% Nécessite : \\usepackage{booktabs, multirow}\n"
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(header + table + "\n")
    else:
        print(header + table)


if __name__ == "__main__":
    main()
