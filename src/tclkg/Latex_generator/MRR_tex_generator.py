#!/usr/bin/env python3
"""
Génère des tableaux LaTeX à partir du fichier de résultats d'ablation
(MRR_avec_comptage_non_atteints.txt).

Une ligne de résultat ressemble à :
  Q6256/reduced_composition/occ_0p3_min_10_timeout_86400/qcn2_Q6256_reduced_composition.json:
      non atteints = 1/100 (cumul 10 epochs) | weighted MRR = 0.9176...

Une ligne ignorée ressemble à :
  Skipping ablation for .../occ_0p9_min_150_.../qcn2_Q215380_....json:
      98 valid observed domains (at least 100 required)

Sortie par défaut : un seul tableau regroupant toutes les classes (bases), avec
  - une ligne par (classe, seuil `min`), les lignes d'une même classe étant
    regroupées visuellement par un \\multirow sur la colonne "Classe",
  - deux colonnes par valeur de `occ` : nombre de domaines non atteints (NA) et MRR pondéré.
Avec --per-class : un tableau séparé par classe (ancien comportement).

Préambule LaTeX requis : \\usepackage{booktabs,multirow}

Usage :
  python make_tex_table.py MRR_avec_comptage_non_atteints.txt -o tableau.tex
  python make_tex_table.py MRR_avec_comptage_non_atteints.txt --bold-best   # MRR max en gras par ligne
  python make_tex_table.py MRR_avec_comptage_non_atteints.txt --per-class   # un tableau par classe
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Noms lisibles des classes Wikidata (optionnel : ajouter/modifier ici).
CLASS_NAMES = {
    "Q6256": "country",
    "Q82955": "politician",
    "Q215380": "musical group",
}

RESULT_RE = re.compile(
    r"^(?P<path>\S+):\s*non atteints\s*=\s*(?P<na>\d+)\s*/\s*(?P<tot>\d+)"
    r".*?weighted MRR\s*=\s*(?P<mrr>[-+\d.eE]+)\s*$"
)
SKIP_RE = re.compile(r"^Skipping ablation for\s+(?P<path>\S+?):\s*(?P<reason>.*)$")
PARAMS_RE = re.compile(r"occ_(?P<occ>\d+p\d+)_min_(?P<min>\d+)_timeout_(?P<timeout>\d+)")
QID_RE = re.compile(r"qcn2_(Q\d+)_")


def parse_path(path):
    """Retourne (qid, occ, min_support, timeout) ou None si le chemin n'est pas reconnu."""
    m = PARAMS_RE.search(path)
    if not m:
        return None
    q = QID_RE.search(path)
    qid = q.group(1) if q else path.split("/")[0]
    occ = float(m.group("occ").replace("p", "."))
    return qid, occ, int(m.group("min")), int(m.group("timeout"))


def parse_file(filename):
    """Retourne data[qid][(min, occ)] = dict(na, tot, mrr) | dict(skipped=True), et l'ensemble des timeouts."""
    data = defaultdict(dict)
    timeouts = set()
    n_unparsed = 0
    for lineno, line in enumerate(Path(filename).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        m = RESULT_RE.match(line)
        if m:
            info = parse_path(m.group("path"))
            if info is None:
                n_unparsed += 1
                print(f"[warn] ligne {lineno}: chemin non reconnu", file=sys.stderr)
                continue
            qid, occ, mn, to = info
            timeouts.add(to)
            key = (mn, occ)
            if key in data[qid]:
                print(f"[warn] doublon pour {qid} occ={occ} min={mn} (ligne {lineno}), écrasé", file=sys.stderr)
            data[qid][key] = {"na": int(m.group("na")), "tot": int(m.group("tot")), "mrr": float(m.group("mrr"))}
            continue
        m = SKIP_RE.match(line)
        if m:
            info = parse_path(m.group("path"))
            if info is None:
                n_unparsed += 1
                continue
            qid, occ, mn, to = info
            timeouts.add(to)
            data[qid][(mn, occ)] = {"skipped": True, "reason": m.group("reason")}
            continue
        n_unparsed += 1
        print(f"[warn] ligne {lineno} ignorée : {line[:80]}", file=sys.stderr)
    return data, timeouts


def fmt_occ(occ):
    return f"{occ:g}"


def build_table(qid, cells, timeouts, bold_best=False, decimals=3):
    occs = sorted({occ for (_, occ) in cells})
    mins = sorted({mn for (mn, _) in cells})
    tots = {c["tot"] for c in cells.values() if "tot" in c}
    tot_txt = f"{tots.pop()}" if len(tots) == 1 else "N"
    has_skip = any(c.get("skipped") for c in cells.values())

    name = CLASS_NAMES.get(qid)
    title = f"{qid} ({name})" if name else qid
    to_txt = ""
    if len(timeouts) == 1:
        to_txt = f" (timeout : {next(iter(timeouts))}\\,s)"

    caption = (
        f"Ablation sur la classe {title}{to_txt}. "
        f"NA : nombre de domaines non atteints sur {tot_txt} (cumul sur 10 epochs) ; "
        f"MRR : MRR pondéré."
    )
    if has_skip:
        caption += " \\og -- \\fg{} : ablation non réalisée (moins de 100 domaines observés valides)."

    col_spec = "r" + "".join("rr" for _ in occs)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{tab:ablation-{qid}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # En-tête : occ
    head1 = [""] + [f"\\multicolumn{{2}}{{c}}{{$\\mathit{{occ}}={fmt_occ(o)}$}}" for o in occs]
    lines.append(" & ".join(head1) + " \\\\")
    lines.append(
        "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(occs)))
    )
    head2 = ["$\\mathit{min}$"] + ["NA", "MRR"] * len(occs)
    lines.append(" & ".join(head2) + " \\\\")
    lines.append("\\midrule")

    for mn in mins:
        # MRR max de la ligne (hors cellules ignorées) pour la mise en gras
        row_mrrs = [cells[(mn, o)]["mrr"] for o in occs if (mn, o) in cells and "mrr" in cells[(mn, o)]]
        best = max(row_mrrs) if row_mrrs else None
        row = [str(mn)]
        for o in occs:
            c = cells.get((mn, o))
            if c is None or c.get("skipped"):
                row.append("\\multicolumn{2}{c}{--}")
                continue
            mrr = f"{c['mrr']:.{decimals}f}"
            if bold_best and best is not None and c["mrr"] == best:
                mrr = f"\\textbf{{{mrr}}}"
            row.append(str(c["na"]))
            row.append(mrr)
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def class_title(qid):
    name = CLASS_NAMES.get(qid)
    return f"{qid} ({name})" if name else qid


def build_combined_table(data, timeouts, bold_best=False, decimals=3, per_class_best=True):
    """Un seul tableau, toutes classes confondues, avec les classes regroupées par \\multirow."""
    qids = sorted(data)
    occs = sorted({occ for cells in data.values() for (_, occ) in cells})
    has_skip = any(c.get("skipped") for cells in data.values() for c in cells.values())
    tots = {c["tot"] for cells in data.values() for c in cells.values() if "tot" in c}
    tot_txt = f"{tots.pop()}" if len(tots) == 1 else "N"

    to_txt = ""
    if len(timeouts) == 1:
        to_txt = f" (timeout : {next(iter(timeouts))}\\,s)"

    caption = (
        f"Ablation sur l'ensemble des classes{to_txt}. "
        f"NA : nombre de domaines non atteints sur {tot_txt} (cumul sur 10 epochs) ; "
        f"MRR : MRR pondéré."
    )
    if has_skip:
        caption += " \\og -- \\fg{} : ablation non réalisée (moins de 100 domaines observés valides)."
    if bold_best:
        scope = "par classe" if per_class_best else "sur tout le tableau"
        caption += f" Le meilleur MRR {scope} est en gras."

    col_spec = "l r" + "".join("rr" for _ in occs)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:ablation-all}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    head1 = ["", ""] + [f"\\multicolumn{{2}}{{c}}{{$\\mathit{{occ}}={fmt_occ(o)}$}}" for o in occs]
    lines.append(" & ".join(head1) + " \\\\")
    lines.append("".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(occs))))
    head2 = ["Classe", "$\\mathit{min}$"] + ["NA", "MRR"] * len(occs)
    lines.append(" & ".join(head2) + " \\\\")
    lines.append("\\midrule")

    global_mrrs = [c["mrr"] for cells in data.values() for c in cells.values() if "mrr" in c]
    global_best = max(global_mrrs) if global_mrrs else None

    for qi, qid in enumerate(qids):
        cells = data[qid]
        mins = sorted({mn for (mn, _) in cells})
        class_mrrs = [c["mrr"] for c in cells.values() if "mrr" in c]
        class_best = max(class_mrrs) if class_mrrs else None
        for ri, mn in enumerate(mins):
            row_prefix = f"\\multirow{{{len(mins)}}}{{*}}{{{class_title(qid)}}}" if ri == 0 else ""
            row = [row_prefix, str(mn)]
            for o in occs:
                c = cells.get((mn, o))
                if c is None or c.get("skipped"):
                    row.append("\\multicolumn{2}{c}{--}")
                    continue
                mrr = f"{c['mrr']:.{decimals}f}"
                if bold_best:
                    best = class_best if per_class_best else global_best
                    if best is not None and c["mrr"] == best:
                        mrr = f"\\textbf{{{mrr}}}"
                row.append(str(c["na"]))
                row.append(mrr)
            lines.append(" & ".join(row) + " \\\\")
        if qi < len(qids) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="fichier de résultats (.txt)")
    ap.add_argument("-o", "--output", help="fichier .tex de sortie (défaut : stdout)")
    ap.add_argument("--bold-best", action="store_true", help="met en gras le meilleur MRR de chaque ligne/classe")
    ap.add_argument("--decimals", type=int, default=3, help="décimales pour le MRR (défaut : 3)")
    ap.add_argument("--per-class", action="store_true", help="un tableau séparé par classe au lieu d'un seul tableau combiné")
    args = ap.parse_args()

    data, timeouts = parse_file(args.input)
    if not data:
        sys.exit("Aucun résultat reconnu dans le fichier.")

    if args.per_class:
        tables = [
            build_table(qid, data[qid], timeouts, bold_best=args.bold_best, decimals=args.decimals)
            for qid in sorted(data)
        ]
        body = "\n\n".join(tables)
        need = "\\usepackage{booktabs}"
    else:
        body = build_combined_table(data, timeouts, bold_best=args.bold_best, decimals=args.decimals)
        need = "\\usepackage{booktabs,multirow}"

    out = f"% Généré par make_tex_table.py — nécessite {need}\n\n" + body + "\n"

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        n = sum(len(v) for v in data.values())
        print(f"{n} configurations, {len(data)} classes écrites dans {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()