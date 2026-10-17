# Learner 2 Runtime Comparison

Benchmarks were run without saving learner outputs, so the existing `Results/` baseline files were not overwritten.

## Runtime Results

### Q215380

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `111.6214s` | `2.6039s` | `2.2685s` | `49.20x` | `1.15x` | `109.35s` |
| Propagation | `160.2587s` | `1.0789s` | `0.6066s` | `264.19x` | `1.78x` | `159.65s` |
| Total learner2 | `273.3701s` | `3.8107s` | `2.9998s` | `91.13x` | `1.27x` | `270.37s` |

Semantic counts stayed unchanged for `Q215380`.

### Q6256

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `9055.8165s` | `6.7736s` | `5.1781s` | `1748.87x` | `1.31x` | `9050.64s` |
| Propagation | `426.4171s` | `1.8511s` | `1.4261s` | `299.01x` | `1.30x` | `424.99s` |
| Total learner2 | `9485.4482s` | `8.8631s` | `6.8450s` | `1385.75x` | `1.29x` | `9478.60s` |

Semantic counts stayed unchanged for `Q6256`.

### Q82955

| Stage | Old | Previous Optimized | Latest | Speedup vs Old | Speedup vs Previous | Time Saved vs Old |
|---|---:|---:|---:|---:|---:|---:|
| Oracle | `9783.7439s` | not run | `34.4111s` | `284.32x` | n/a | `9749.33s` |
| Propagation | `3528.2207s` | not run | `14.2261s` | `248.01x` | n/a | `3513.99s` |
| Total learner2 | `13322.7398s` | not run | `49.6066s` | `268.57x` | n/a | `13273.13s` |

Semantic counts stayed unchanged for `Q82955`.

## Why The Code Is Faster

The approach and semantic outputs are preserved. The speedups come from reducing Python overhead in the same computations.

### Oracle Phase

The oracle was the largest bottleneck. Previously, every property pair scanned entities and created an `AllenRelation` object for every pair of triples, then built a dictionary of all Allen axiom checks.

Changes:

- Added a reusable property/entity oracle index so each `Query()` call only iterates entities that actually contain both queried properties.
- Removed per-triple-pair `AllenRelation` object allocation.
- Replaced `check_all_axioms()` dictionary creation with direct interval comparisons that increment the same Allen relation counters.
- Added a chunked NumPy vectorized path for normal datetime intervals, so large triple cross-products are evaluated in array operations instead of Python nested loops.
- Store indexed interval endpoints as integer day values instead of datetime scalars, reducing NumPy comparison overhead.
- Kept a scalar fallback for unusual intervals such as missing bounds.

### Propagation Phase

Propagation repeatedly reads and updates QCN domains. Previously, it repeatedly rebuilt score/status dictionaries and repeatedly scanned the queue for membership.

Changes:

- Added a `queued` set next to the `deque`, making queue membership checks constant-time instead of linear-time.
- Replaced repeated hot-path calls to `relation_scores()`, `relation_statuses()`, and `relation_entries()` with direct payload access inside `propagateAndFilter()` where domains are already internally controlled.
- Replaced repeated full inverse-domain rebuilds with incremental converse relation updates through `ALLEN_CONVERSE`.
- Cached each domain's positive scores, composable scores, and observed-status flag, refreshing only changed pairs and their converses.
- Updated propagation caches incrementally for each changed relation instead of rescanning both changed domains.
- Added a memoized Allen composition fast path for repeated scored-domain compositions.

### Copy And Stats Overhead

The learner also spent time copying large QCN structures and scanning them multiple times for report statistics.

Changes:

- Removed safe unnecessary deep copies after oracle and propagation.
- Removed unnecessary deep copies of loaded entities/properties in `main()`.
- Removed the remaining oracle query-result deep copy and replaced validated converse rebuilding with a fast hot-path converse builder.
- Combined after-oracle QCN stats and report stats into one traversal.

## Verification

The optimized code was checked with:

```bash
PYTHONPATH=src .venv/bin/pytest
PYTHONPATH=src .venv/bin/python -m tclkg.ground_truth verify --include-slow
.venv/bin/ruff check src/tclkg/qcn_generator2.py
```

## Mesures du 17 octobre 2026 — référence verrouillée

Les données et le code d’origine ont été figés avant optimisation. Les références ont été calculées avec cette copie du code sur les trois jeux complets reconstitués depuis `data/Data_chunk*`. Les six JSON préexistants dans `WWWfactChecking/result_train/tcl` ont été conservés et leurs SHA-256 revérifiés.

### Méthode

Médiane de trois répétitions dans des processus Python 3.14 neufs et séquentiels, sur la même machine et avec le même `uv.lock`. Le chronométrage comprend le chargement, la construction du réseau, `learner2`, les exports JSON/statistiques et l’analyse des scores. Le démarrage de Python, les transferts entre processus et le calcul des empreintes de vérification sont exclus. Les données sont relues pour vérifier leur SHA-256 avant le chronométrage : le cache du système de fichiers est chaud dans les deux versions.

Les exécutions individuelles utilisent le mode conservative, la composition réduite, un seuil d’occurrence nul et les minimums 10 / 30 / 128 pour Q6256 / Q215380 / Q82955. Elles ne réutilisent pas d’oracle préexistant.

### Temps mesurés

| Jeu | Avant | Après | Temps économisé | Réduction | Accélération |
|---|---:|---:|---:|---:|---:|
| Q6256 | 35.31 s | 19.59 s | 15.72 s | 44.5 % | 1.80× |
| Q215380 | 9.82 s | 4.39 s | 5.43 s | 55.3 % | 2.24× |
| Q82955 | 180.34 s | 56.54 s | 123.80 s | 68.6 % | 3.19× |

Détail de la propagation seule (temps CPU, médianes) :

| Jeu | Avant | Après | Accélération |
|---|---:|---:|---:|
| Q6256 | 17.296 s | 7.621 s | 2.27× |
| Q215380 | 2.667 s | 0.112 s | 23.76× |
| Q82955 | 97.486 s | 0.924 s | 105.48× |

### Campagne avec réutilisation de l’oracle

La campagne mesurée contient trois configurations Q215380 : `(occurrence, minimum)` = `(0, 0)`, `(0.5, 30)`, `(0.9, 150)`. Chaque répétition repart sans cache. La première configuration remplit le cache disque, les deux suivantes le lisent. Les coûts de publication et de lecture du cache sont inclus.

La médiane passe de **30.58 s à 8.22 s**, soit **22.36 s économisées (73.1 %, 3.72×)**. Le nombre d’appels à l’oracle passe de 42 084 à 14 028. Ces valeurs portent sur les trois configurations mesurées ; aucune durée n’est extrapolée à la grille complète de 30 configurations par base.

### Équivalence des résultats

- Comparaison des trois états (`initial`, `after_oracle`, `after_propagation`) pour les trois bases, à chaque répétition.
- Empreintes couvrant chaque paire de propriétés, contrainte, support, score d’inférence, statut et étape. Les flottants sont comparés exactement ; seuls l’ordre des dictionnaires et les représentations JSON des tuples/listes sont normalisés.
- Relecture des snapshots compressés de référence et optimisés, puis comparaison avec le manifeste portable. Aucune différence.
- Comparaison complète des trois configurations de campagne à chaque répétition. Aucune différence.
- 192 cas portables figés couvrent les quatre modes de propagation, les deux compositions et les seuils.

### Modifications

- Parcours des seuls chemins dont les deux sources observées sont admissibles ; suppression des réinsertions inutiles dans la file conservative.
- Chemin scalaire pour les paires de groupes entiers de taille 1×1 ; maintien du traitement NumPy pour les autres cas et des relations multiples des intervalles ponctuels.
- Filtrage des relations sans occurrence avant de copier leurs payloads.
- Oracle réutilisable par campagne, clé calculée depuis les données, le code de préparation et l’environnement ; publication atomique et préparation unique par base avant les jobs suivants.
- Apprentissage CLI dans le processus possédant déjà les données, timeout assuré par son parent et respect de `--max-parallel`.
- Export JSON compact par domaine avec le même schéma ; analyse des scores depuis les réseaux en mémoire.

### Vérifications et limites connues

Les vérifications ciblées terminent avec le code 0 : **240 tests, deux sous-tests**, Ruff, formatage, Pyright et les quatre hooks pre-commit existants. Le comparateur relit les snapshots et vérifie aussi l’intégrité des données, du code de référence, du verrou de dépendances et des six résultats préexistants.

La suite globale ne passe pas intégralement avant cette intervention : `fact_checking.py` est absent et empêche la collecte du test associé. En excluant uniquement ce fichier de test pour examiner le reste, le code d’origine a 60 tests réussis et six échecs ; le code actuel a 261 tests réussis et les six mêmes échecs. Les noms des échecs ont été comparés automatiquement. Ils concernent les anciennes attentes de la mini-pipeline, deux statistiques de `random_experiments` et trois tests de `tkgs_stats`. Ces modules de statistiques et leurs tests n’ont pas été modifiés. Les diagnostics de départ des contrôles globaux Ruff/Pyright sont également conservés.

### Fichiers et reproduction

- Verrou portable complet : `ground_truth/performance_manifest.json`.
- Cas portables : `ground_truth/performance_cases.json`.
- Source d’origine, provenance et snapshots : `performance_artifacts/reference/`.
- Snapshots optimisés : `performance_artifacts/optimized/results/`.
- Mesures brutes : `performance_artifacts/baseline.json`, `optimized.json`, `baseline_sweep.json`, `optimized_sweep.json`.
- Journaux de vérification : `performance_artifacts/checks/`.
- Relecture indépendante : `uv run python scripts/benchmark_performance.py verify`.
- Protocole détaillé : section « Performance snapshots and measurements » du `README.md`.
