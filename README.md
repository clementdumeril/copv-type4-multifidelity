# Livrables Finaux - Correction Multi-Fidélité COPV Type IV

Ce dossier rassemble l'ensemble des livrables associés au projet d'optimisation et de recalage multi-fidélité des réservoirs composites à hydrogène de type IV (COPV). Il contient les rapports techniques de fin (français et anglais), les jeux de données (DOE), ainsi que l'intégralité du code source structuré.

---

## 📂 Structure du Dossier `livrable_final/`

Le dossier est organisé comme suit :

```text
livrable_final/
├── README_FINAL.md            # Ce fichier (index général et guide de lecture)
├── reports/                   # Rapports de synthèse technique (PDF et LaTeX)
│   ├── french/
│   │   ├── main_public_fr.tex # Source LaTeX du rapport français (8 pages)
│   │   └── rapport_public_fr.pdf # Rapport technique français compilé (PDF)
│   ├── english/
│   │   ├── main_public_en.tex # Source LaTeX du rapport anglais (8 pages)
│   │   └── rapport_public_en.pdf # Rapport technique anglais compilé (PDF)
│   ├── references_public.bib  # Fichier de bibliographie unifié (BibTeX)
│   ├── figures/               # Graphiques et diagrammes insérés dans les rapports
│   └── tables/                # Tableaux LaTeX compilés de manière autonome
├── data/                      # Données de calibration et d'optimisation
│   ├── doe_11l_single_boss_cases.json # Manifeste et métadonnées du DOE (384 cas)
│   ├── fiber_proxy_dataset.csv        # Données de contraintes extraites (CalculiX vs Python)
│   └── extended_statistical_metrics.json # Métriques de performance des régresseurs
└── code/                      # Code source Python structuré par domaine
    ├── optimization/          # Intégration du correcteur au code d'optimisation
    │   └── ga_correction.py   # Couche de correction multi-fidélité intégrée au GA (Interface)
    ├── calibration/           # Entraînement et validation statistique du MLP
    │   ├── extended_statistical_validation.py # Script de validation croisée et d'évaluation OOD
    │   └── verify_latex.py    # Outil de vérification sémantique des rapports LaTeX
    ├── plotting/              # Scripts de génération des graphiques scientifiques
    │   ├── generate_workflow_en.py       # Générateur du diagramme de flux (Figure 1)
    │   ├── plot_mesh_convergence_en.py   # Traceur de l'étude de maillage (Figure 3)
    │   ├── plot_material_sensitivity_en.py # Histogramme Monte Carlo (Figure 7a)
    │   └── plot_progressive_damage_en.py  # Graphique de dégradation de rigidité (Figure 7b)
    └── analysis_and_reports/  # Scripts annexes de post-traitement et de rapports
        ├── anisotropic_layer_report.py
        ├── generate_type4_web_report.py
        ├── type4_compare_outputs.py
        ├── type4_dome_visualization.py
        ├── type4_full_tank_paraview.py
        ├── openfoam_coherence_campaign.py
        └── openfoam_copv_solid_run.py
```

---

## 📝 Résumé de la Méthodologie et des Résultats

L'objectif de cette étude est de calibrer un **correcteur statistique externe** sur un modèle analytique rapide (basé sur la CLT et la théorie des membranes) sans modifier son code source. La référence est un modèle éléments finis axisymétrique 2.5D sous **CalculiX CrunchiX**, résolvant précisément le dôme (loi d'enroulement géodésique de Clairaut et gradient d'épaisseur) sous conditions de **pivot glissant** (`single_boss_reference`).

### Points clés :
*   **Observable cible :** Indice de contrainte pure dans la fibre de carbone (`fiber_proxy`), évitant les échelles aberrantes des indicateurs de matrice combinés.
*   **Dataset principal (Dataset C) :** Campagne de **384 cas** (288 d'entraînement, 48 de validation, 48 de test hors-distribution).
*   **Modèle retenu :** Un perceptron multicouche (MLP Tanh 24x12) qui réduit l'erreur absolue moyenne (MAE) à **0,102** ($R^2$ de **0,862** en validation croisée).
*   **Vérification de convergence :** Utilisation des quantiles de contrainte (**p99/p95**) pour amortir les singularités géométriques polaires près des embases.
*   **Benchmarks :** Recalage et ordres de grandeur comparés avec succès sur plusieurs références physiques de la littérature (*Hu 2021*, *Agne 2025*, *Paik 2022*).

---

## 🚀 Guide de Reproduction rapide

### 1. Prérequis
Assurez-vous d'avoir Python 3 installé avec les bibliothèques suivantes :
```bash
pip install numpy pandas matplotlib
```

### 2. Régénérer les Métriques et les Graphiques du MLP
Pour ré-entraîner les modèles (OLS, Ridge, MLP) et générer les graphiques d'évaluation (`extended_benchmark_nuage_en.png` et `extended_benchmark_residus_en.png`) :
```bash
# Définir les variables d'environnement vers les fichiers de données du dépôt
$env:TYPE4_OUTPUTS_DIR="d:/SimulationD/genetic_claude_V2/type4_calibration/outputs_11l_single_boss"
$env:TYPE4_CASES_FILE="d:/SimulationD/genetic_claude_V2/type4_calibration/config/doe_11l_single_boss_cases.json"

# Lancer le script de calibration
python code/calibration/extended_statistical_validation.py
```

### 3. Couche d'Optimisation
> [!NOTE]  
> Le code de l'Algorithme Génétique (AG) est un framework externe collaboratif/propriétaire et a été exclu de ce dépôt public pour des raisons de propriété intellectuelle. Le script [ga_correction.py](code/optimization/ga_correction.py) est conservé à titre d'exemple d'implémentation pour illustrer comment la correction multi-fidélité s'interface avec la fonction de fitness du GA.

### 4. Compiler les Rapports LaTeX
Pour compiler les documents sources LaTeX (MiKTeX nécessaire) :
```bash
cd reports/french
pdflatex main_public_fr.tex
bibtex main_public_fr
pdflatex main_public_fr.tex
pdflatex main_public_fr.tex
```
*(Suivre la même procédure dans `reports/english` pour la version anglaise).*
