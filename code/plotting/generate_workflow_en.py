"""Workflow diagram for the README and the reports.

    python code/plotting/generate_workflow_en.py [en|fr]

Writes figures/fig_workflow_en.png (en) or figures/fig_workflow_fr.png (fr).
Dashed grey boxes are the external codebase (fast model and genetic optimizer).
"""
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

FIG = Path(__file__).resolve().parents[2] / "figures"

TEXT = {
    "en": {
        1: ("1. Fast analytical model", "external code",
            "Thick multilayer cylinder (Lekhnitskii)\nHashin / Tsai-Wu / Puck\n< 1 ms, cylinder only"),
        2: ("2. CalculiX reference", "",
            "384 dome-resolved cases\nS8R composite shells, geodesic winding\none boss fixed, one sliding"),
        3: ("3. Target", "",
            "Fiber index |σ11|/X, worst ply\nC = FI_CalculiX / FI_fast\nC from 1.0 to 4.5 on the DOE"),
        4: ("4. MLP corrector", "",
            "24x12 tanh, NumPy\npredicts log C from design inputs\nplus p95 residual margin"),
        5: ("5. Genetic optimizer", "external code",
            "thinnest wall with the corrected\nfailure index below 1 at 87 MPa"),
        6: ("6. CalculiX re-check", "",
            "17 GA designs re-meshed\nand re-solved\ncompared at p95 and maximum"),
        7: ("7. CalculiX model vs published burst tests (comparison only)", "",
            "Hu 2021: -8 %   |   Agne 2025: -4 to -14 %\nJin 2022: -10 to -22 %   |   DLR 2025: -74 %, not reproduced"),
    },
    "fr": {
        1: ("1. Modèle analytique rapide", "code externe",
            "Cylindre multicouche épais (Lekhnitskii)\nHashin / Tsai-Wu / Puck\n< 1 ms, virole seule"),
        2: ("2. Référence CalculiX", "",
            "384 cas résolus au dôme\ncoques composites S8R, enroulement géodésique\nune embase fixe, une glissante"),
        3: ("3. Cible", "",
            "Indice fibre |σ11|/X, pli critique\nC = FI_CalculiX / FI_rapide\nC de 1.0 à 4.5 sur le DOE"),
        4: ("4. Correcteur MLP", "",
            "24x12 tanh, NumPy\nprédit log C à partir du design\nplus marge résiduelle p95"),
        5: ("5. Optimiseur génétique", "code externe",
            "paroi la plus mince avec l'indice\nde rupture corrigé < 1 à 87 MPa"),
        6: ("6. Re-calcul CalculiX", "",
            "17 designs GA remaillés\net recalculés\ncomparés au p95 et au maximum"),
        7: ("7. Modèle CalculiX vs essais d'éclatement publiés (comparaison seulement)", "",
            "Hu 2021 : -8 %   |   Agne 2025 : -4 à -14 %\nJin 2022 : -10 à -22 %   |   DLR 2025 : -74 %, non reproduit"),
    },
}

STYLE = {
    1: dict(color="#f2f2f2", edgecolor="#8a8a8a", external=True),
    2: dict(color="#e8f6f3", edgecolor="#1abc9c"),
    3: dict(color="#fef9e7", edgecolor="#f39c12"),
    4: dict(color="#fbf2f2", edgecolor="#e74c3c"),
    5: dict(color="#f2f2f2", edgecolor="#8a8a8a", external=True),
    6: dict(color="#eef4f8", edgecolor="#3182bd"),
    7: dict(color="#fdfefe", edgecolor="#2c3e50"),
}

BOXES = {1: (0.5, 4.6, 2.5), 2: (4.0, 4.6, 2.7), 3: (7.7, 4.6, 2.5),
         4: (7.7, 2.4, 2.5), 5: (4.0, 2.4, 2.7), 6: (0.5, 2.4, 2.5), 7: (2.0, 0.4, 6.7)}


def generate_chart(lang: str = "en") -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    for key, (x, y, w) in BOXES.items():
        h = 1.2
        title, tag, body = TEXT[lang][key]
        style = STYLE[key]
        external = style.get("external", False)
        if not external:
            ax.add_patch(patches.FancyBboxPatch((x + 0.04, y - 0.04), w, h, boxstyle="round,pad=0.1", fc="#cccccc", alpha=0.3))
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=style["color"], ec=style["edgecolor"],
                                            lw=1.5, ls="--" if external else "-"))
        title_color = "#6b6b6b" if external else "#0b3c5d"
        ax.text(x + w / 2, y + h - 0.15, title, ha="center", va="top", fontsize=9.5, fontweight="bold", color=title_color)
        if tag:
            ax.text(x + w / 2, y + h - 0.42, f"({tag})", ha="center", va="top", fontsize=8, style="italic", color="#6b6b6b")
        ax.text(x + w / 2, y + (h - 0.35) / 2, body, ha="center", va="center", fontsize=8.5,
                color="#555555" if external else "#333333", linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#555555", shrinkA=5, shrinkB=5, mutation_scale=15))

    arrow(3.1, 5.2, 3.9, 5.2)    # fast model -> CalculiX DOE (same designs, both models)
    arrow(6.8, 5.2, 7.6, 5.2)    # CalculiX -> target C
    arrow(8.95, 4.5, 8.95, 3.7)  # target -> MLP
    arrow(7.6, 3.0, 6.8, 3.0)    # MLP -> GA
    arrow(3.9, 3.0, 3.1, 3.0)    # GA designs -> CalculiX re-check

    plt.tight_layout()
    out = FIG / ("fig_workflow_en.png" if lang == "en" else "fig_workflow_fr.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    generate_chart(sys.argv[1] if len(sys.argv) > 1 else "en")
