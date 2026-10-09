"""Workflow diagram for the README and the reports. Usage: python generate_workflow_en.py [en|fr]"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

TEXT = {
    "en": [('1. Fast Python Analytical Model', 'Thick multilayer cylinder (Lekhnitskii)\nFast execution (<1 ms)\nCylinder section only'), ('2. Finite Element Reference', '384 dome-resolved cases\nSliding boss (free axial motion)\n2.5D shell model under CalculiX'), ('3. Physical Observable', 'Fibre index |s11|/X, worst ply\nC = FI_CalculiX / FI_fast\nC between 1.0 and 4.5 on the DOE'), ('4. Multi-Fidelity MLP Corrector', 'Multilayer Perceptron (24x12)\nTanh activation\nPredicts C from design inputs\nSecured by p95 residual margin'), ('5. Genetic Optimization', 'Genetic Algorithm (GA)\nCorrected fibre constraint\nSearch for optimal layups'), ('6. Finite Element Verification', 'Validation of GA candidates\nunder CalculiX\nConservatism check'), ('7. Experimental & Physical Benchmarks (Relative Validation)', 'Order-of-magnitude quantitative comparison with published data:\n- Jin 2022 EX-A (65.2 MPa, p99 fibre proxy) | - Hu 2021 (70 MPa, first fibre damage)\n- Agne 2025 (GFRP / CFRP, exact reconstruction of layup table)')],
    "fr": [('1. Modèle Analytique Python', 'Cylindre multicouche épais (Lekhnitskii)\nRapide (< 1 ms)\nUniquement section virole'), ('2. Référence CalculiX', '384 cas résolus au dôme\nPivot glissant libre axial\nModèle de coque 2.5D'), ('3. Observable Physique', 'Indice fibre |s11|/X, pli critique\nC = FI_CalculiX / FI_rapide\nC entre 1.0 et 4.5 sur le DOE'), ('4. Correcteur MLP', 'Réseau Tanh (24x12)\nPrédiction du ratio C\nSécurisé par marge p95'), ('5. Optimisation Génétique', "Algorithme génétique (GA)\nContrainte fibre corrigée\nRecherche d'empilements"), ('6. Vérification Éléments Finis', 'Validation des candidats GA\nsous CalculiX\nTaux de conservatisme'), ('7. Benchmarks Expérimentaux & Physiques (Validation Relative)', "Comparaison quantitative d'ordre de grandeur avec les données publiées :\n- Jin 2022 EX-A (65.2 MPa, p99 proxy fibre) | - Hu 2021 (70 MPa, premier dommage fibre)\n- Agne 2025 (GFRP / CFRP, reconstruction exacte de table de plis)")],
}

def generate_chart(lang: str = "en"):
    T = TEXT[lang]
    LOOP = "Feedback loop" if lang == "en" else "Boucle de rétroaction"
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis('off')
    
    # Background color of the figure
    fig.patch.set_facecolor('white')

    def draw_box(x, y, w, h, title, text, color='#f0f5fa', edgecolor='#1f77b4', title_color='#0b3c5d'):
        # Shadow
        rect_shadow = patches.FancyBboxPatch((x+0.04, y-0.04), w, h, boxstyle="round,pad=0.1", fc='#cccccc', alpha=0.3)
        ax.add_patch(rect_shadow)
        
        # Main box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc=color, ec=edgecolor, lw=1.5)
        ax.add_patch(rect)
        
        # Title text
        ax.text(x + w/2, y + h - 0.15, title, ha='center', va='top', fontsize=9.5, fontweight='bold', color=title_color)
        # Body text
        ax.text(x + w/2, y + (h - 0.25)/2 + 0.05, text, ha='center', va='center', fontsize=8.5, color='#333333', linespacing=1.3)

    # 1. Row 1: Pipeline de Données
    draw_box(0.5, 4.6, 2.5, 1.2, 
             T[0][0], 
             T[0][1],
             color='#eef4f8', edgecolor='#3182bd')
             
    draw_box(4.0, 4.6, 2.7, 1.2, 
             T[1][0], 
             T[1][1],
             color='#e8f6f3', edgecolor='#1abc9c')
             
    draw_box(7.7, 4.6, 2.5, 1.2, 
             T[2][0], 
             T[2][1],
             color='#fef9e7', edgecolor='#f39c12')

    # 2. Row 2: Correction & Optimisation
    draw_box(7.7, 2.4, 2.5, 1.2, 
             T[3][0], 
             T[3][1],
             color='#fbf2f2', edgecolor='#e74c3c')
             
    draw_box(4.0, 2.4, 2.7, 1.2, 
             T[4][0], 
             T[4][1],
             color='#f3eff5', edgecolor='#8e44ad')
             
    draw_box(0.5, 2.4, 2.5, 1.2, 
             T[5][0], 
             T[5][1],
             color='#f4f6f6', edgecolor='#7f8c8d')

    # 3. Validation / Étude de fond (bottom)
    draw_box(2.0, 0.4, 6.7, 1.2, 
             T[6][0], 
             T[6][1],
             color='#fdfefe', edgecolor='#2c3e50', title_color='#2c3e50')

    # Helper function for arrows
    def draw_arrow(x1, y1, x2, y2, text=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color='#555555', shrinkA=5, shrinkB=5, mutation_scale=15))
        if text:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.1, text, ha='center', va='bottom', fontsize=8, color='#555555')

    # Draw Arrows between boxes
    draw_arrow(3.1, 5.2, 3.9, 5.2)  # 1 -> 2
    draw_arrow(6.8, 5.2, 7.6, 5.2)  # 2 -> 3
    draw_arrow(8.95, 4.5, 8.95, 3.7) # 3 -> 4
    draw_arrow(7.6, 3.0, 6.8, 3.0)  # 4 -> 5
    draw_arrow(3.9, 3.0, 3.1, 3.0)  # 5 -> 6
    draw_arrow(1.75, 3.7, 1.75, 4.5, LOOP) # 6 -> 1

    # Connect Validation
    draw_arrow(5.35, 4.5, 5.35, 3.7) # 2 -> 5 (CalculiX is the reference for GA)
    draw_arrow(5.35, 2.3, 5.35, 1.7) # 5 -> 7 (Validation of GA candidates)

    plt.tight_layout()
    plt.savefig(str(Path(__file__).resolve().parents[2] / 'reports' / 'figures' / ('fig_workflow_en.png' if lang == 'en' else 'fig_workflow.png')), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Workflow figure ({lang}) generated.")

if __name__ == "__main__":
    import sys
    generate_chart(sys.argv[1] if len(sys.argv) > 1 else "en")
