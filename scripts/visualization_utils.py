import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import base64

def make_water_loss_viz(initial_weight_kg: float,
                        predicted_loss_kg: float,
                        age: int,
                        gender: str,
                        image_path: str = "assets/body_water_ref.png",
                        save_path: str = "assets/hydration_viz.png",
                        return_base64: bool = False) -> dict:
    """
    Generate a hydration visualization showing body composition and water loss percentage.
    Displays a body image at the top and a summary table + hydration bar below it.
    """

    # --- Step 1: Define average body water percentage by life stage ---
    if age < 2:
        avg_water_pct = 80
    elif age < 12:
        avg_water_pct = 70 if age < 5 else 65
    elif gender.lower() == "female":
        avg_water_pct = 55
    elif age > 60:
        avg_water_pct = 50
    else:
        avg_water_pct = 60

    # --- Step 2: Compute percentages ---
    percent_loss = (predicted_loss_kg / initial_weight_kg) * 100
    remaining_water = avg_water_pct - percent_loss
    warning = percent_loss > 2.25

    # --- Step 3: Load image if available ---
    img = mpimg.imread(image_path) if os.path.exists(image_path) else None

    # --- Step 4: Create figure ---
    fig = plt.figure(figsize=(5, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1.8])  # top: image, bottom: table + bar

    # --- Top subplot: image ---
    ax_img = fig.add_subplot(gs[0])
    ax_img.axis("off")
    if img is not None:
        ax_img.imshow(img, aspect='equal')
        ax_img.set_anchor('C')
        ax_img.set_position([0.15, 0.55, 0.7, 0.35])  # x, y, width, height
    else:
        ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=12, color="gray")

    # --- Bottom subplot: summary + neat horizontal bar ---
    ax = fig.add_subplot(gs[1])
    ax.axis("off")

    # Create table data
    table_data = [
        ["Initial weight (kg)", f"{initial_weight_kg:.2f}"],
        ["Predicted water loss (kg)", f"{predicted_loss_kg:.3f}"],
        ["% Body weight lost", f"{percent_loss:.2f}%"],
        ["Avg body water %", f"{avg_water_pct:.1f}%"],
        ["Remaining water %", f"{remaining_water:.2f}%"],
    ]

    # Add table
    table = ax.table(cellText=table_data,
                     colLabels=["Metric", "Value"],
                     loc="upper center",
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.4)

    # Style the header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#007acc")
        else:
            cell.set_facecolor("#f2f8fd")

    # --- Alert message ---
    alert_y = 1.09
    if warning:
        ax.text(0.02, alert_y, "⚠️ Predicted water loss exceeds 2.25% — dehydration risk!",
                color="red", fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)
    else:
        ax.text(0.02, alert_y, "Hydration levels within safe range.",
                color="green", fontsize=11, fontweight="bold", va="top", transform=ax.transAxes)

    # --- Clean horizontal hydration bar ---
    bar_y = 0.25   # lowered to appear below the table
    bar_height = 0.12

    # Background (100% total body water)
    ax.barh(bar_y, 100, color="#e0e0e0", height=bar_height, edgecolor="gray", zorder=1)

    # Average body water (light blue)
    ax.barh(bar_y, avg_water_pct, color="#64bae5", height=bar_height, edgecolor="none", zorder=2)

    # Predicted loss overlay (bright blue)
    ax.barh(bar_y, percent_loss, color="#fb2626", height=bar_height, edgecolor="none", zorder=3)

    # Labels
    ax.text(avg_water_pct + 1.5, bar_y, f"Avg {avg_water_pct:.1f}%", va="center", fontsize=9, zorder=4)
    ax.text(percent_loss + 1.5, bar_y + 0.09, f"Lost {percent_loss:.2f}%", va="center", fontsize=9, color="#fb2626", zorder=4)
    ax.text(101, bar_y, "100%", va="center", fontsize=9, color="gray", zorder=4)

    ax.set_xlim(0, max(105, avg_water_pct + 10))
    ax.set_ylim(0, 1.0)   # ensures the bar stays fully visible below the table

    # Remove ticks and spines
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add padding between table and bar
    plt.subplots_adjust(hspace=0.6)

    # --- Step 5: Save visualization ---
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    return {
        "viz_path": save_path,
        "percent_loss": percent_loss,
        "avg_water_pct": avg_water_pct,
        "remaining_water_pct": remaining_water,
        "warning_>2pct": warning,
    }
