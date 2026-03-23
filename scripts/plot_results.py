"""Generate publication-quality comparison plots for AutoFilter results."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

DOCS = "docs"

# Data
models = ["0.8B", "2B", "4B"]
no_think_em = [26.3, 42.8, 57.0]
think_em = [37.0, 47.2, 61.7]
no_think_f1 = [57.9, 70.3, 77.9]
think_f1 = [64.3, 71.9, 82.6]
no_think_sv = [100.0, 95.9, 92.8]
think_sv = [99.3, 98.3, 95.9]
no_think_fa = [97.2, 93.8, 90.6]
think_fa = [90.9, 92.9, 92.5]

colors_no = ["#4A90D9", "#3B7DD8", "#2C6ABF"]
colors_yes = ["#E8734A", "#E05A2D", "#D44516"]
c_no = "#4A90D9"
c_yes = "#E8734A"

x = np.arange(len(models))
w = 0.32


# ── Plot 1: EM comparison (thinking vs no thinking) ──────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - w/2, no_think_em, w, label="No Thinking", color=c_no, edgecolor="white", linewidth=1.5, zorder=3)
bars2 = ax.bar(x + w/2, think_em, w, label="With Thinking", color=c_yes, edgecolor="white", linewidth=1.5, zorder=3)

for bar, val in zip(bars1, no_think_em):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val}%", ha='center', va='bottom', fontweight='bold', fontsize=11, color="#333")
for bar, val in zip(bars2, think_em):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val}%", ha='center', va='bottom', fontweight='bold', fontsize=11, color="#333")

# Gain annotations
for i in range(len(models)):
    gain = think_em[i] - no_think_em[i]
    mid_x = x[i] + w/2 + 0.05
    mid_y = (no_think_em[i] + think_em[i]) / 2
    ax.annotate(f"+{gain:.1f}%", xy=(mid_x, think_em[i] + 4), fontsize=9, color="#2a8636", fontweight='bold', ha='center')

ax.set_xlabel("Model Size", fontsize=13, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13, fontweight='bold')
ax.set_title("Exact Match: Thinking vs No Thinking", fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"Qwen3.5-{m}" for m in models], fontsize=12)
ax.set_ylim(0, 75)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{DOCS}/em_thinking_comparison.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 2: Model scaling (all metrics) ──────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metrics = [
    ("Exact Match (%)", think_em, no_think_em),
    ("F1 Score (%)", think_f1, no_think_f1),
    ("Field Accuracy (%)", think_fa, no_think_fa),
    ("Structural Validity (%)", think_sv, no_think_sv),
]

for ax, (title, think_vals, no_think_vals) in zip(axes, metrics):
    bars1 = ax.bar(x - w/2, no_think_vals, w, label="No Thinking", color=c_no, edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + w/2, think_vals, w, label="With Thinking", color=c_yes, edgecolor="white", linewidth=1)
    for bar, val in zip(bars1, no_think_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#333")
    for bar, val in zip(bars2, think_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#333")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}" for m in models], fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=9, loc='upper left')
fig.suptitle("AutoFilter: Multi-Metric Comparison Across Model Sizes", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{DOCS}/all_metrics_comparison.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 3: Thinking gain by model size ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
gains = [think_em[i] - no_think_em[i] for i in range(3)]
gradient_colors = ["#66BB6A", "#43A047", "#2E7D32"]
bars = ax.bar(models, gains, color=gradient_colors, edgecolor="white", linewidth=2, width=0.5, zorder=3)

for bar, val in zip(bars, gains):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"+{val:.1f}%", ha='center', va='bottom', fontsize=13, fontweight='bold', color="#1B5E20")

ax.set_xlabel("Model Size", fontsize=13, fontweight='bold')
ax.set_ylabel("EM Gain from Thinking (%)", fontsize=13, fontweight='bold')
ax.set_title("Impact of Thinking Mode by Model Size", fontsize=15, fontweight='bold', pad=15)
ax.set_xticklabels([f"Qwen3.5-{m}" for m in models], fontsize=12)
ax.set_ylim(0, 15)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{DOCS}/thinking_gain.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 4: Scaling curve ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
params = [0.866, 1.5, 4.6]  # billions

ax.plot(params, no_think_em, 'o-', color=c_no, linewidth=2.5, markersize=10, label="No Thinking", zorder=3)
ax.plot(params, think_em, 's-', color=c_yes, linewidth=2.5, markersize=10, label="With Thinking", zorder=3)

for i, (p, nt, t) in enumerate(zip(params, no_think_em, think_em)):
    ax.annotate(f"{nt}%", (p, nt), textcoords="offset points", xytext=(-15, -18), fontsize=10, fontweight='bold', color=c_no)
    ax.annotate(f"{t}%", (p, t), textcoords="offset points", xytext=(-15, 10), fontsize=10, fontweight='bold', color=c_yes)

ax.fill_between(params, no_think_em, think_em, alpha=0.1, color=c_yes)
ax.set_xlabel("Parameters (Billions)", fontsize=13, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13, fontweight='bold')
ax.set_title("Scaling Behavior: EM vs Model Size", fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim(15, 70)
ax.set_xlim(0.5, 5.2)
ax.grid(alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{DOCS}/scaling_curve.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 5: Per-difficulty heatmap (4B thinking - best model) ────────────
difficulties = [
    "easy", "medium", "negative", "adversarial_typos", "adversarial_numbers_as_words",
    "adversarial_slang", "adversarial", "adversarial_mixed", "adversarial_abbreviations",
    "array_easy", "array_medium", "array_hard",
    "or_easy", "or_medium", "or_hard",
    "or_mixed_easy", "or_mixed_medium", "or_mixed_hard",
    "hallucination", "hallucination_full", "cross_schema",
    "mix", "hard"
]

# 4B thinking best checkpoint EM per difficulty (from eval results)
em_4b_think = [
    95, 75, 80, 90, 90,
    75, 70, 65, 55,
    85, 75, 15,
    85, 70, 20,
    70, 70, 15,
    45, 50, 55,
    40, 5
]

# Sort by EM descending
sorted_pairs = sorted(zip(difficulties, em_4b_think), key=lambda x: x[1], reverse=True)
difficulties = [p[0] for p in sorted_pairs]
em_4b_think = [p[1] for p in sorted_pairs]

fig, ax = plt.subplots(figsize=(12, 7))
colors_map = []
for v in em_4b_think:
    if v >= 80: colors_map.append("#2E7D32")
    elif v >= 60: colors_map.append("#66BB6A")
    elif v >= 40: colors_map.append("#FFA726")
    elif v >= 20: colors_map.append("#EF5350")
    else: colors_map.append("#B71C1C")

y_pos = np.arange(len(difficulties))
bars = ax.barh(y_pos, em_4b_think, color=colors_map, edgecolor="white", linewidth=0.5, height=0.7)

for bar, val in zip(bars, em_4b_think):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val}%", va='center', fontsize=9, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(difficulties, fontsize=9)
ax.set_xlabel("Exact Match (%)", fontsize=12, fontweight='bold')
ax.set_title("Qwen3.5-4B + Thinking: Performance by Query Difficulty", fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 105)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2E7D32', label='80-100% (Strong)'),
    Patch(facecolor='#66BB6A', label='60-79% (Good)'),
    Patch(facecolor='#FFA726', label='40-59% (Moderate)'),
    Patch(facecolor='#EF5350', label='20-39% (Weak)'),
    Patch(facecolor='#B71C1C', label='0-19% (Failing)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig(f"{DOCS}/difficulty_breakdown.png", dpi=200, bbox_inches='tight')
plt.close()

print("Plots saved to docs/")
print("  - em_thinking_comparison.png")
print("  - all_metrics_comparison.png")
print("  - thinking_gain.png")
print("  - scaling_curve.png")
print("  - difficulty_breakdown.png")
