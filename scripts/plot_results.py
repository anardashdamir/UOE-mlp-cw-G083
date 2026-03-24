"""Generate comparison plots for AutoFilter results."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import Patch

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 12

DOCS = "docs"

# ── Data ─────────────────────────────────────────────────────────────────
models = ["0.8B", "2B", "4B"]

# Zero-shot (no thinking)
zs_em = [3.5, 1.7, 0.4]
zs_f1 = [6.2, 3.9, 2.7]

# SFT
no_think_em = [25.9, 42.4, 57.0]
think_em    = [36.3, 47.0, 61.7]
no_think_f1 = [58.1, 70.2, 77.4]
think_f1    = [64.0, 71.6, 82.6]
no_think_sv = [100.0, 95.4, 93.0]
think_sv    = [99.6, 98.7, 95.9]
no_think_fa = [97.2, 93.6, 90.7]
think_fa    = [90.8, 93.0, 92.5]

# GRPO (4B thinking only)
grpo_em = 67.6
grpo_f1 = 85.7

c_zs  = "#9E9E9E"
c_no  = "#4A90D9"
c_yes = "#E8734A"
c_grpo = "#7B1FA2"

x = np.arange(len(models))
w = 0.22


# ── Plot 1: Full pipeline comparison (zero-shot → SFT → GRPO) ───────────
fig, ax = plt.subplots(figsize=(11, 6))

bars_zs = ax.bar(x - 1.5*w, zs_em, w, label="Zero-shot", color=c_zs, edgecolor="white", linewidth=1.5, zorder=3)
bars_no = ax.bar(x - 0.5*w, no_think_em, w, label="SFT", color=c_no, edgecolor="white", linewidth=1.5, zorder=3)
bars_th = ax.bar(x + 0.5*w, think_em, w, label="SFT + Thinking", color=c_yes, edgecolor="white", linewidth=1.5, zorder=3)

# GRPO bar only for 4B
grpo_x = x[2] + 1.5*w
ax.bar(grpo_x, grpo_em, w, label="SFT + Thinking + GRPO", color=c_grpo, edgecolor="white", linewidth=1.5, zorder=3)
ax.text(grpo_x, grpo_em + 1, f"{grpo_em}%", ha='center', va='bottom', fontweight='bold', fontsize=10, color="#4A148C")

for bars, vals in [(bars_zs, zs_em), (bars_no, no_think_em), (bars_th, think_em)]:
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val}%",
                ha='center', va='bottom', fontweight='bold', fontsize=9, color="#333")

ax.set_xlabel("Model Size", fontsize=13, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13, fontweight='bold')
ax.set_title("AutoFilter: Full Pipeline — Zero-shot → SFT → GRPO", fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels([f"Qwen3.5-{m}" for m in models], fontsize=12)
ax.set_ylim(0, 80)
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{DOCS}/em_full_pipeline.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 2: EM thinking vs no thinking (SFT only) ───────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
w2 = 0.32
bars1 = ax.bar(x - w2/2, no_think_em, w2, label="No Thinking", color=c_no, edgecolor="white", linewidth=1.5, zorder=3)
bars2 = ax.bar(x + w2/2, think_em, w2, label="With Thinking", color=c_yes, edgecolor="white", linewidth=1.5, zorder=3)

for bar, val in zip(bars1, no_think_em):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val}%", ha='center', va='bottom', fontweight='bold', fontsize=11, color="#333")
for bar, val in zip(bars2, think_em):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, f"{val}%", ha='center', va='bottom', fontweight='bold', fontsize=11, color="#333")

for i in range(len(models)):
    gain = think_em[i] - no_think_em[i]
    ax.annotate(f"+{gain:.1f}%", xy=(x[i] + w2/2 + 0.05, think_em[i] + 4), fontsize=9, color="#2a8636", fontweight='bold', ha='center')

ax.set_xlabel("Model Size", fontsize=13, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13, fontweight='bold')
ax.set_title("Exact Match: Thinking vs No Thinking (SFT)", fontsize=15, fontweight='bold', pad=15)
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


# ── Plot 3: Multi-metric comparison ──────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
metrics = [
    ("Exact Match (%)", think_em, no_think_em),
    ("F1 Score (%)", think_f1, no_think_f1),
    ("Field Accuracy (%)", think_fa, no_think_fa),
    ("Structural Validity (%)", think_sv, no_think_sv),
]

for ax, (title, think_vals, no_think_vals) in zip(axes, metrics):
    bars1 = ax.bar(x - w2/2, no_think_vals, w2, label="No Thinking", color=c_no, edgecolor="white", linewidth=1)
    bars2 = ax.bar(x + w2/2, think_vals, w2, label="With Thinking", color=c_yes, edgecolor="white", linewidth=1)
    for bar, val in zip(bars1, no_think_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#333")
    for bar, val in zip(bars2, think_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold', color="#333")
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].legend(fontsize=9, loc='upper left')
fig.suptitle("AutoFilter: Multi-Metric Comparison Across Model Sizes", fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{DOCS}/all_metrics_comparison.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 4: Thinking gain ────────────────────────────────────────────────
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


# ── Plot 5: Scaling curve (now with zero-shot + GRPO) ────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
params = [0.8, 2.0, 4.0]

ax.plot(params, zs_em, 'D--', color=c_zs, linewidth=2, markersize=8, label="Zero-shot", zorder=3)
ax.plot(params, no_think_em, 'o-', color=c_no, linewidth=2.5, markersize=10, label="SFT", zorder=3)
ax.plot(params, think_em, 's-', color=c_yes, linewidth=2.5, markersize=10, label="SFT + Thinking", zorder=3)
ax.plot([4.0], [grpo_em], '*', color=c_grpo, markersize=18, label="SFT + Thinking + GRPO", zorder=4)

for i, p in enumerate(params):
    ax.annotate(f"{zs_em[i]}%", (p, zs_em[i]), textcoords="offset points", xytext=(12, -5), fontsize=9, color=c_zs, fontweight='bold')
    ax.annotate(f"{no_think_em[i]}%", (p, no_think_em[i]), textcoords="offset points", xytext=(-20, -18), fontsize=10, fontweight='bold', color=c_no)
    ax.annotate(f"{think_em[i]}%", (p, think_em[i]), textcoords="offset points", xytext=(-15, 10), fontsize=10, fontweight='bold', color=c_yes)

ax.annotate(f"{grpo_em}%", (4.0, grpo_em), textcoords="offset points", xytext=(15, 5), fontsize=11, fontweight='bold', color=c_grpo)

ax.fill_between(params, no_think_em, think_em, alpha=0.08, color=c_yes)
ax.set_xlabel("Parameters (Billions)", fontsize=13, fontweight='bold')
ax.set_ylabel("Exact Match (%)", fontsize=13, fontweight='bold')
ax.set_title("Scaling Behavior: Zero-shot → SFT → GRPO", fontsize=15, fontweight='bold', pad=15)
ax.legend(fontsize=10, loc='upper left')
ax.set_ylim(-5, 80)
ax.set_xlim(0.3, 5.0)
ax.grid(alpha=0.3, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f"{DOCS}/scaling_curve.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 6: Difficulty breakdown — SFT vs GRPO (4B thinking) ─────────────
difficulties = [
    "easy", "medium", "hard", "negative", "cross_schema", "mix",
    "adversarial", "adversarial_abbreviations", "adversarial_mixed",
    "adversarial_numbers_as_words", "adversarial_slang", "adversarial_typos",
    "array_easy", "array_medium", "array_hard",
    "or_easy", "or_medium", "or_hard",
    "or_mixed_easy", "or_mixed_medium", "or_mixed_hard",
    "hallucination", "hallucination_full",
]

# SFT 4B thinking EM (from eval_sft_fp16.json)
sft_em = [
    90, 45, 0, 85, 70, 30,
    70, 65, 40, 75, 90, 80,
    65, 50, 15,
    80, 45, 10,
    50, 60, 5,
    35, 85,
]

# GRPO 4B thinking EM (from eval_grpo_fp16.json)
grpo_em_diff = [
    100, 95, 5, 100, 95, 50,
    75, 80, 50, 95, 75, 70,
    55, 80, 25,
    90, 90, 10,
    75, 85, 10,
    50, 95,
]

# Sort by GRPO EM descending
sorted_idx = sorted(range(len(difficulties)), key=lambda i: grpo_em_diff[i], reverse=True)
difficulties = [difficulties[i] for i in sorted_idx]
sft_em = [sft_em[i] for i in sorted_idx]
grpo_em_diff = [grpo_em_diff[i] for i in sorted_idx]

fig, ax = plt.subplots(figsize=(13, 8))
y_pos = np.arange(len(difficulties))
bar_h = 0.35

bars_sft = ax.barh(y_pos + bar_h/2, sft_em, bar_h, label="SFT (4B thinking)", color=c_yes, edgecolor="white", linewidth=0.5, alpha=0.7)
bars_grpo = ax.barh(y_pos - bar_h/2, grpo_em_diff, bar_h, label="SFT + GRPO (4B thinking)", color=c_grpo, edgecolor="white", linewidth=0.5)

for bar, val in zip(bars_sft, sft_em):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val}%", va='center', fontsize=8, color="#333")
for bar, val in zip(bars_grpo, grpo_em_diff):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f"{val}%", va='center', fontsize=8, fontweight='bold', color="#4A148C")

ax.set_yticks(y_pos)
ax.set_yticklabels(difficulties, fontsize=9)
ax.set_xlabel("Exact Match (%)", fontsize=12, fontweight='bold')
ax.set_title("SFT vs GRPO: Performance by Query Difficulty (4B + Thinking)", fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 110)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(f"{DOCS}/difficulty_breakdown.png", dpi=200, bbox_inches='tight')
plt.close()


# ── Plot 7: GRPO improvement delta ──────────────────────────────────────
deltas = [grpo_em_diff[i] - sft_em[i] for i in range(len(difficulties))]
colors_delta = ["#2E7D32" if d > 0 else "#D32F2F" if d < 0 else "#9E9E9E" for d in deltas]

fig, ax = plt.subplots(figsize=(13, 8))
bars = ax.barh(y_pos, deltas, 0.6, color=colors_delta, edgecolor="white", linewidth=0.5)

for bar, val, name in zip(bars, deltas, difficulties):
    if val != 0:
        offset = 1.5 if val > 0 else -1.5
        ha = 'left' if val > 0 else 'right'
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height()/2,
                f"{val:+d}%", va='center', ha=ha, fontsize=9, fontweight='bold',
                color="#1B5E20" if val > 0 else "#B71C1C")

ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(difficulties, fontsize=9)
ax.set_xlabel("EM Change from GRPO (%)", fontsize=12, fontweight='bold')
ax.set_title("GRPO Impact: Per-Difficulty EM Change vs SFT Baseline", fontsize=13, fontweight='bold', pad=15)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [
    Patch(facecolor='#2E7D32', label='Improved'),
    Patch(facecolor='#D32F2F', label='Degraded'),
]
ax.legend(handles=legend_elements, fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(f"{DOCS}/grpo_delta.png", dpi=200, bbox_inches='tight')
plt.close()


print("Plots saved to docs/:")
print("  1. em_full_pipeline.png      — Zero-shot → SFT → GRPO comparison")
print("  2. em_thinking_comparison.png — Thinking vs No Thinking (SFT)")
print("  3. all_metrics_comparison.png — Multi-metric across model sizes")
print("  4. thinking_gain.png          — Thinking mode gain by size")
print("  5. scaling_curve.png          — Scaling with zero-shot + GRPO")
print("  6. difficulty_breakdown.png   — SFT vs GRPO per difficulty")
print("  7. grpo_delta.png             — GRPO improvement delta")
