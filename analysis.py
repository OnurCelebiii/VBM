"""
analysis.py
-----------
Full statistical analysis of GNSS position error data.
VBM604 — Statistical Data Analysis | Hacettepe University

Sections
--------
1.  Data loading & overview
2.  Descriptive statistics
3.  Normality tests  (Shapiro-Wilk, D'Agostino K²)
4.  Hypothesis test  — environment effect (Welch t-test, Cohen's d)
5.  One-way ANOVA   — constellation effect + Tukey HSD post-hoc
6.  Pearson correlation matrix
7.  Multiple linear regression (OLS via statsmodels)
8.  Visualisations   (figures/fig1_overview.png,
                       figures/fig2_corr_regression.png,
                       figures/fig3_constellation_detail.png)
9.  Save full report to output/statistical_report.txt
"""

from pathlib import Path
import warnings
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy import stats
from scipy.stats import (
    shapiro, normaltest, levene, ttest_ind,
    pearsonr, f_oneway
)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = Path("data")  / "gnss_measurements.csv"
FIG_DIR     = Path("figures")
OUTPUT_PATH = Path("output") / "statistical_report.txt"

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Report buffer ──────────────────────────────────────────────────────────────
_report_lines: list[str] = []

def report(*args, **kwargs):
    """Print to console AND accumulate in report buffer."""
    line = " ".join(str(a) for a in args)
    print(line, **kwargs)
    _report_lines.append(line)

def section(title: str):
    banner = "\n" + "=" * 70 + "\n  " + title + "\n" + "=" * 70
    report(banner)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
section("1. DATA LOADING & OVERVIEW")

df = pd.read_csv(DATA_PATH)
report(f"Shape        : {df.shape}")
report(f"Columns      : {list(df.columns)}")
report(f"\nDtypes:\n{df.dtypes.to_string()}")
report(f"\nMissing values:\n{df.isnull().sum().to_string()}")
report(f"\nFirst 5 rows:\n{df.head().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
section("2. DESCRIPTIVE STATISTICS")

err = df["position_error_m"]
desc = err.describe()

report(f"\nposition_error_m — summary:\n{desc.to_string()}")
report(f"\nSkewness  : {err.skew():.4f}")
report(f"Kurtosis  : {err.kurt():.4f}")
report(f"CV (%)    : {(err.std() / err.mean()) * 100:.2f}")

report("\n--- By Environment ---")
report(df.groupby("environment")["position_error_m"].describe().to_string())

report("\n--- By Constellation ---")
report(df.groupby("constellation")["position_error_m"].describe().to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────
section("3. NORMALITY TESTS")

# Shapiro-Wilk (uses random subsample for large N)
sample_for_sw = err.sample(n=min(5000, len(err)), random_state=42)
sw_stat, sw_p = shapiro(sample_for_sw)
report(f"Shapiro-Wilk   : W = {sw_stat:.4f},  p = {sw_p:.3e}")
report(f"  → {'NOT normal' if sw_p < 0.05 else 'Normal'}")

# D'Agostino K²
k2_stat, k2_p = normaltest(err)
report(f"\nD'Agostino K²  : stat = {k2_stat:.2f},  p = {k2_p:.3e}")
report(f"  → {'NOT normal' if k2_p < 0.05 else 'Normal'}")

report(
    "\nInterpretation: Distribution is right-skewed (non-normal). "
    "Parametric tests remain valid under CLT for N=3000."
)

# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPOTHESIS TEST — ENVIRONMENT EFFECT
# ─────────────────────────────────────────────────────────────────────────────
section("4. HYPOTHESIS TEST — ENVIRONMENT EFFECT")

open_sky = df.loc[df["environment"] == "Open Sky",    "position_error_m"]
urban    = df.loc[df["environment"] == "Urban Canyon", "position_error_m"]

report(f"Open Sky    : n={len(open_sky)},  mean={open_sky.mean():.3f} m,  std={open_sky.std():.3f} m")
report(f"Urban Canyon: n={len(urban)},  mean={urban.mean():.3f} m,  std={urban.std():.3f} m")
report(f"Difference  : {urban.mean() - open_sky.mean():.3f} m")

# Levene variance homogeneity
lev_stat, lev_p = levene(open_sky, urban)
report(f"\nLevene test  : stat = {lev_stat:.3f},  p = {lev_p:.3e}")
report(f"  → Variances {'EQUAL' if lev_p >= 0.05 else 'NOT equal'}")

# Welch t-test
t_stat, t_p = ttest_ind(open_sky, urban, equal_var=False)
report(f"\nWelch t-test : t = {t_stat:.3f},  p = {t_p:.3e}")
report(f"  → H₀ {'NOT rejected' if t_p >= 0.05 else 'REJECTED'}  (α = 0.05)")

# Cohen's d
pooled_std = np.sqrt((open_sky.std()**2 + urban.std()**2) / 2)
cohens_d   = abs(urban.mean() - open_sky.mean()) / pooled_std
report(f"\nCohen's d    : {cohens_d:.3f}  (effect size: {'small' if cohens_d < 0.5 else 'medium' if cohens_d < 0.8 else 'large' if cohens_d < 1.2 else 'very large'})")

# ─────────────────────────────────────────────────────────────────────────────
# 5. ONE-WAY ANOVA — CONSTELLATION EFFECT
# ─────────────────────────────────────────────────────────────────────────────
section("5. ONE-WAY ANOVA — CONSTELLATION EFFECT")

groups = [
    df.loc[df["constellation"] == c, "position_error_m"].values
    for c in ["GPS", "Galileo", "GLONASS", "BeiDou"]
]
f_stat, anova_p = f_oneway(*groups)
report(f"ANOVA F = {f_stat:.3f},  p = {anova_p:.3e}")
report(f"  → H₀ {'NOT rejected' if anova_p >= 0.05 else 'REJECTED'}  (α = 0.05)")

# Tukey HSD post-hoc
report("\nTukey HSD post-hoc:")
tukey = pairwise_tukeyhsd(
    df["position_error_m"],
    df["constellation"],
    alpha=0.05
)
report(tukey.summary())

# ─────────────────────────────────────────────────────────────────────────────
# 6. PEARSON CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
section("6. PEARSON CORRELATION ANALYSIS")

numeric_cols = ["position_error_m", "pdop", "num_satellites",
                "cn0_dbhz", "multipath_index", "elevation_deg"]
corr_matrix  = df[numeric_cols].corr()
report("\nCorrelation matrix:\n" + corr_matrix.to_string())

report("\nCorrelations with position_error_m:")
for col in numeric_cols[1:]:
    r, p = pearsonr(df["position_error_m"], df[col])
    sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    report(f"  {col:<22} r = {r:+.3f}   p = {p:.3e}   {sig}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. MULTIPLE LINEAR REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
section("7. MULTIPLE LINEAR REGRESSION (OLS)")

formula = (
    "position_error_m ~ pdop + cn0_dbhz + elevation_deg + num_satellites "
    "+ multipath_index + C(constellation) + C(environment)"
)
model  = smf.ols(formula, data=df).fit()
report(model.summary().as_text())

report(f"\nKey metrics:")
report(f"  R²         : {model.rsquared:.4f}")
report(f"  Adj. R²    : {model.rsquared_adj:.4f}")
report(f"  F-statistic: {model.fvalue:.2f}  (p = {model.f_pvalue:.3e})")
report(f"  AIC        : {model.aic:.2f}")
report(f"  BIC        : {model.bic:.2f}")

report("\nTop coefficients by |t-value|:")
coef_df = pd.DataFrame({
    "coef":  model.params,
    "t":     model.tvalues,
    "p":     model.pvalues,
}).sort_values("t", key=abs, ascending=False)
report(coef_df.head(10).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab10")
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.1)

# ── Figure 1: Overview ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("GNSS Position Error — Overview", fontsize=15, fontweight="bold", y=0.99)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# 1a — Histogram + KDE
ax1 = fig.add_subplot(gs[0, 0])
err.plot(kind="hist", bins=60, density=True, ax=ax1,
         color="#4c9be8", edgecolor="white", alpha=0.8, label="Histogram")
err.plot(kind="kde", ax=ax1, color="#e84c4c", linewidth=2, label="KDE")
ax1.set_title("Distribution of Position Error")
ax1.set_xlabel("Position Error (m)")
ax1.set_ylabel("Density")
ax1.legend(fontsize=9)

# 1b — Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
sm.qqplot(err, line="s", ax=ax2, alpha=0.4, markersize=2,
          markerfacecolor="#4c9be8")
ax2.set_title("Q-Q Plot (Normal Reference)")

# 1c — Box plot by environment
ax3 = fig.add_subplot(gs[0, 2])
df.boxplot(column="position_error_m", by="environment", ax=ax3,
           patch_artist=True,
           boxprops=dict(facecolor="#4c9be8", color="navy"),
           medianprops=dict(color="red", linewidth=2))
ax3.set_title("Position Error by Environment")
ax3.set_xlabel("Environment")
ax3.set_ylabel("Position Error (m)")
plt.sca(ax3); plt.title("Position Error by Environment"); plt.suptitle("")

# 1d — Violin plot by environment
ax4 = fig.add_subplot(gs[1, 0])
sns.violinplot(data=df, x="environment", y="position_error_m",
               palette=["#4c9be8", "#e84c4c"], inner="quartile", ax=ax4)
ax4.set_title("Violin: Error by Environment")
ax4.set_xlabel("Environment")
ax4.set_ylabel("Position Error (m)")

# 1e — Violin by constellation
ax5 = fig.add_subplot(gs[1, 1])
sns.violinplot(data=df, x="constellation", y="position_error_m",
               order=["Galileo", "GPS", "BeiDou", "GLONASS"],
               palette="tab10", inner="quartile", ax=ax5)
ax5.set_title("Violin: Error by Constellation")
ax5.set_xlabel("Constellation")
ax5.set_ylabel("Position Error (m)")

# 1f — CDF
ax6 = fig.add_subplot(gs[1, 2])
for env, grp in df.groupby("environment"):
    sorted_err = np.sort(grp["position_error_m"])
    cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    ax6.plot(sorted_err, cdf, label=env, linewidth=2)
ax6.set_title("CDF of Position Error")
ax6.set_xlabel("Position Error (m)")
ax6.set_ylabel("Cumulative Probability")
ax6.legend(fontsize=9)

fig.savefig(FIG_DIR / "fig1_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved fig1_overview.png")

# ── Figure 2: Correlation & Regression ────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 3, figsize=(16, 10))
fig2.suptitle("Correlation & Regression Analysis", fontsize=14, fontweight="bold")

# 2a — Heatmap
ax = axes2[0, 0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax,
            annot_kws={"size": 8}, linewidths=0.5)
ax.set_title("Pearson Correlation Matrix")

# 2b — pdop vs error
ax = axes2[0, 1]
sample = df.sample(500, random_state=1)
ax.scatter(sample["pdop"], sample["position_error_m"],
           alpha=0.3, s=10, color="#4c9be8")
m, b = np.polyfit(df["pdop"], df["position_error_m"], 1)
x_line = np.linspace(df["pdop"].min(), df["pdop"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#e84c4c", linewidth=2)
ax.set_title(f"PDOP vs Position Error  (r={pearsonr(df['pdop'], df['position_error_m'])[0]:.3f})")
ax.set_xlabel("PDOP")
ax.set_ylabel("Position Error (m)")

# 2c — num_satellites vs error
ax = axes2[0, 2]
ax.scatter(sample["num_satellites"], sample["position_error_m"],
           alpha=0.3, s=10, color="#2ca02c")
m, b = np.polyfit(df["num_satellites"], df["position_error_m"], 1)
x_line = np.linspace(df["num_satellites"].min(), df["num_satellites"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#e84c4c", linewidth=2)
ax.set_title(f"Num Satellites vs Error  (r={pearsonr(df['num_satellites'], df['position_error_m'])[0]:.3f})")
ax.set_xlabel("Number of Satellites")
ax.set_ylabel("Position Error (m)")

# 2d — cn0 vs error
ax = axes2[1, 0]
ax.scatter(sample["cn0_dbhz"], sample["position_error_m"],
           alpha=0.3, s=10, color="#ff7f0e")
m, b = np.polyfit(df["cn0_dbhz"], df["position_error_m"], 1)
x_line = np.linspace(df["cn0_dbhz"].min(), df["cn0_dbhz"].max(), 200)
ax.plot(x_line, m * x_line + b, color="#e84c4c", linewidth=2)
ax.set_title(f"CN0 vs Position Error  (r={pearsonr(df['cn0_dbhz'], df['position_error_m'])[0]:.3f})")
ax.set_xlabel("CN0 (dB-Hz)")
ax.set_ylabel("Position Error (m)")

# 2e — Residuals vs fitted
ax = axes2[1, 1]
fitted    = model.fittedvalues
residuals = model.resid
ax.scatter(fitted, residuals, alpha=0.2, s=8, color="#9467bd")
ax.axhline(0, color="red", linewidth=1.5)
ax.set_title("Residuals vs Fitted Values")
ax.set_xlabel("Fitted Values (m)")
ax.set_ylabel("Residuals")

# 2f — Predicted vs actual
ax = axes2[1, 2]
ax.scatter(df["position_error_m"], fitted, alpha=0.2, s=8, color="#8c564b")
lims = [df["position_error_m"].min(), df["position_error_m"].max()]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
ax.set_title(f"Actual vs Predicted  (R²={model.rsquared:.3f})")
ax.set_xlabel("Actual Position Error (m)")
ax.set_ylabel("Predicted Position Error (m)")
ax.legend(fontsize=9)

fig2.tight_layout()
fig2.savefig(FIG_DIR / "fig2_corr_regression.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved fig2_corr_regression.png")

# ── Figure 3: Constellation Detail ────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
fig3.suptitle("Constellation Comparison — Detailed View",
              fontsize=14, fontweight="bold")

order = ["Galileo", "GPS", "BeiDou", "GLONASS"]

# 3a — Mean error bar chart
ax = axes3[0, 0]
means = df.groupby("constellation")["position_error_m"].mean().reindex(order)
sems  = df.groupby("constellation")["position_error_m"].sem().reindex(order)
bars  = ax.bar(order, means, yerr=sems, capsize=5,
               color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
               alpha=0.8, edgecolor="black")
ax.set_title("Mean Position Error ± SE by Constellation")
ax.set_ylabel("Mean Position Error (m)")
ax.set_xlabel("Constellation")
for bar, val in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
            f"{val:.2f}", ha="center", fontsize=9)

# 3b — Box plots per constellation
ax = axes3[0, 1]
data_to_plot = [df.loc[df["constellation"] == c, "position_error_m"].values for c in order]
bp = ax.boxplot(data_to_plot, labels=order, patch_artist=True,
                medianprops=dict(color="red", linewidth=2))
colors_bp = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Position Error Distribution by Constellation")
ax.set_ylabel("Position Error (m)")

# 3c — Error by constellation + environment (grouped)
ax = axes3[1, 0]
pivot = df.groupby(["constellation", "environment"])["position_error_m"].mean().unstack()
pivot.reindex(order).plot(kind="bar", ax=ax, color=["#4c9be8", "#e84c4c"],
                          edgecolor="black", alpha=0.85)
ax.set_title("Mean Error: Constellation × Environment")
ax.set_ylabel("Mean Position Error (m)")
ax.set_xlabel("Constellation")
ax.legend(title="Environment", fontsize=9)
ax.tick_params(axis="x", rotation=0)

# 3d — Tukey HSD visualisation
ax = axes3[1, 1]
try:
    from statsmodels.graphics.factorplots import interaction_plot
    interaction_plot(
        x=df["constellation"],
        trace=df["environment"],
        response=df["position_error_m"],
        ax=ax,
        colors=["#1f77b4", "#e84c4c"],
        markers=["o", "s"],
        ms=8,
    )
    ax.set_title("Interaction Plot: Constellation × Environment")
    ax.set_ylabel("Mean Position Error (m)")
    ax.set_xlabel("Constellation")
except Exception:
    # Fallback: simple bar
    pivot.reindex(order).T.plot(kind="bar", ax=ax,
                                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
                                alpha=0.85, edgecolor="black")
    ax.set_title("Environment × Constellation (grouped)")
    ax.set_ylabel("Mean Position Error (m)")
    ax.tick_params(axis="x", rotation=0)

fig3.tight_layout()
fig3.savefig(FIG_DIR / "fig3_constellation_detail.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved fig3_constellation_detail.png")

# ─────────────────────────────────────────────────────────────────────────────
# 9. SAVE REPORT
# ─────────────────────────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(_report_lines))

print(f"\nFull statistical report saved → {OUTPUT_PATH}")
print("\nAnalysis complete.")
