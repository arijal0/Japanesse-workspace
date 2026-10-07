"""Independent re-analysis of the released human ratings (Gao et al., 2026).

Recomputes every headline number from data/aggre_huaman_scores.csv without
reusing the authors' notebook code, then adds robustness checks the paper does
not report.

Run from the repository root:
    .venv/bin/python replication/01_independent_analysis.py
"""

import itertools
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "replication" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["LLM-jp", "Llama", "Phi", "Qwen", "GLM"]
DIMS = ["PDI", "IDV", "UAI", "MAS", "LTO", "IND"]
COLORS = {"LLM-jp": "#6D1637", "Llama": "#F0B353", "Phi": "#4BA082", "Qwen": "#6255E9", "GLM": "#808080"}

PAPER_JWCAS = {"Phi": 3.502, "GLM": 3.468, "LLM-jp": 3.433, "Qwen": 3.390, "Llama": 3.325}

# Question -> diagnostic category, transcribed from the questionnaire comments in
# statistical_analysis.ipynb (cell 10). Q5 is always "Overall Fit".
CATEGORY = {
    "PDI": {1: "Linguistic Form", 3: "Linguistic Form", 2: "Social Strategy", 4: "Social Strategy"},
    "IDV": {1: "Strategic Stance", 2: "Strategic Stance", 3: "Value Alignment", 4: "Value Alignment"},
    "UAI": {1: "Value Alignment", 4: "Value Alignment", 2: "Strategic Action", 3: "Strategic Action"},
    "MAS": {1: "Value Alignment", 3: "Value Alignment", 2: "Strategic Stance", 4: "Strategic Stance"},
    "LTO": {1: "Strategic Stance", 4: "Strategic Stance", 2: "Value & Process", 3: "Value & Process"},
    "IND": {1: "Strategic Stance", 4: "Strategic Stance", 3: "Value & Action", 2: "Value & Action"},
}
LAYER = {
    "Linguistic Form": "L1 Form",
    "Value Alignment": "L2 Values",
    "Value & Process": "L2 Values",
    "Value & Action": "L2 Values",
    "Social Strategy": "L3 Action",
    "Strategic Stance": "L3 Action",
    "Strategic Action": "L3 Action",
}

RNG = np.random.default_rng(42)
N_BOOT = 2000


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def load():
    df = pd.read_csv(ROOT / "data" / "aggre_huaman_scores.csv")
    df["q"] = df["question_id"].str[1:].astype(int)
    df["category"] = [CATEGORY[d].get(q, "Overall Fit") for d, q in zip(df["dimension"], df["q"])]
    df["layer"] = df["category"].map(LAYER).fillna("Overall")
    return df


def audit(df):
    header("A. Data audit")
    per_rater = df.groupby("rater_id")
    print(f"rows={len(df):,}  raters={df['rater_id'].nunique():,}  missing scores={df['score'].isna().sum()}")
    print(f"rows per rater: {per_rater.size().value_counts().to_dict()}  (5 models x 5 questions)")
    print(f"dimensions per rater: {per_rater['dimension'].nunique().value_counts().to_dict()}")
    print("raters per dimension:", df.groupby("dimension")["rater_id"].nunique().reindex(DIMS).to_dict())
    dist = df["score"].value_counts(normalize=True).sort_index()
    print("score distribution:", {k: f"{v:.1%}" for k, v in dist.items()})
    straight = (per_rater["score"].nunique() == 1).sum()
    print(f"straight-liners (same score on all 25 items): {straight} ({straight / df['rater_id'].nunique():.1%})")
    print("\nNote: the CSV has no response/scenario ID, so ratings cannot be linked back to")
    print("individual generated texts or to the 5 scenarios within a dimension.")


def jwcas(q5):
    return q5.groupby(["model", "dimension"])["score"].mean().unstack()[DIMS].mean(axis=1)


def reproduce_jwcas(df):
    header("B. JWCAS recomputed from raw Q5 ratings")
    q5 = df[df["q"] == 5]
    dim_means = q5.groupby(["model", "dimension"])["score"].mean().unstack().reindex(MODELS)[DIMS]
    scores = dim_means.mean(axis=1)
    table = dim_means.copy()
    table["JWCAS"] = scores
    table["paper"] = pd.Series(PAPER_JWCAS)
    table["abs diff"] = (table["JWCAS"] - table["paper"]).abs()
    print(table.sort_values("JWCAS", ascending=False).round(3).to_string())
    assert (table["abs diff"] < 0.0005).all(), "JWCAS does not match the paper"
    print("\nAll five JWCAS values match the paper to 3 decimals.")
    print(f"Spread between best and worst model: {scores.max() - scores.min():.3f} points on a 1-5 scale")
    print(f"Pooled SD of a single Q5 rating: {q5['score'].std():.3f}")
    table.round(4).to_csv(OUT / "jwcas_by_dimension.csv")
    return q5, scores


def bootstrap_jwcas(q5):
    header("C. Rater-level bootstrap of JWCAS (resampling raters within each dimension)")
    wide = q5.pivot_table(index=["dimension", "rater_id"], columns="model", values="score")
    groups = {d: wide.loc[d].to_numpy() for d in DIMS}
    boots = np.empty((N_BOOT, len(MODELS)))
    for b in range(N_BOOT):
        dim_means = [g[RNG.integers(0, len(g), len(g))].mean(axis=0) for g in groups.values()]
        boots[b] = np.mean(dim_means, axis=0)
    cols = list(wide.columns)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    ranks = (-boots).argsort(axis=1).argsort(axis=1) + 1
    res = pd.DataFrame(
        {
            "JWCAS": wide.groupby(level=0).mean().mean().reindex(cols).values,
            "CI95 low": lo,
            "CI95 high": hi,
            "P(rank 1)": (ranks == 1).mean(axis=0),
            "mean rank": ranks.mean(axis=0),
        },
        index=cols,
    ).sort_values("JWCAS", ascending=False)
    print(res.round(3).to_string())
    res.round(4).to_csv(OUT / "jwcas_bootstrap.csv")
    return res


def icc_two_way(mat):
    """Shrout & Fleiss ICC(2,1) and ICC(2,k) for a complete targets x raters matrix."""
    n, k = mat.shape
    grand = mat.mean()
    ss_rows = k * ((mat.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((mat.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((mat - grand) ** 2).sum() - ss_rows - ss_cols
    msr, msc, mse = ss_rows / (n - 1), ss_cols / (k - 1), ss_err / ((n - 1) * (k - 1))
    icc1 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    icck = (msr - mse) / (msr + (msc - mse) / n)
    return icc1, icck, k


def reliability(df):
    header("D. Inter-rater reliability, implemented from the ANOVA formulas")
    rows = []
    for d in DIMS:
        sub = df[df["dimension"] == d]
        mat = sub.pivot_table(index=["model", "question_id"], columns="rater_id", values="score").to_numpy()
        icc1, icck, k = icc_two_way(mat)
        rows.append({"dimension": d, "raters (k)": k, "ICC(2,1) single rater": icc1, "ICC(2,k) mean of k raters": icck})
    res = pd.DataFrame(rows).set_index("dimension")
    print(res.round(3).to_string())
    print(f"\nMean ICC(2,k) = {res['ICC(2,k) mean of k raters'].mean():.3f}   (paper: 0.837)")
    print(f"Mean ICC(2,1) = {res['ICC(2,1) single rater'].mean():.3f}")
    print("\nInterpretation: the paper's 0.837 is the reliability of the *average* of ~290")
    print("raters. Any single rater agrees with the others only weakly; the signal emerges")
    print("because so many ratings are averaged.")
    res.round(4).to_csv(OUT / "icc_by_dimension.csv")


def mixed_model(q5):
    header("E. Linear mixed model  score ~ model + (1 | rater)  on Q5")
    data = q5.assign(model=pd.Categorical(q5["model"], categories=["GLM"] + [m for m in MODELS if m != "GLM"]))
    fit = smf.mixedlm("score ~ C(model)", data, groups=data["rater_id"]).fit(method="powell")
    params, cov = fit.params, fit.cov_params()

    def coef(m):
        name = f"C(model)[T.{m}]"
        return name if name in params else None

    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        ca, cb = coef(a), coef(b)
        diff = (params[ca] if ca else 0) - (params[cb] if cb else 0)
        var = (cov.loc[ca, ca] if ca else 0) + (cov.loc[cb, cb] if cb else 0)
        if ca and cb:
            var -= 2 * cov.loc[ca, cb]
        z = diff / np.sqrt(var)
        rows.append({"comparison": f"{a} vs {b}", "diff": diff, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    res["sig"] = res["p_fdr"] < 0.05
    res["d (diff / rating SD)"] = res["diff"] / q5["score"].std()
    print(res.sort_values("p_fdr").round(4).to_string(index=False))
    res.round(5).to_csv(OUT / "lmm_pairwise.csv", index=False)


def nonparametric(q5):
    header("F. Non-parametric check: Friedman + Wilcoxon signed-rank on Q5 (raters as blocks)")
    wide = q5.pivot_table(index="rater_id", columns="model", values="score")[MODELS]
    chi2, p = stats.friedmanchisquare(*[wide[m] for m in MODELS])
    print(f"Friedman chi2={chi2:.2f}, p={p:.2e}")
    rows = []
    for a, b in itertools.combinations(MODELS, 2):
        w = stats.wilcoxon(wide[a], wide[b], zero_method="wilcox")
        rows.append({"comparison": f"{a} vs {b}", "mean diff": (wide[a] - wide[b]).mean(), "p": w.pvalue,
                     "ties (%)": (wide[a] == wide[b]).mean() * 100})
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    res["sig"] = res["p_fdr"] < 0.05
    print(res.sort_values("p_fdr").round(4).to_string(index=False))
    print("\nThe 'ties' column shows how often a rater gave both models the same Q5 score.")


def straightliner_sensitivity(df):
    header("G. Sensitivity: drop raters who gave the same score to all 25 items")
    flat = df.groupby("rater_id")["score"].nunique() == 1
    kept = df[~df["rater_id"].map(flat)]
    full = jwcas(df[df["q"] == 5])
    trimmed = jwcas(kept[kept["q"] == 5])
    res = pd.DataFrame({"all raters": full, "without straight-liners": trimmed}).sort_values("all raters", ascending=False)
    print(f"dropped {flat.sum()} raters, kept {(~flat).sum()}")
    print(res.round(3).to_string())
    print("Ranking unchanged:", list(full.sort_values(ascending=False).index) == list(trimmed.sort_values(ascending=False).index))


def layers(df):
    header("H. Three-layer view (Form / Values / Action) from diagnostic items Q1-Q4")
    diag = df[df["q"] < 5]
    coverage = diag.groupby("layer").apply(
        lambda g: ", ".join(f"{d}:" + "+".join(f"Q{q}" for q in sorted(g[g['dimension'] == d]['q'].unique()))
                            for d in DIMS if d in set(g["dimension"])), include_groups=False)
    print("Which items feed each layer:")
    for layer, items in coverage.items():
        print(f"  {layer:10s} <- {items}")

    by_cat = diag.groupby(["layer", "category", "dimension", "model"])["score"].mean().unstack()[MODELS]
    print("\nCategory means (rows) by model (columns):")
    print(by_cat.round(3).to_string())

    layer_means = by_cat.groupby(level="layer").mean()
    print("\nLayer means (unweighted mean of the category x dimension cells above):")
    print(layer_means.round(3).to_string())
    print("\nWithin-model profile (layer mean minus that model's mean across layers):")
    print((layer_means - layer_means.mean()).round(3).to_string())
    by_cat.round(4).to_csv(OUT / "layer_category_means.csv")
    layer_means.round(4).to_csv(OUT / "layer_means.csv")
    return by_cat


def form_vs_strategy_test(df):
    header("I. Direct test of 'LLM-jp sounds right but acts less right' (PDI only)")
    print("PDI is the only dimension with both Form items (Q1, Q3) and Action items (Q2, Q4),")
    print("so it is the only place the claim can be tested within the same raters.\n")
    pdi = df[(df["dimension"] == "PDI") & (df["q"] < 5)]
    per = pdi.groupby(["rater_id", "model", "layer"])["score"].mean().unstack(["model", "layer"])
    rows = []
    for other in [m for m in MODELS if m != "LLM-jp"]:
        gap_form = per[("LLM-jp", "L1 Form")] - per[(other, "L1 Form")]
        gap_action = per[("LLM-jp", "L3 Action")] - per[(other, "L3 Action")]
        contrast = gap_form - gap_action
        t = stats.ttest_1samp(contrast, 0)
        rows.append({"vs": other, "LLM-jp minus other (Form)": gap_form.mean(),
                     "LLM-jp minus other (Action)": gap_action.mean(),
                     "interaction": contrast.mean(), "t": t.statistic, "p": t.pvalue})
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p"], method="fdr_bh")[1]
    print(res.round(4).to_string(index=False))
    print("\nA positive, significant interaction means LLM-jp's advantage (or smaller deficit)")
    print("on Form shrinks on Action, which is the paper's 'form > strategy' pattern.")
    res.round(5).to_csv(OUT / "pdi_form_vs_action_test.csv", index=False)


def paper_vs_code(df, q5):
    header("J. Checks of statements in the paper against the released data/code")

    fit = smf.mixedlm("score ~ C(model)", q5, groups=q5["rater_id"]).fit(method="powell")
    wald = fit.wald_test_terms(scalar=True).summary_frame().loc["C(model)"]
    chi2, df_ = float(wald["chi2"]), int(wald["df constraint"])
    print(f"1) Wald chi2 for model effect = {chi2:.2f} on {df_} df  ->  chi2/df = {chi2 / df_:.2f}")
    print("   Paper reports F(4, 8580) = 20.93. So the paper's F is the Wald chi2/df from a")
    print("   rater-only model; the paper text says scenario ID was also a random intercept,")
    print("   but the CSV has no scenario column and the notebook does not include it.")

    rater_var, resid_var = float(fit.cov_re.iloc[0, 0]), float(fit.scale)
    print(f"\n2) ICC from the LMM variance components = {rater_var:.3f} / ({rater_var:.3f} + {resid_var:.3f})"
          f" = {rater_var / (rater_var + resid_var):.3f}")
    print("   Paper says the 0.837 ICC was computed 'from the LMM's variance components', but")
    print("   0.837 actually comes from pingouin ICC2k averaged over dimensions (section D).")
    print("   The LMM quantity measures how much raters differ in overall leniency, not agreement.")

    print("\n3) Cronbach's alpha of the 5 items per dimension (rows = rater x model); the values")
    print("   are blank in the HTML version of the paper:")
    for d in DIMS:
        items = df[df["dimension"] == d].pivot_table(index=["rater_id", "model"], columns="q", values="score")
        k = items.shape[1]
        alpha = k / (k - 1) * (1 - items.var(ddof=1).sum() / items.sum(axis=1).var(ddof=1))
        print(f"   {d}: alpha = {alpha:.3f}")

    scores = jwcas(q5)
    print(f"\n4) Paper's Discussion mentions 'a 0.5-point difference on a 5-point scale' between top")
    print(f"   and lower models. Actual JWCAS range = {scores.max() - scores.min():.3f}.")


def figures(df, scores, boot, by_cat):
    order = scores.sort_values().index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(order, scores[order], color=[COLORS[m] for m in order], height=0.55)
    ax.errorbar(scores[order], order, xerr=[scores[order] - boot.loc[order, "CI95 low"],
                                            boot.loc[order, "CI95 high"] - scores[order]],
                fmt="none", ecolor="black", capsize=4)
    for m in order:
        ax.text(boot.loc[m, "CI95 high"] + 0.01, m, f"{scores[m]:.3f}", va="center", fontsize=10)
    ax.set_xlim(3.0, 3.7)
    ax.set_xlabel("JWCAS (mean of 6 dimension means of Q5), 95% rater-bootstrap CI")
    ax.set_title("Reproduced JWCAS from released human ratings")
    fig.tight_layout()
    fig.savefig(OUT / "fig_jwcas.png", dpi=160)
    plt.close(fig)

    q5 = df[df["q"] == 5].groupby(["model", "dimension"])["score"].mean().unstack().reindex(MODELS)[DIMS]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(q5.values, cmap="RdYlGn", vmin=3.05, vmax=3.75)
    ax.set_xticks(range(len(DIMS)), DIMS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    for i in range(len(MODELS)):
        for j in range(len(DIMS)):
            ax.text(j, i, f"{q5.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean Q5")
    ax.set_title("Overall-fit (Q5) by model and Hofstede dimension")
    fig.tight_layout()
    fig.savefig(OUT / "fig_q5_heatmap.png", dpi=160)
    plt.close(fig)

    layer_means = by_cat.groupby(level="layer").mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(layer_means.index))
    for m in MODELS:
        ax.plot(x, layer_means[m], marker="o", lw=2, color=COLORS[m], label=m)
    ax.set_xticks(x, ["Layer 1\nForm (PDI only)", "Layer 2\nValues", "Layer 3\nAction"])
    ax.set_ylabel("mean rating (Q1-Q4 items)")
    ax.set_title("Three-layer profile per model")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_layers.png", dpi=160)
    plt.close(fig)


def main():
    sys.stdout = Tee(OUT / "independent_analysis.txt")
    df = load()
    audit(df)
    q5, scores = reproduce_jwcas(df)
    boot = bootstrap_jwcas(q5)
    reliability(df)
    mixed_model(q5)
    nonparametric(q5)
    straightliner_sensitivity(df)
    by_cat = layers(df)
    form_vs_strategy_test(df)
    paper_vs_code(df, q5)
    figures(df, scores, boot, by_cat)
    print(f"\nOutputs written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
