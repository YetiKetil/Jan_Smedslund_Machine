"""
Jan Smedslund Semantic Predetermination Detector
=================================================
Developed by Ketil Arnulf · BI Norwegian Business School
In memory of Jan Smedslund (1929–2026)

Run with:  streamlit run smedslund_app.py
"""

import streamlit as st
import anthropic
from openai import OpenAI
import base64
import json
import re
import copy
import io
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Jan Smedslund Semantic Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLAUDE_MODEL       = "claude-opus-4-5"
SMEDSLUND_R2_BENCH = 0.428   # Theoretically derived benchmark

# ── Prompts — verbatim from stage1_theory_v7.ipynb ───────────────────────────

THEORY_PROMPT = """
You are a theory extraction agent for a semantic replication pipeline.
The pipeline tests whether argued relationships between psychological
constructs are semantically predetermined — predictable from the
conceptual language of construct definitions alone.

=== STEP 1: ELIGIBILITY — CHECK THESE FIRST ===

Set eligible=false and stop if ANY of these apply:

INELIGIBLE — META-ANALYSIS:
  Paper aggregates effect sizes across multiple prior studies.
  Signals: 'meta-analysis', 'k = N studies', 'studies were coded',
  'pooled effect', 'publication bias', 'forest plot', 'funnel plot'.
  Reason: aggregated effects are not fresh observations against a theory.

INELIGIBLE — CONCEPTUAL/REVIEW:
  Paper proposes a theoretical model or reviews literature without
  collecting new empirical data. Signals: 'we propose a framework',
  'conceptual model', 'theoretical contribution', 'typology',
  'no data were collected', 'this paper reviews'.

INELIGIBLE — PURE VALIDATION:
  Paper develops or validates a measurement scale without testing
  structural hypotheses between distinct constructs. A validation
  paper that ALSO tests hypotheses about relationships between
  constructs may still be eligible.
  Signals: 'scale development', 'psychometric properties',
  'confirmatory factor analysis', 'convergent validity'
  WITHOUT any directional between-construct predictions.

INELIGIBLE — TOO FEW RELATIONSHIPS:
  Paper reports fewer than 3 directional hypothesised relationships
  between distinct constructs. Single-predictor studies, studies
  with only one outcome, and case studies are ineligible.

INELIGIBLE — NO EFFECT SIZES:
  Paper reports no quantitative effect sizes (betas, correlations,
  path coefficients, d, eta-squared). Qualitative studies and
  papers reporting only significance without magnitude are ineligible.

=== STEP 2: EXTRACTION RULES (if eligible) ===

RULE 1 — CONSTRUCT DEFINITIONS:
  Extract from THEORETICAL sections only: introduction, theory,
  hypothesis development. Do NOT use the methods/measures section
  as the primary source — that describes operationalisation, not
  theoretical meaning. Write 3-6 sentences capturing what the
  construct MEANS in the authors' theoretical argument.

RULE 2 — HYPOTHESISED RELATIONSHIPS ONLY:
  Extract only relationships the authors explicitly hypothesise.
  Not all correlations. Not all paths in a saturated model.
  The hypothesis is the theoretical claim being tested.

RULE 3 — SIGNED EFFECT SIZES:
  Preserve signs. β=-0.33 stays -0.33. Use Step 1 betas for
  hierarchical regressions (unconfounded direct estimate).
  Record regression_step as integer (1, 2, 3...) or null.

RULE 4 — MEDIATION CHAINS:
  Explicitly identify A→B→C triples where B is the argued mediator.
  All three nodes must be distinct constructs.

RULE 5 — R² SEPARATE FROM PATHS:
  relationships: ONLY between-construct directional effects (from!=to)
  explained_variances: ONLY R² for a single construct

RULE 6 — ITEM AVAILABILITY:
  Flag item_availability per construct but do NOT extract items here.
  Values: 'full_in_paper' | 'public_domain_scale' |
          'proprietary_scale' | 'not_available'

=== END RULES ===

RESPOND WITH ONLY valid JSON. No markdown fences.

{
  "study_metadata": {
    "title": "",
    "authors": "",
    "year": "",
    "journal": "",
    "study_type": "survey|experimental|mixed|unclear",
    "n_respondents": null,
    "model_type": "direct_only|mediation|moderation|mediated_moderation|other",
    "design_notes": ""
  },
  "eligibility": {
    "eligible": true,
    "exclusion_reason": "",
    "exclusion_category": "",
    "n_constructs": 0,
    "n_hypotheses": 0,
    "has_mediation_chains": false,
    "semantic_promise": "high|medium|low",
    "semantic_promise_reason": ""
  },
  "constructs": [
    {
      "name": "",
      "role": "predictor|mediator|outcome|moderator",
      "theoretical_definition": "",
      "definition_source": "introduction|theory_section|hypothesis_section|inferred",
      "item_availability": "full_in_paper|public_domain_scale|proprietary_scale|not_available",
      "scale_name": "",
      "is_composite": false
    }
  ],
  "hypotheses": [
    {
      "id": "H1",
      "text": "",
      "from": "",
      "to": "",
      "direction": "positive|negative|unspecified",
      "supported": true
    }
  ],
  "mediation_chains": [
    {
      "predictor": "",
      "mediator": "",
      "outcome": "",
      "type": "full|partial|tested_not_supported",
      "hypothesis_ref": ""
    }
  ],
  "relationships": [
    {
      "from": "",
      "to": "",
      "hypothesis_id": "",
      "effect_size": null,
      "effect_type": "beta|path_coefficient|correlation|other",
      "regression_step": null,
      "group": "pooled",
      "significant": true
    }
  ],
  "explained_variances": [
    {"construct": "", "r_squared": null,
     "regression_step": null, "group": "pooled"}
  ],
  "theoretical_notes": ""
}
"""

PASS2_TEMPLATE = """
You previously extracted construct definitions from a paper.
Now extract ONLY relationships, explained_variances, mediation_chains,
and hypotheses.

Paper: {title} ({authors}, {year})
Constructs: {construct_names}

RULES:
- relationships: hypothesised between-construct effects only (from != to)
- Preserve signs. Tag regression_step for hierarchical regressions.
- mediation_chains: A→B→C where B is the argued mediator, all distinct.
- explained_variances: R² for single constructs only, not paths.
- Minimum 3 relationships must exist for the paper to contribute.

RESPOND WITH ONLY valid JSON:
{{"hypotheses":[],"mediation_chains":[],"relationships":[],
  "explained_variances":[],"theoretical_notes":""}}
"""

RECOVERY_TEMPLATE = """
Your response was cut off. Received:
---
{truncated}
---
Continue from exactly where it cut off. Output ONLY the continuation.
No markdown fences.
"""

# ── Stage 0: Pre-screening (no API calls) ────────────────────────────────────

META_SIGNALS = [
    'meta-analysis', 'meta analysis', 'k = ', 'studies were coded',
    'pooled effect', 'publication bias', 'forest plot', 'funnel plot',
    'systematic review', 'heterogeneity', 'moderator analysis'
]
CONCEPTUAL_SIGNALS = [
    'we propose a framework', 'conceptual model', 'theoretical contribution',
    'typology', 'no data were collected', 'this paper reviews',
    'literature review', 'conceptual paper', 'review article'
]

def prescreening(pdf_bytes):
    """Quick text-based pre-screen. Returns (verdict, detail_message)."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages[:6]:
            text += page.extract_text() or ""
        text_lower = text.lower()

        if len(text) < 300:
            return "POSSIBLY_SCANNED", (
                f"Only {len(text)} characters extracted from first 6 pages. "
                "May be scanned — Claude will read via vision."
            )

        meta_hits = sum(1 for s in META_SIGNALS if s in text_lower)
        if meta_hits >= 2:
            return "LIKELY_EXCLUDE", (
                f"Meta-analysis signals detected ({meta_hits} keyword hits). "
                "Claude will make the final eligibility call."
            )

        conceptual_hits = sum(1 for s in CONCEPTUAL_SIGNALS if s in text_lower)
        if conceptual_hits >= 2:
            return "LIKELY_EXCLUDE", (
                f"Conceptual/review signals detected ({conceptual_hits} keyword hits). "
                "Claude will make the final eligibility call."
            )

        return "ELIGIBLE", (
            f"Text extractable ({len(text):,} chars from first 6 pages). "
            "Sending to Claude for full extraction."
        )
    except Exception as e:
        return "UNKNOWN", f"Pre-screen could not run ({e}). Proceeding to Claude."


# ── Stage 1: Claude theory extraction ────────────────────────────────────────

def _clean(raw):
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()

def _truncated(raw):
    return not raw.rstrip().endswith("}")

def _api_pdf(client, pdf_b64, prompt, max_tokens):
    return client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": [
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf",
                        "data": pdf_b64}},
            {"type": "text", "text": prompt}
        ]}]
    )

def _api_text(client, prompt, max_tokens):
    return client.messages.create(
        model=CLAUDE_MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )

def _recover(client, raw):
    for _ in range(2):
        cont = _clean(_api_text(
            client,
            RECOVERY_TEMPLATE.format(truncated=raw[-2000:]),
            4096).content[0].text)
        raw = raw.rstrip() + "\n" + cont
        if not _truncated(raw):
            break
    return raw

def _pass2(client, pdf_b64, partial, log):
    log("Running pass 2 — extracting relationships and chains...")
    names  = ", ".join(c["name"] for c in partial.get("constructs", []))
    meta   = partial.get("study_metadata", {})
    prompt = PASS2_TEMPLATE.format(
        title=meta.get("title", ""),
        authors=meta.get("authors", ""),
        year=meta.get("year", ""),
        construct_names=names
    )
    raw = _clean(_api_pdf(client, pdf_b64, prompt, 8192).content[0].text)
    if _truncated(raw):
        raw = _recover(client, raw)
    p2 = json.loads(raw)
    for k in ("hypotheses", "mediation_chains", "relationships",
               "explained_variances", "theoretical_notes"):
        if k in p2:
            partial[k] = p2[k]
    n_r  = len(partial.get("relationships", []))
    n_ch = len(partial.get("mediation_chains", []))
    log(f"Pass 2 merged: {n_r} relationships, {n_ch} chains.")
    return partial

def validate_theory(result):
    errors, warnings = [], []
    if not result.get("eligibility", {}).get("eligible", True):
        return True, [], []
    for i, r in enumerate(result.get("relationships", [])):
        if r.get("from") == r.get("to"):
            errors.append(f"relationships[{i}]: self-loop (from == to). R² values → explained_variances.")
    for i, ch in enumerate(result.get("mediation_chains", [])):
        nodes = {ch.get("predictor"), ch.get("mediator"), ch.get("outcome")}
        if len(nodes) < 3:
            errors.append(f"mediation_chains[{i}]: predictor/mediator/outcome not all distinct.")
    n_rels = len(result.get("relationships", []))
    if 0 < n_rels < 3:
        warnings.append(
            f"Only {n_rels} relationships extracted — below the minimum of 3. "
            "Paper cannot contribute to the A>B concordance test."
        )
    for c in result.get("constructs", []):
        w = len(c.get("theoretical_definition", "").split())
        if w < 15:
            warnings.append(f"Thin definition ({w} words): {c['name']}")
    if result.get("eligibility", {}).get("eligible") and not result.get("relationships"):
        warnings.append("Eligible paper but no relationships extracted.")
    return len(errors) == 0, warnings, errors

def extract_theory(pdf_bytes, anthropic_key, log):
    """Full Stage 1 extraction. Returns theory dict."""
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode()
    client  = anthropic.Anthropic(api_key=anthropic_key)
    result  = None

    for attempt, max_tok in enumerate([4096, 8192], start=1):
        log(f"Claude extraction — attempt {attempt} ({max_tok} max tokens)...")
        raw = _clean(_api_pdf(client, pdf_b64, THEORY_PROMPT, max_tok).content[0].text)

        if _truncated(raw):
            log(f"Response truncated ({len(raw)} chars) — recovering...")
            raw = _recover(client, raw)

        try:
            result = json.loads(raw)
            log(f"JSON parsed successfully ({len(raw):,} chars).")
            break
        except json.JSONDecodeError as e:
            log(f"JSON parse error: {e}")
            if attempt == 2:
                try:
                    partial = json.loads(raw[:raw.rfind("}") + 1])
                    result  = _pass2(client, pdf_b64, partial, log)
                    break
                except Exception as e2:
                    raise ValueError(f"Cannot parse Claude response after two attempts: {e2}")

    # Proactive pass 2 if eligible but no relationships came back
    if (result is not None
            and not result.get("relationships")
            and result.get("eligibility", {}).get("eligible")):
        log("Eligible paper with no relationships — triggering pass 2...")
        result = _pass2(client, pdf_b64, result, log)

    _, warnings, errors = validate_theory(result)
    result["_validation"] = {"warnings": warnings, "errors": errors}
    return result


# ── Stage 2: Embeddings and cosine analysis ───────────────────────────────────

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def get_embeddings(texts, openai_key):
    client = OpenAI(api_key=openai_key)
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=texts
    )
    return np.array([item.embedding for item in response.data])

def run_stage2(theory, openai_key, log):
    """Embed definitions, compute cosines, run all three criteria."""
    constructs    = theory.get("constructs", [])
    relationships = theory.get("relationships", [])
    chains        = theory.get("mediation_chains", [])

    if len(constructs) < 2:
        return None, "Fewer than 2 constructs — cosine analysis impossible."

    names = [c["name"] for c in constructs]
    defs  = [c.get("theoretical_definition", c["name"]) for c in constructs]

    # Use cached cosine matrix from stored retrieval if available
    _cache = theory.get("_stage2_cache", {})
    if (_cache
            and _cache.get("cosine_matrix")
            and len(_cache.get("constructs", [])) == len(names)):
        log("Restoring cosine matrix from database cache — no embedding call needed.")
        cos_mat = np.array(_cache["cosine_matrix"])
        _use_cache = True
    else:
        log(f"Fetching embeddings for {len(names)} constructs (text-embedding-3-large)...")
        embeddings = get_embeddings(defs, openai_key)
        _use_cache = False

    # Full cosine matrix
    n = len(names)
    if not _use_cache:
        cos_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cos_mat[i][j] = _cosine(embeddings[i], embeddings[j])
        log("Cosine matrix computed.")

    name_idx = {name: i for i, name in enumerate(names)}

    # ── Valid pairs: hypothesised relationships with effect sizes ─────────
    valid_rels = [
        r for r in relationships
        if r.get("from") and r.get("to")
        and r["from"] != r["to"]
        and r.get("effect_size") is not None
        and name_idx.get(r["from"]) is not None
        and name_idx.get(r["to"]) is not None
    ]

    pair_data = [{
        "from":           r["from"],
        "to":             r["to"],
        "cosine":         cos_mat[name_idx[r["from"]]][name_idx[r["to"]]],
        "signed_effect":  float(r["effect_size"]),
        "unsigned_effect": abs(float(r["effect_size"])),
        "effect_type":    r.get("effect_type", ""),
        "step":           r.get("regression_step")
    } for r in valid_rels]

    # ── Criterion 2: A>B concordance ─────────────────────────────────────
    log("Computing A>B concordance...")
    concordant = discordant = tied = 0
    for p1, p2 in combinations(range(len(pair_data)), 2):
        a, b = pair_data[p1], pair_data[p2]
        if a["cosine"] == b["cosine"] or a["unsigned_effect"] == b["unsigned_effect"]:
            tied += 1
        elif (a["cosine"] > b["cosine"]) == (a["unsigned_effect"] > b["unsigned_effect"]):
            concordant += 1
        else:
            discordant += 1

    total_ab = concordant + discordant
    ab_rate  = concordant / total_ab if total_ab > 0 else None

    # ── Criterion 3: A>B>C mediation gradient ────────────────────────────
    log("Computing A>B>C mediation gradient...")
    abc_results = []
    for ch in chains:
        pred, med, out = ch.get("predictor"), ch.get("mediator"), ch.get("outcome")
        pi, mi, oi = name_idx.get(pred), name_idx.get(med), name_idx.get(out)
        if None in (pi, mi, oi):
            continue
        cos_ab = cos_mat[pi][mi]   # predictor → mediator
        cos_bc = cos_mat[mi][oi]   # mediator  → outcome
        cos_ac = cos_mat[pi][oi]   # predictor → outcome (distal)
        passes = (cos_ab > cos_ac) and (cos_bc > cos_ac)
        abc_results.append({
            "chain":     f"{pred} → {med} → {out}",
            "predictor": pred, "mediator": med, "outcome": out,
            "cos_ab": float(cos_ab), "cos_bc": float(cos_bc), "cos_ac": float(cos_ac),
            "passes": passes,
            "chain_type": ch.get("type", "")
        })

    abc_pass  = sum(1 for r in abc_results if r["passes"])
    abc_total = len(abc_results)
    abc_rate  = abc_pass / abc_total if abc_total > 0 else None

    # ── Criterion 1: Within-study Spearman ───────────────────────────────
    log("Computing within-study Spearman correlations...")
    signed_rho = signed_p = unsigned_rho = unsigned_p = None
    n_pairs = len(pair_data)
    cosine_range = None

    # For Spearman/concordance, exclude pairs with |effect| > BETA_CEILING
    pair_data_filtered = [p for p in pair_data
                          if abs(p.get("unsigned_effect", 0)) <= BETA_CEILING]
    n_pairs_filtered   = len(pair_data_filtered)

    if n_pairs_filtered >= 3:
        cosines          = [p["cosine"] for p in pair_data_filtered]
        signed_effects   = [p["signed_effect"] for p in pair_data_filtered]
        unsigned_effects = [p["unsigned_effect"] for p in pair_data_filtered]
        cosine_range     = max(cosines) - min(cosines)

        sr = stats.spearmanr(cosines, signed_effects)
        ur = stats.spearmanr(cosines, unsigned_effects)
        signed_rho,   signed_p   = float(sr.statistic), float(sr.pvalue)
        unsigned_rho, unsigned_p = float(ur.statistic), float(ur.pvalue)
        log(f"Spearman: signed ρ={signed_rho:.3f} (p={signed_p:.3f}), "
            f"unsigned ρ={unsigned_rho:.3f} (p={unsigned_p:.3f}), n={n_pairs} pairs.")
    elif n_pairs < 3:
        log(f"Only {n_pairs} pairs — Spearman not computed (need ≥3).")
    else:
        log(f"Only {n_pairs_filtered} pairs after |β| > {BETA_CEILING} filter — Spearman not computed.")

    # ── Explained variance ────────────────────────────────────────────────
    r2_vals = [ev["r_squared"] for ev in theory.get("explained_variances", [])
               if ev.get("r_squared") is not None]
    avg_r2  = float(np.mean(r2_vals)) if r2_vals else None

    log("Stage 2 complete.")
    return {
        "constructs":    names,
        "cosine_matrix": cos_mat.tolist(),
        "cosine_range":  float(cosine_range) if cosine_range is not None else None,
        "pair_data":     pair_data,
        "ab": {
            "concordant":        concordant,
            "discordant":        discordant,
            "tied":              tied,
            "rate":              float(ab_rate) if ab_rate is not None else None,
            "total_comparisons": total_ab
        },
        "abc": {
            "results":   abc_results,
            "pass":      abc_pass,
            "total":     abc_total,
            "rate":      float(abc_rate) if abc_rate is not None else None
        },
        "spearman": {
            "signed_rho":   signed_rho,
            "signed_p":     signed_p,
            "unsigned_rho": unsigned_rho,
            "unsigned_p":   unsigned_p,
            "n_pairs":      n_pairs
        },
        "avg_r2": avg_r2
    }


# ── Verdict logic ─────────────────────────────────────────────────────────────

# Semantic inflation thresholds
INFLATION_COSINE_THRESHOLD = 0.50   # mean cosine above which uniform predetermination is suspected
INFLATION_BETA_THRESHOLD   = 0.30   # mean |beta| above which elevated effects confirm inflation
BETA_CEILING               = 2.0    # |beta| above this is almost certainly non-standardised
                                    # Standardised coefficients cannot genuinely exceed ±1.0;
                                    # values above 2.0 indicate odds ratios, unstandardised OLS,
                                    # or logistic betas on a different scale. Excluded from
                                    # dashboard displays; retained in raw database.

def compute_verdict(stage2):
    """
    Three ordinal signals; each above threshold scores 1 point.
    2–3 signals → Semantically Structured
    1 signal     → Partially Structured
    0 signals    → Empirically Independent — unless semantic inflation criteria apply
    0 signals + high mean cosine + high mean |beta| → Semantic Inflation

    Semantic inflation: the ordinal within-paper test returns no signal because
    all construct pairs are so uniformly semantically similar that there is no
    variance for the ranking test to detect. The elevated mean effect confirms
    predetermination at the absolute level rather than the ordinal level.
    """
    ab_rate  = stage2["ab"]["rate"]
    abc_rate = stage2["abc"]["rate"]
    rho      = stage2["spearman"]["signed_rho"]
    n_pairs  = stage2["spearman"]["n_pairs"]

    # Compute mean cosine and mean |beta| from pair data for inflation check
    pair_data    = stage2.get("pair_data", [])
    mean_cosine  = float(np.mean([p["cosine"] for p in pair_data])) if pair_data else None
    mean_abs_beta = float(np.mean([p["unsigned_effect"] for p in pair_data])) if pair_data else None

    signals = 0
    reasons = []

    if ab_rate is not None:
        if ab_rate >= 0.60:
            signals += 1
            reasons.append(f"A>B concordance {100*ab_rate:.1f}% ≥ 60% threshold")
        else:
            reasons.append(f"A>B concordance {100*ab_rate:.1f}% < 60% threshold")

    if abc_rate is not None:
        if abc_rate >= 0.55:
            signals += 1
            reasons.append(f"A>B>C pass rate {100*abc_rate:.1f}% ≥ 55% threshold")
        else:
            reasons.append(f"A>B>C pass rate {100*abc_rate:.1f}% < 55% threshold")

    if rho is not None and n_pairs >= 5:
        if rho > 0:
            signals += 1
            reasons.append(f"Within-study ρ = {rho:.3f} > 0 (n={n_pairs} pairs)")
        else:
            reasons.append(f"Within-study ρ = {rho:.3f} ≤ 0 (n={n_pairs} pairs)")
    elif n_pairs < 5:
        reasons.append(f"Within-study Spearman not scored ({n_pairs} pairs, need ≥5)")

    # Check for semantic inflation before assigning final label
    inflation = (
        signals < 2
        and mean_cosine is not None
        and mean_abs_beta is not None
        and mean_cosine >= INFLATION_COSINE_THRESHOLD
        and mean_abs_beta >= INFLATION_BETA_THRESHOLD
    )

    if signals >= 2:
        label, icon, color = "SEMANTICALLY STRUCTURED", "🔴", "#ef4444"
    elif inflation:
        label, icon, color = "SEMANTIC INFLATION",      "🟣", "#a855f7"
        reasons.append(
            f"Semantic inflation detected: mean cosine {mean_cosine:.3f} ≥ {INFLATION_COSINE_THRESHOLD} "
            f"and mean |β| {mean_abs_beta:.3f} ≥ {INFLATION_BETA_THRESHOLD}. "
            f"Uniform predetermination suspected — ordinal test has no variance to detect."
        )
    elif signals == 1:
        label, icon, color = "PARTIALLY STRUCTURED",   "🟡", "#f59e0b"
    else:
        label, icon, color = "EMPIRICALLY INDEPENDENT","🟢", "#22c55e"

    return label, icon, color, signals, reasons, mean_cosine, mean_abs_beta


# ── Plots ─────────────────────────────────────────────────────────────────────

_DARK_BG  = "#1e1e2e"
_FONT_COL = "#e0e0e0"
_BASE_LAYOUT = dict(
    paper_bgcolor=_DARK_BG, plot_bgcolor=_DARK_BG,
    font=dict(color=_FONT_COL), margin=dict(l=20, r=20, t=45, b=20)
)

def plot_cosine_heatmap(names, cos_mat):
    mat = np.array(cos_mat)
    fig = go.Figure(go.Heatmap(
        z=mat, x=names, y=names,
        colorscale="Blues", zmin=0, zmax=1,
        text=[[f"{mat[i][j]:.3f}" for j in range(len(names))]
              for i in range(len(names))],
        texttemplate="%{text}", textfont={"size": 11},
        showscale=True
    ))
    fig.update_layout(
        title="Cosine Similarity Matrix — Definition Level",
        height=max(350, 60 * len(names)),
        xaxis=dict(tickangle=-35),
        **_BASE_LAYOUT
    )
    return fig

def plot_ab_bar(ab):
    c, d, t = ab["concordant"], ab["discordant"], ab["tied"]
    total   = c + d + t
    fig = go.Figure(go.Bar(
        x=["Concordant", "Discordant", "Tied"],
        y=[c, d, t],
        marker_color=["#4ade80", "#f87171", "#94a3b8"],
        text=[f"{v}<br>({100*v/total:.0f}%)" if total else "0" for v in [c, d, t]],
        textposition="auto"
    ))
    if total > 0:
        fig.add_hline(
            y=total * 0.5, line_dash="dash", line_color="#94a3b8",
            annotation_text="50% chance baseline", annotation_font_color="#94a3b8"
        )
    rate_str = f"{100*ab['rate']:.1f}%" if ab["rate"] is not None else "—"
    fig.update_layout(
        title=f"A>B Concordance: {rate_str}  ({ab['total_comparisons']} pair-comparisons)",
        height=300, showlegend=False, **_BASE_LAYOUT
    )
    return fig

def plot_scatter(pair_data):
    if not pair_data:
        return None
    cosines = [p["cosine"] for p in pair_data]
    effects = [p["signed_effect"] for p in pair_data]
    labels  = [f"{p['from']} → {p['to']}" for p in pair_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cosines, y=effects, mode="markers+text",
        text=labels, textposition="top center",
        textfont=dict(size=9, color=_FONT_COL),
        marker=dict(size=10, color="#60a5fa", line=dict(width=1, color="#1e40af"))
    ))
    if len(cosines) >= 3:
        m, b = np.polyfit(cosines, effects, 1)
        xs = np.linspace(min(cosines), max(cosines), 50)
        fig.add_trace(go.Scatter(
            x=xs, y=m * xs + b, mode="lines",
            line=dict(color="#f59e0b", dash="dash", width=1.5),
            name="OLS trend"
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#475569")
    fig.update_layout(
        title="Cosine Similarity vs Signed Effect Size (β)",
        xaxis_title="Cosine similarity", yaxis_title="Signed β",
        height=350, showlegend=False, **_BASE_LAYOUT
    )
    return fig

def plot_abc_chains(abc_results):
    labels = [r["chain"] for r in abc_results]
    cos_ab = [r["cos_ab"] for r in abc_results]
    cos_bc = [r["cos_bc"] for r in abc_results]
    cos_ac = [r["cos_ac"] for r in abc_results]
    colors = ["#4ade80" if r["passes"] else "#f87171" for r in abc_results]

    fig = go.Figure()
    # Shade each chain by pass/fail
    for i, r in enumerate(abc_results):
        col = "rgba(74,222,128,0.2)" if r["passes"] else "rgba(248,113,113,0.2)"
        fig.add_vrect(x0=i - 0.4, x1=i + 0.4,
                      fillcolor=col, opacity=1.0, line_width=0)

    fig.add_trace(go.Bar(name="cos(A,B) pred→med",
                         x=labels, y=cos_ab, marker_color="#60a5fa"))
    fig.add_trace(go.Bar(name="cos(B,C) med→out",
                         x=labels, y=cos_bc, marker_color="#818cf8"))
    fig.add_trace(go.Bar(name="cos(A,C) distal",
                         x=labels, y=cos_ac, marker_color="#94a3b8"))

    fig.update_layout(
        title="A→B→C Cosine Gradients  (green bg = PASS, red bg = FAIL)",
        barmode="group", height=350,
        xaxis=dict(tickangle=-20), **_BASE_LAYOUT
    )
    return fig

def plot_r2_gauge(avg_r2):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=avg_r2,
        number={"suffix": "", "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": _FONT_COL},
            "bar": {"color": "#60a5fa"},
            "bgcolor": "#334155",
            "steps": [
                {"range": [0, SMEDSLUND_R2_BENCH], "color": "#1e293b"},
                {"range": [SMEDSLUND_R2_BENCH, 1], "color": "#164e63"}
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.85,
                "value": SMEDSLUND_R2_BENCH
            }
        },
        title={"text": f"Mean Empirical R²<br><sub>Smedslund benchmark = {SMEDSLUND_R2_BENCH}</sub>",
               "font": {"color": _FONT_COL}}
    ))
    fig.update_layout(height=250, paper_bgcolor=_DARK_BG,
                      font=dict(color=_FONT_COL), margin=dict(l=20, r=20, t=10, b=10))
    return fig



# ── Database accumulation ────────────────────────────────────────────────────
#
# Strategy:
#   1. If Supabase credentials are present (st.secrets or env vars) → Supabase.
#      This is the case when running on Streamlit Community Cloud.
#   2. Otherwise → local CSV files in the same folder as this script.
#      This is the case when running locally on your Mac.
#
# Both paths are duplicate-safe. The local CSV format exactly matches
# pooled_db_pathB.csv and batch_theory_summary.csv from the Jupyter pipeline.

import os

# ── Supabase client (lazy, cached) ────────────────────────────────────────────

def _get_supabase():
    """Return a Supabase client if credentials are available, else None."""
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL") or os.environ.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY") or os.environ.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None

# ── Local CSV fallback ────────────────────────────────────────────────────────

DB_DIR         = os.path.dirname(os.path.abspath(__file__))
POOLED_DB_PATH = os.path.join(DB_DIR, "pooled_db_pathB.csv")
SUMMARY_PATH   = os.path.join(DB_DIR, "batch_theory_summary.csv")

POOLED_COLS = [
    "study_id", "year", "construct_a", "construct_b",
    "cosine", "signed_effect", "unsigned_effect", "path_type", "source_file"
]
SUMMARY_COLS = [
    "file", "year", "authors", "title", "journal", "study_type", "n",
    "model_type", "n_constructs", "n_chains", "n_pairs", "cosine_range",
    "ab_concordant", "ab_discordant", "ab_rate",
    "abc_pass", "abc_total", "abc_rate",
    "signed_rho", "signed_p", "unsigned_rho", "unsigned_p",
    "avg_empirical_r2", "status"
]

def _load_or_create(path, columns):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=columns)

# ── Shared record builders ────────────────────────────────────────────────────

def _build_pair_rows(theory, stage2, source_filename, study_id, year):
    return [{
        "study_id":        study_id,
        "year":            year,
        "construct_a":     p["from"],
        "construct_b":     p["to"],
        "cosine":          round(p["cosine"], 6),
        "signed_effect":   p["signed_effect"],
        "unsigned_effect": p["unsigned_effect"],
        "path_type":       "B",
        "source_file":     source_filename
    } for p in stage2.get("pair_data", [])]

def _build_summary_row(theory, stage2, source_filename, year, authors):
    meta = theory.get("study_metadata", {})
    ab   = stage2.get("ab", {})
    abc  = stage2.get("abc", {})
    sp   = stage2.get("spearman", {})
    return {
        "file":             source_filename,
        "year":             year,
        "authors":          authors,
        "title":            meta.get("title", ""),
        "journal":          meta.get("journal", ""),
        "study_type":       meta.get("study_type", ""),
        "n":                meta.get("n_respondents"),
        "model_type":       meta.get("model_type", ""),
        "n_constructs":     len(theory.get("constructs", [])),
        "n_chains":         len(theory.get("mediation_chains", [])),
        "n_pairs":          sp.get("n_pairs", 0),
        "cosine_range":     round(stage2.get("cosine_range") or 0, 4),
        "ab_concordant":    ab.get("concordant", 0),
        "ab_discordant":    ab.get("discordant", 0),
        "ab_rate":          round(ab["rate"], 4) if ab.get("rate") is not None else None,
        "abc_pass":         abc.get("pass", 0),
        "abc_total":        abc.get("total", 0),
        "abc_rate":         round(abc["rate"], 4) if abc.get("rate") is not None else None,
        "signed_rho":       round(sp["signed_rho"], 4) if sp.get("signed_rho") is not None else None,
        "signed_p":         round(sp["signed_p"], 4) if sp.get("signed_p") is not None else None,
        "unsigned_rho":     round(sp["unsigned_rho"], 4) if sp.get("unsigned_rho") is not None else None,
        "unsigned_p":       round(sp["unsigned_p"], 4) if sp.get("unsigned_p") is not None else None,
        "avg_empirical_r2": round(stage2["avg_r2"], 4) if stage2.get("avg_r2") is not None else None,
        "mean_cosine":      round(float(np.mean([p["cosine"] for p in stage2.get("pair_data",[])])), 4)
                            if stage2.get("pair_data") else None,
        "mean_abs_beta":    round(float(np.mean([p["unsigned_effect"] for p in stage2.get("pair_data",[])])), 4)
                            if stage2.get("pair_data") else None,
        "status":           "analysed"
    }

# ── Supabase save ─────────────────────────────────────────────────────────────

def _save_supabase(client, theory, stage2, source_filename, study_id, year, authors):
    """Insert into Supabase. Returns (pairs_added, already_existed)."""
    # Check duplicate in paper_summary
    existing = (client.table("paper_summary")
                .select("id")
                .eq("file", source_filename)
                .execute())
    already_existed = len(existing.data) > 0

    pairs_added = 0
    if not already_existed:
        # Insert summary row
        summary_row = _build_summary_row(theory, stage2, source_filename, year, authors)
        client.table("paper_summary").insert(summary_row).execute()

        # Insert pair rows
        pair_rows = _build_pair_rows(theory, stage2, source_filename, study_id, year)
        if pair_rows:
            for i in range(0, len(pair_rows), 100):
                client.table("pooled_pairs").insert(pair_rows[i:i+100]).execute()
            pairs_added = len(pair_rows)

    # Always try to save theory JSON (even if summary/pairs already existed)
    _save_theory_supabase(client, theory, source_filename, study_id, year, authors,
                          stage2=stage2)

    return pairs_added, already_existed

def _save_theory_supabase(client, theory, source_filename, study_id, year,
                          authors, stage2=None):
    """
    Save full theory JSON to theory_extractions. Duplicate-safe.
    If stage2 is provided, cosine_matrix and pair_data are cached inside
    the JSON so reports can be regenerated without re-calling OpenAI.
    """
    try:
        existing = (client.table("theory_extractions")
                    .select("id").eq("file", source_filename).execute())
        if existing.data:
            return
        meta = theory.get("study_metadata", {})
        theory_clean = {k: v for k, v in theory.items()
                        if k not in ("_validation",)}
        if stage2 is not None:
            theory_clean["_stage2_cache"] = {
                "constructs":    stage2.get("constructs", []),
                "cosine_matrix": stage2.get("cosine_matrix", []),
                "pair_data":     stage2.get("pair_data", []),
            }
        client.table("theory_extractions").insert({
            "file":        source_filename,
            "study_id":    study_id,
            "year":        str(year),
            "authors":     authors,
            "title":       meta.get("title", ""),
            "theory_json": theory_clean
        }).execute()
    except Exception:
        pass


def _search_theory_supabase(query):
    """
    Search theory_extractions by author or year substring.
    Returns list of dicts with file, study_id, year, authors, title.
    """
    client = _get_supabase()
    if not client:
        return []
    try:
        q = str(query).strip()
        # Search authors and title with ilike
        results = (client.table("theory_extractions")
                   .select("file, study_id, year, authors, title")
                   .or_(f"authors.ilike.%{q}%,title.ilike.%{q}%,year.eq.{q}")
                   .order("year", desc=True)
                   .limit(20)
                   .execute())
        return results.data or []
    except Exception:
        return []


def _fetch_theory_supabase(source_file):
    """Fetch the full theory JSON for a given source_file."""
    client = _get_supabase()
    if not client:
        return None
    try:
        result = (client.table("theory_extractions")
                  .select("theory_json")
                  .eq("file", source_file)
                  .single()
                  .execute())
        return result.data.get("theory_json") if result.data else None
    except Exception:
        return None


def _corpus_counts_supabase(client):
    """Return (n_papers, n_pairs) from Supabase."""
    try:
        papers = client.table("paper_summary").select("id", count="exact").execute()
        pairs  = client.table("pooled_pairs").select("id", count="exact").execute()
        return papers.count or 0, pairs.count or 0
    except Exception:
        return 0, 0

# ── Local CSV save ────────────────────────────────────────────────────────────

def _save_local(theory, stage2, source_filename, study_id, year, authors):
    """Append to local CSV files. Returns (pairs_added, already_existed)."""
    pooled = _load_or_create(POOLED_DB_PATH, POOLED_COLS)
    already_existed = study_id in pooled["study_id"].values if len(pooled) else False

    pairs_added = 0
    if not already_existed:
        pair_rows = _build_pair_rows(theory, stage2, source_filename, study_id, year)
        if pair_rows:
            pooled = pd.concat([pooled, pd.DataFrame(pair_rows)], ignore_index=True)
            pooled.to_csv(POOLED_DB_PATH, index=False)
            pairs_added = len(pair_rows)

        summary = _load_or_create(SUMMARY_PATH, SUMMARY_COLS)
        if source_filename not in (summary["file"].values if len(summary) else []):
            summary_row = _build_summary_row(theory, stage2, source_filename, year, authors)
            summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)
            summary.to_csv(SUMMARY_PATH, index=False)

    return pairs_added, already_existed

def _corpus_counts_local():
    """Return (n_papers, n_pairs) from local CSVs."""
    try:
        n_papers = len(pd.read_csv(SUMMARY_PATH)) if os.path.exists(SUMMARY_PATH) else 0
        n_pairs  = len(pd.read_csv(POOLED_DB_PATH)) if os.path.exists(POOLED_DB_PATH) else 0
        return n_papers, n_pairs
    except Exception:
        return 0, 0

# ── Public interface ──────────────────────────────────────────────────────────

def save_to_local_db(theory, stage2, source_filename):
    """
    Save results to Supabase (if available) or local CSV (fallback).
    Returns (pairs_added, already_existed, backend_label).
    """
    meta     = theory.get("study_metadata", {})
    authors  = meta.get("authors", "Unknown")
    year     = meta.get("year", "")
    study_id = f"{authors.split(',')[0].strip()} ({year})"

    client = _get_supabase()
    if client:
        pairs_added, already_existed = _save_supabase(
            client, theory, stage2, source_filename, study_id, year, authors)
        n_papers, n_pairs = _corpus_counts_supabase(client)
        return pairs_added, already_existed, "Supabase", n_papers, n_pairs
    else:
        pairs_added, already_existed = _save_local(
            theory, stage2, source_filename, study_id, year, authors)
        n_papers, n_pairs = _corpus_counts_local()
        return pairs_added, already_existed, "local CSV", n_papers, n_pairs


# ── Streamlit UI ──────────────────────────────────────────────────────────────

# ── Free-access window helpers ────────────────────────────────────────────────

from datetime import datetime, timezone

DAILY_LIMIT = 20   # max free analyses per UTC day

def _free_window_status():
    """
    Returns (is_open, time_remaining_str, until_dt_or_None).
    Reads FREE_ACCESS_UNTIL from st.secrets (ISO-8601 string, UTC assumed).
    Returns is_open=False immediately if the key is absent or expired.
    """
    raw = st.secrets.get("FREE_ACCESS_UNTIL", "") or os.environ.get("FREE_ACCESS_UNTIL", "")
    if not raw:
        return False, "", None
    try:
        until = datetime.fromisoformat(str(raw).strip())
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        now   = datetime.now(timezone.utc)
        if now >= until:
            return False, "", until
        delta = until - now
        hours, rem = divmod(int(delta.total_seconds()), 3600)
        mins        = rem // 60
        remaining   = f"{hours}h {mins}m" if hours else f"{mins}m"
        return True, remaining, until
    except Exception:
        return False, "", None


def _daily_usage():
    """
    Returns current UTC-day analysis count from Supabase usage_counter table.
    Returns (count, limit_reached).
    """
    client = _get_supabase()
    if not client:
        return 0, False
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = client.table("usage_counter").select("count").eq("date", today).execute()
        count = result.data[0]["count"] if result.data else 0
        return count, count >= DAILY_LIMIT
    except Exception:
        return 0, False


def _increment_daily_usage():
    """Upsert today's counter row in Supabase usage_counter."""
    client = _get_supabase()
    if not client:
        return
    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        result = client.table("usage_counter").select("count").eq("date", today).execute()
        if result.data:
            new_count = result.data[0]["count"] + 1
            client.table("usage_counter").update({"count": new_count}).eq("date", today).execute()
        else:
            client.table("usage_counter").insert({"date": today, "count": 1}).execute()
    except Exception:
        pass   # never block an analysis over a counter failure


def _get_host_keys():
    """Return (ant_key, oai_key) from Streamlit secrets, or (None, None)."""
    ant = st.secrets.get("ANTHROPIC_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    oai = st.secrets.get("OPENAI_API_KEY", "")    or os.environ.get("OPENAI_API_KEY", "")
    return (ant or None, oai or None)


def _fmt_author(authors_str, year):
    """Format author string as 'Familyname et al. (year)' for hover labels."""
    if not authors_str or str(authors_str).strip() in ("", "nan"):
        return f"Unknown ({year})"
    s = str(authors_str).strip()
    if ";" in s:
        first_author = s.split(";")[0].strip()
        n_authors    = len(s.split(";"))
        family = first_author.split(",")[0].strip() if "," in first_author                  else first_author.split()[-1].rstrip(".,;")
    else:
        parts        = s.split(",")
        first_author = parts[0].strip()
        n_authors    = len(parts)
        tokens       = first_author.split()
        family       = tokens[-1].rstrip(".,;") if tokens else first_author
    suffix = " et al." if n_authors > 1 else ""
    return f"{family}{suffix} ({year})"


def sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Jan Smedslund\nSemantic Detector")
        st.caption("*In memory of Jan Smedslund (1929–2026)*")
        st.divider()

        st.subheader("Navigate")
        page = st.radio(
            "Page", ["Analyse a Paper", "Corpus Dashboard", "How to Read Results"],
            label_visibility="collapsed"
        )
        st.divider()

        # ── Check free-access window ──────────────────────────────────────
        is_open, remaining, until_dt = _free_window_status()
        host_ant, host_oai = _get_host_keys()
        free_active = is_open and bool(host_ant) and bool(host_oai)

        if free_active:
            daily_count, limit_reached = _daily_usage()
            if limit_reached:
                st.error(
                    f"🛑 **Daily limit reached** ({DAILY_LIMIT} free analyses today).  \n"
                    "Please enter your own API keys below, or try again tomorrow."
                )
                free_active = False
            else:
                st.success(
                    f"🔓 **Open access active** — {remaining} remaining  \n"
                    f"No API keys needed. Free analyses today: "
                    f"{daily_count} / {DAILY_LIMIT}."
                )
        elif until_dt and not is_open:
            st.info("🔒 The open-access window has closed. Please enter your own API keys.")

        # ── API key inputs (shown when free access is not active) ─────────
        if not free_active:
            st.subheader("API Keys")
            ant_key = st.text_input(
                "Anthropic API Key", type="password",
                help="Used for theory extraction via Claude claude-opus-4-5"
            )
            oai_key = st.text_input(
                "OpenAI API Key", type="password",
                help="Used for construct embeddings (text-embedding-3-large)"
            )
            st.info(
                "🔒 **Your API keys are private.**  \n"
                "They are used only to process your paper in this session "
                "and are never stored, logged, or transmitted anywhere "
                "other than directly to Anthropic and OpenAI."
            )
        else:
            # Use host keys silently
            ant_key = host_ant
            oai_key = host_oai
            st.caption("🔑 API keys provided by the host for this session.")

        st.divider()
        with st.expander("About this tool"):
            st.markdown(
                "Jan Smedslund argued that most empirical findings in psychology "
                "are **pseudo-empirical** — predictable a priori from the "
                "conceptual language researchers use to define their constructs. "
                "If construct A's definition semantically overlaps with construct "
                "B's, their empirical correlation is linguistically obligated "
                "before any data are collected.\n\n"
                "This tool operationalises that argument: it extracts construct "
                "definitions from a paper's theoretical sections, embeds them "
                "with OpenAI's large embedding model, and tests whether the "
                "resulting cosine similarities predict the reported effect sizes.\n\n"
                "**Three criteria are applied:**\n"
                "- **Pooled Spearman ρ** — cosine vs signed β (within-study)\n"
                "- **A>B concordance** — higher cosine → larger |effect|?\n"
                "- **A>B>C gradient** — mediator sits semantically between "
                "predictor and outcome?\n\n"
                "*In memory of Jan Smedslund (1929–2026)*"
            )
        st.caption("Developed by Ketil Arnulf · BI Norwegian Business School")
    return page, ant_key, oai_key


# ── Guide page ────────────────────────────────────────────────────────────────

def show_guide():
    st.title("How to Read Your Results")
    st.markdown(
        "This page explains what the Jan Smedslund Semantic Detector measures, "
        "what each number means, and how to interpret the verdict for your paper."
    )
    st.divider()

    st.header("The core argument")
    st.markdown("""
Jan Smedslund (1929–2026) argued throughout his career that most empirical findings
in psychology are **pseudo-empirical** — not genuine discoveries about the world,
but consequences of the language researchers use to define their constructs.

If the theoretical definition of construct A overlaps semantically with the definition
of construct B, then a positive correlation between them follows from language alone,
before any data are collected. Smedslund called this *semantic predetermination*: the
finding is built into the conceptual framework, not extracted from nature.

This tool tests that argument automatically. It reads the theoretical sections of your
paper, extracts how each construct is defined in language, measures the semantic
overlap between definitions using AI embeddings, and then asks: **do those overlaps
predict the effect sizes you actually found?**

If they do, your findings may have been predictable a priori. If they do not, your
data are doing something the theory did not already guarantee — which is precisely
what an empirical finding is supposed to be.
""")
    st.divider()

    st.header("Is my paper eligible?")
    st.markdown("""
The tool applies an automatic eligibility check before any analysis runs.
A paper must satisfy **all** of the following to proceed:

| Requirement | Details |
|---|---|
| **Empirical quantitative study** | Must collect and analyse primary data. Review articles, conceptual papers, and theoretical frameworks are excluded. |
| **Directional hypotheses** | At least 3 explicit hypothesised relationships between distinct constructs (e.g. "H1: A is positively related to B"). |
| **Reported effect sizes** | Must report standardised coefficients, correlations, path coefficients, or similar quantitative effect estimates. Significance-only results are not sufficient. |
| **Not a meta-analysis** | Meta-analytic effect sizes aggregate across prior studies and are not fresh observations against a single theoretical argument. |
| **Not a pure validation study** | Scale development papers without structural hypotheses between distinct constructs are excluded. A validation paper that also tests between-construct hypotheses may be eligible. |

**If your paper is rejected**, the tool will tell you the specific reason. This does not
cost API credits for the embedding step — the eligibility check runs first and stops
early if the paper does not qualify.
""")
    st.divider()

    st.header("The three criteria")

    with st.expander("📊  Criterion 1 — Pooled Spearman ρ  (the primary statistical test)", expanded=True):
        st.markdown("""
**What it measures:** The Spearman rank correlation between the cosine similarity
of construct definition pairs and the effect sizes reported for those same pairs.

**How to read it:**
- A **positive ρ** means that pairs with more semantically overlapping definitions
  also tend to show larger empirical effects — consistent with predetermination.
- A **negative or near-zero ρ** means the data are ordering effects differently
  from what the language would predict — a sign of genuine empirical content.
- The **signed ρ** uses the original sign of β, testing whether directionality is
  encoded in language. The **unsigned ρ** uses |β|, testing whether magnitude is.
  The signed ρ is expected to exceed the unsigned ρ — direction is more deeply
  embedded in conceptual language than precise magnitude.

**Individual-study caution:** With 4–15 pairs per study, within-study ρ values are
indicative only. The pooled corpus Spearman — combining all studies — is the
statistically meaningful test.

**Corpus benchmark:** ρ = 0.259 (p = 5.1 × 10⁻¹⁴) across 103 studies, 819 pairs.
""")

    with st.expander("📊  Criterion 2 — A>B Concordance  (the universal within-study test)"):
        st.markdown("""
**What it measures:** For every pair of hypothesised construct pairs (i, j) in your
study: is the pair with the higher cosine similarity also the pair with the larger
absolute effect size?

- **50%** = chance. Semantic order and effect order are unrelated.
- **Above 50%** = semantically closer pairs more often show larger effects.
- **Below 50%** = the data order effects opposite to language proximity.

**Interpretation bands:**
- ≥ 70% — strong semantic structure
- 55–69% — moderate semantic structure
- < 55% — weak or potentially empirically independent

**Corpus benchmark:** 60.8% across 84 papers (5,151 pair comparisons).
""")

    with st.expander("📊  Criterion 3 — A>B>C Mediation Gradient  (papers with mediation only)"):
        st.markdown("""
**What it measures:** For each mediation chain A → B → C, the test checks whether
the mediator B sits semantically *between* A and C in embedding space:

- cos(A, B) > cos(A, C) — predictor is closer to mediator than to outcome
- cos(B, C) > cos(A, C) — mediator is closer to outcome than predictor is

**PASS** — the mediation is a semantic gradient already present in how the constructs
are defined. The mediation argument mirrors a conceptual hierarchy in language.

**FAIL** — the mediator introduces genuine conceptual distance. A failing chain
suggests the mediation may have real empirical content beyond language alone.

**Corpus benchmark:** 54.1% of chains pass across 37 papers (222 chains).
""")

    st.divider()
    st.header("The verdict")
    st.markdown("""
Each criterion above its threshold scores one signal. The three thresholds are:
A>B concordance ≥ 60%, A>B>C pass rate ≥ 55%, within-study signed ρ > 0.
""")

    with st.expander("🔴  Semantically Structured  (2–3 signals)"):
        st.markdown("""
The findings in this paper were largely predictable from the conceptual language
of the construct definitions. The constructs are linguistically related, and the
effect sizes add limited empirical information beyond what language already implied.

This is the most common verdict in the corpus. It is not a personal failing — it
reflects how psychological constructs are typically defined. The interesting
follow-up question is *how much* of the signal is linguistic versus causal.
""")

    with st.expander("🟡  Partially Structured  (1 signal)"):
        st.markdown("""
Some semantic structure is present but the data are also doing something the
language did not guarantee. The paper contains a mix: some relationships that
follow from construct definitions, and potentially some genuine empirical signal.

These papers reward closer inspection of individual construct pairs: which pairs
drive the concordance signal, and which resist semantic prediction?
""")

    with st.expander("🟢  Empirically Independent  (0 signals)"):
        st.markdown("""
The effect sizes are not well predicted by semantic proximity. The data order
relationships differently from what construct definitions would imply — the
pattern most consistent with genuine empirical discovery.

A green verdict does not automatically mean the study is excellent. It could
reflect constructs defined in genuinely independent language (a good sign),
measurement idiosyncrasies, or a very small number of construct pairs (limited
power). The heatmap, cosine range, and pair count are important context.
""")

    with st.expander("🟣  Semantic Inflation  (0–1 signals + elevated cosine and effects)"):
        st.markdown("""
This is a special case requiring careful interpretation. The paper appears
empirically independent on the ordinal tests, but its mean cosine similarity and
mean effect size are both elevated relative to the corpus.

**The mechanism.** When all construct pairs are *uniformly* semantically similar,
the ordinal ranking test has no variance to detect. If every pair has cosine around
0.62–0.65, there is no meaningful ordering to follow — even though the effects
themselves may be large precisely *because* all constructs are semantically
predetermined. The absence of ordinal signal is a measurement failure caused by
restriction of range, not evidence of empirical independence.

**How it is detected.** Three converging signals:
- Mean cosine across all construct pairs ≥ 0.50 (uniformly high overlap)
- Mean absolute effect size ≥ 0.30 (elevated effects consistent with predetermination)
- Fewer than 2 ordinal signals from standard criteria

**Corpus-level detection.** As the corpus grows, a supplementary analysis fits a
regression of mean effects on mean cosine across all papers. Papers with large
*positive* residuals from this trend — showing more effect than expected even
given their cosine level — are the strongest inflation candidates. This approach
is self-calibrating: as new papers accumulate, the reference distribution becomes
more precise and borderline cases become easier to classify.

**What it means for your paper.** Your constructs may be so conceptually
intertwined that the findings were highly predictable from their definitions alone,
even though the ordinal test could not detect this directly. This does not mean
your data are wrong, but it suggests that the relationships may be substantially
driven by shared conceptual language rather than independent causal processes.

**The heatmap signature.** Inflation papers typically show a heatmap that is
uniformly mid-to-high blue with little variation between cells. If you see this
pattern alongside a semantic inflation verdict, it confirms the diagnosis.
""")

    st.markdown("""
**Important framing.** None of these verdicts is a personal criticism of the
researchers. Semantic predetermination is a structural feature of how psychological
constructs are defined in theoretical language. Most quantitative psychology falls
somewhere on the structured-to-inflation spectrum. Papers that are genuinely
empirically independent — truly green — contain the findings with the strongest
claim to be discoveries about the world rather than reflections of our vocabulary.
""")

    st.divider()
    st.header("The cosine similarity heatmap")
    st.markdown("""
The heatmap shows pairwise cosine similarity between every pair of construct
definitions. Values range from 0 (unrelated) to 1 (identical). The diagonal is
always 1.0.

- **High values (> 0.80):** Very close in meaning. Empirical correlation likely obligated.
- **Moderate values (0.35–0.65):** The arguable band — related enough to hypothesise,
  distinct enough to appear non-trivial. Most published research lives here, and most
  semantic predetermination occurs here.
- **Low values (< 0.30):** Conceptually distinct. Less likely to be predetermined.
- **Compressed range (all values similar):** The signature of potential semantic inflation.
  The heatmap looks uniformly coloured with little variation, meaning the ordinal test
  has almost no cosine variance to work with.

Definitions are extracted from **theoretical sections only** — introduction, theory,
hypothesis development — never from the methods/measures section. Smedslund\'s
argument is about conceptual language, not psychometric instruments.
""")

    st.divider()
    st.header("The empirical R\u00b2 benchmark")
    st.markdown("""
Where your paper reports explained variance (R\u00b2), the tool compares it to
Smedslund\'s theoretically derived benchmark of **0.428**.

This value was derived analytically: if within-factor item loadings are \u2265 0.70
and cross-loadings are \u2264 0.30 (standard psychometric quality), the expected
correlation between factor scores is approximately 0.65, yielding R\u00b2 \u2248 0.428.
Studies using reliable self-report measures of semantically related constructs
should explain around 43% of variance \u2014 before any data collection, purely
from the measurement structure.

**Corpus mean:** approximately 0.35 across papers with extractable R\u00b2. Papers
**exceeding** the benchmark are candidates for particularly strong semantic
predetermination. Papers **well below** it may measure more genuinely distinct
constructs, or involve non-survey measures less susceptible to predetermination.

**R\u00b2 and inflation:** Semantic inflation papers often show R\u00b2 well above the
benchmark, because shared semantic content inflates explained variance throughout
the model. An R\u00b2 above 0.55 combined with a structured or inflation verdict is
strong evidence that definitional overlap is the primary driver of the findings.
""")

    st.divider()
    st.header("About and contact")
    st.markdown("""
This tool was developed by **Ketil Arnulf** (BI Norwegian Business School) in
collaboration with Claude (Anthropic) as a memorial to Jan Smedslund (1929–2026).

For questions about the tool, access to the dataset, or collaboration,
contact: **ketil.arnulf@bi.no**

*Every paper analysed through this tool (with the user's consent) contributes
to the growing corpus. The benchmark statistics on the Dashboard update in real
time as new analyses are added.*
""")


# ── Corpus Dashboard ──────────────────────────────────────────────────────────

# ── Dashboard verdict helper ──────────────────────────────────────────────────

def _dashboard_verdict(row):
    """Apply 4-category verdict to a summary row (uses stored mean_cosine/mean_abs_beta)."""
    signals = 0
    if pd.notna(row.get("ab_rate")) and row["ab_rate"] >= 0.60:        signals += 1
    if (pd.notna(row.get("abc_rate")) and pd.notna(row.get("abc_total"))
            and row["abc_total"] > 0 and row["abc_rate"] >= 0.55):     signals += 1
    if (pd.notna(row.get("signed_rho")) and pd.notna(row.get("n_pairs"))
            and row["n_pairs"] >= 5 and row["signed_rho"] > 0):        signals += 1

    mc   = row.get("mean_cosine")
    mb   = row.get("mean_abs_beta")
    inflation = (
        signals < 2
        and pd.notna(mc) and pd.notna(mb)
        and mc >= INFLATION_COSINE_THRESHOLD
        and mb >= INFLATION_BETA_THRESHOLD
    )
    if signals >= 2:   return "Semantically Structured"
    if inflation:      return "Semantic Inflation"
    if signals == 1:   return "Partially Structured"
    return "Empirically Independent"


@st.cache_data(ttl=120)   # refresh every 2 minutes
def _load_corpus():
    """Load all pairs and summaries from Supabase (or local CSV fallback)."""
    client = _get_supabase()
    if client:
        try:
            # Pairs — may exceed 1000 rows; page through
            all_pairs = []
            offset = 0
            while True:
                batch = (client.table("pooled_pairs")
                         .select("*")
                         .range(offset, offset + 999)
                         .execute())
                all_pairs.extend(batch.data)
                if len(batch.data) < 1000:
                    break
                offset += 1000
            pairs_df = pd.DataFrame(all_pairs) if all_pairs else pd.DataFrame()

            # Summaries
            all_summary = []
            offset = 0
            while True:
                batch = (client.table("paper_summary")
                         .select("*")
                         .range(offset, offset + 999)
                         .execute())
                all_summary.extend(batch.data)
                if len(batch.data) < 1000:
                    break
                offset += 1000
            summary_df = pd.DataFrame(all_summary) if all_summary else pd.DataFrame()
            return pairs_df, summary_df, "Supabase"
        except Exception as e:
            st.warning(f"Supabase load failed ({e}) — trying local CSV.")

    # Local fallback
    pairs_df   = pd.read_csv(POOLED_DB_PATH)   if os.path.exists(POOLED_DB_PATH) else pd.DataFrame()
    summary_df = pd.read_csv(SUMMARY_PATH)      if os.path.exists(SUMMARY_PATH)  else pd.DataFrame()
    return pairs_df, summary_df, "local CSV"


def _pooled_spearman(pairs_df):
    """Compute pooled signed and unsigned Spearman across all pairs.
    Pairs with |effect| > BETA_CEILING are excluded as likely non-standardised."""
    df = pairs_df.dropna(subset=["cosine", "signed_effect", "unsigned_effect"])
    n_raw    = len(df)
    df       = df[df["unsigned_effect"] <= BETA_CEILING]
    n_excluded = n_raw - len(df)
    if len(df) < 10:
        return None, None, None, None, len(df), n_excluded
    sr = stats.spearmanr(df["cosine"], df["signed_effect"])
    ur = stats.spearmanr(df["cosine"], df["unsigned_effect"])
    return float(sr.statistic), float(sr.pvalue), float(ur.statistic), float(ur.pvalue), len(df), n_excluded
    sr = stats.spearmanr(df["cosine"], df["signed_effect"])
    ur = stats.spearmanr(df["cosine"], df["unsigned_effect"])
    return float(sr.statistic), float(sr.pvalue), float(ur.statistic), float(ur.pvalue), len(df)


def show_dashboard():
    st.title("Corpus Dashboard")
    st.markdown(
        "Live view of all papers analysed through this tool, "
        "updating automatically as researchers contribute new analyses."
    )

    with st.spinner("Loading corpus from database…"):
        pairs_df, summary_df, backend = _load_corpus()

    if pairs_df.empty or summary_df.empty:
        st.warning("No data in the corpus yet. Analyse some papers first.")
        return

    n_papers = len(summary_df)
    n_pairs  = len(pairs_df)

    # ── Apply 4-category verdicts ─────────────────────────────────────────
    summary_df["verdict"] = summary_df.apply(_dashboard_verdict, axis=1)

    # ── Top metrics ───────────────────────────────────────────────────────
    signed_rho, signed_p, unsigned_rho, unsigned_p, n_valid, n_excl = _pooled_spearman(pairs_df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Papers in corpus", n_papers)
    m2.metric("Construct pairs",  n_pairs)
    m3.metric(
        "Pooled signed ρ",
        f"{signed_rho:.3f}" if signed_rho is not None else "—",
        help="Spearman between cosine similarity and signed β across all pairs. "
             "Corpus baseline (Arnulf 2026): ρ = 0.259, p = 5.1×10⁻¹⁴"
    )
    m4.metric(
        "p-value",
        f"{signed_p:.2e}" if signed_p is not None else "—"
    )

    # A>B corpus-wide rate
    ab_vals = summary_df["ab_rate"].dropna()
    m5.metric(
        "Mean A>B concordance",
        f"{ab_vals.mean()*100:.1f}%" if len(ab_vals) else "—",
        help="Average within-study A>B concordance rate across all papers. "
             "Corpus baseline: 60.8%"
    )

    excl_note = (f" · {n_excl} pairs excluded (|β| > {BETA_CEILING}, likely non-standardised)"
                 if n_excl and n_excl > 0 else "")
    st.caption(f"Data source: {backend} · Refreshes every 2 minutes{excl_note}")
    st.divider()

    # ── Pooled scatter ────────────────────────────────────────────────────
    st.subheader("Pooled Cosine vs Effect Size")
    plot_df = pairs_df.dropna(subset=["cosine", "signed_effect"])
    plot_df = plot_df[plot_df["unsigned_effect"] <= BETA_CEILING]
    if len(plot_df) > 0:
        # Sample for performance if very large
        sample = plot_df.sample(min(len(plot_df), 1500), random_state=42)

        # Build hover label: "A → B  |  Familyname et al. (year)"
        # Build lookup: study_id key → formatted author label
        if "study_id" in sample.columns and "authors" in summary_df.columns:
            sid_to_fmt = {}
            for _, row in summary_df.iterrows():
                sid = str(row.get("authors","")).split(",")[0].strip().split()[-1].rstrip(".,;")
                yr  = str(row.get("year",""))
                key = f"{sid} ({yr})"
                sid_to_fmt[key] = _fmt_author(row.get("authors",""), yr)

            def pair_hover(row):
                a = str(row.get("construct_a", "A"))
                b = str(row.get("construct_b", "B"))
                sid = str(row.get("study_id", ""))
                author_label = sid_to_fmt.get(sid, sid)
                return f"{a} → {b}  |  {author_label}"

            hover_labels = sample.apply(pair_hover, axis=1).values
        else:
            hover_labels = sample.get("study_id", sample.index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample["cosine"], y=sample["signed_effect"],
            mode="markers",
            name="Construct pair",
            marker=dict(size=5, color="#60a5fa", opacity=0.5),
            hovertemplate="<b>%{customdata}</b><br>cosine=%{x:.3f} β=%{y:.3f}<extra></extra>",
            customdata=hover_labels,
            showlegend=False
        ))
        if len(plot_df) >= 10:
            m, b = np.polyfit(plot_df["cosine"], plot_df["signed_effect"], 1)
            xs = np.linspace(plot_df["cosine"].min(), plot_df["cosine"].max(), 50)
            fig.add_trace(go.Scatter(
                x=xs, y=m*xs+b, mode="lines",
                line=dict(color="#f59e0b", dash="dash", width=2),
                name=f"OLS trend",
                showlegend=False
            ))
            # Show rho as a clean annotation in the top-left corner
            rho_str = f"Pooled signed ρ = {signed_rho:.3f}   p = {signed_p:.2e}   n = {n_valid:,} pairs"
            fig.add_annotation(
                x=0.01, y=0.97, xref="paper", yref="paper",
                text=rho_str, showarrow=False,
                font=dict(color="#f59e0b", size=13),
                align="left", bgcolor="rgba(0,0,0,0.3)",
                borderpad=4
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#475569")
        fig.update_layout(
            xaxis_title="Cosine similarity (definition level)",
            yaxis_title="Signed effect size (β)",
            height=400,
            showlegend=False,
            **_BASE_LAYOUT
        )
        st.plotly_chart(fig, use_container_width=True)
        if len(plot_df) > 1500:
            st.caption(f"Showing a random sample of 1,500 of {len(plot_df):,} pairs for display speed.")

    # ── Paper-level: mean cosine vs mean |β| ────────────────────────────
    st.subheader("Paper-Level Cosine vs Effect Size")
    st.caption(
        "Each dot is one paper. Confirms the corpus-level signal at the paper level "
        "and identifies semantic inflation cases (purple) where uniform predetermination "
        "defeats the ordinal within-paper test."
    )

    VERDICT_COLORS = {
        "Semantically Structured":  "#ef4444",
        "Partially Structured":     "#f59e0b",
        "Empirically Independent":  "#22c55e",
        "Semantic Inflation":       "#a855f7"
    }

    if "mean_cosine" in summary_df.columns and "mean_abs_beta" in summary_df.columns:
        pl_df = summary_df.dropna(subset=["mean_cosine", "mean_abs_beta"])
        if len(pl_df) >= 5:
            fig_pl = go.Figure()
            for v, col in VERDICT_COLORS.items():
                sub = pl_df[pl_df["verdict"] == v]
                if len(sub) == 0:
                    continue
                year_col  = sub.get("year", pd.Series([""] * len(sub))).astype(str)
                hover = sub.apply(
                    lambda r: _fmt_author(r.get("authors",""), r.get("year","")), axis=1
                )
                marker_opts = dict(size=8, color=col, opacity=0.75,
                                   line=dict(width=1, color=col))
                if v == "Semantic Inflation":
                    marker_opts["symbol"] = "diamond"
                    marker_opts["size"]   = 10
                fig_pl.add_trace(go.Scatter(
                    x=sub["mean_cosine"], y=sub["mean_abs_beta"],
                    mode="markers", name=v,
                    marker=marker_opts,
                    customdata=hover,
                    hovertemplate="<b>%{customdata}</b><br>mean cosine %{x:.3f}<br>mean |β| %{y:.3f}<extra></extra>"
                ))
            # OLS trend line across all papers
            if len(pl_df) >= 5:
                mc_all = pl_df["mean_cosine"].values
                mb_all = pl_df["mean_abs_beta"].values
                m_pl, b_pl = np.polyfit(mc_all, mb_all, 1)
                xs_pl = np.linspace(mc_all.min(), mc_all.max(), 50)
                r_pl, p_pl = stats.spearmanr(mc_all, mb_all)
                fig_pl.add_trace(go.Scatter(
                    x=xs_pl, y=m_pl*xs_pl + b_pl, mode="lines",
                    name="OLS trend", showlegend=False,
                    line=dict(color="#94a3b8", dash="dash", width=1.5)
                ))
                fig_pl.add_annotation(
                    x=0.01, y=0.97, xref="paper", yref="paper",
                    text=f"Spearman ρ = {r_pl:.3f}   p = {p_pl:.3e}   n = {len(pl_df)} papers",
                    showarrow=False,
                    font=dict(color="#94a3b8", size=12),
                    align="left", bgcolor="rgba(0,0,0,0.3)", borderpad=4
                )
            # Inflation zone shading
            fig_pl.add_vrect(
                x0=INFLATION_COSINE_THRESHOLD, x1=1.0,
                fillcolor="rgba(168,85,247,0.07)", line_width=0
            )
            fig_pl.add_hrect(
                y0=INFLATION_BETA_THRESHOLD, y1=2.0,
                fillcolor="rgba(168,85,247,0.07)", line_width=0
            )
            fig_pl.add_annotation(
                x=INFLATION_COSINE_THRESHOLD + 0.01,
                y=INFLATION_BETA_THRESHOLD + 0.02,
                xref="x", yref="y",
                text="⚠ inflation zone", showarrow=False,
                font=dict(color="#a855f7", size=11)
            )
            fig_pl.update_layout(
                xaxis_title="Mean cosine similarity (paper level)",
                yaxis_title="Mean |β| (paper level)",
                height=420,
                legend=dict(font=dict(color=_FONT_COL, size=11),
                            bgcolor="rgba(0,0,0,0.3)",
                            bordercolor="#475569", borderwidth=1),
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig_pl, use_container_width=True)

            # Inflation summary
            n_inflation = (summary_df["verdict"] == "Semantic Inflation").sum()
            if n_inflation > 0:
                st.warning(
                    f"⚠ **{n_inflation} paper{'s' if n_inflation > 1 else ''} flagged as Semantic Inflation** "
                    f"(mean cosine ≥ {INFLATION_COSINE_THRESHOLD}, mean |β| ≥ {INFLATION_BETA_THRESHOLD}). "
                    "These papers appear empirically independent on ordinal tests but show elevated effects "
                    "consistent with uniform predetermination. They should not be interpreted as genuine "
                    "empirical discoveries."
                )
        else:
            st.info("Paper-level scatter requires mean_cosine data. "
                    "Run the Supabase backfill notebook cell to populate existing records.")
    else:
        st.info("mean_cosine column not yet in database. "
                "Run the SQL and backfill steps to enable this plot.")

    # ── Corpus-level inflation analysis ──────────────────────────────────
    if "mean_cosine" in summary_df.columns and "mean_abs_beta" in summary_df.columns:
        pl_full = summary_df.dropna(subset=["mean_cosine", "mean_abs_beta"])
        if len(pl_full) >= 10:
            st.subheader("Corpus-Level Inflation Detection")
            st.caption(
                "Three complementary methods identify probable inflation cases beyond "
                "the fixed-threshold primary criterion. Each uses the full corpus as reference."
            )

            mc = pl_full["mean_cosine"].values
            mb = pl_full["mean_abs_beta"].values

            # Method 1: Regression residuals
            m_reg, b_reg = np.polyfit(mc, mb, 1)
            residuals = mb - (m_reg * mc + b_reg)
            pl_full = pl_full.copy()
            pl_full["residual"] = residuals

            # Method 2: Percentile-based (75th pct both dimensions)
            pct75_cosine = float(np.percentile(mc, 75))
            pct75_beta   = float(np.percentile(mb, 75))
            pl_full["pct_flag"] = (
                (pl_full["mean_cosine"] >= pct75_cosine) &
                (pl_full["mean_abs_beta"] >= pct75_beta)
            )

            # Method 3: Cosine range restriction
            if "cosine_range" in pl_full.columns:
                cr = pl_full["cosine_range"].fillna(1.0)
                pl_full["range_flag"] = (cr < 0.08) & (pl_full["mean_cosine"] > 0.42)
            else:
                pl_full["range_flag"] = False

            # Composite: how many methods flag each paper
            pl_full["inflation_signals"] = (
                (pl_full["verdict"] == "Semantic Inflation").astype(int) +
                pl_full["pct_flag"].astype(int) +
                pl_full["range_flag"].astype(int) +
                (residuals > np.percentile(residuals, 80)).astype(int)
            )

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                n_pct = pl_full["pct_flag"].sum()
                st.metric("Percentile flag (top 25% both)",
                          n_pct,
                          help=f"Mean cosine ≥ {pct75_cosine:.3f} AND mean |β| ≥ {pct75_beta:.3f} (75th percentile thresholds, self-calibrating)")
            with col_m2:
                n_range = pl_full["range_flag"].sum()
                st.metric("Range restriction flag",
                          n_range,
                          help="Cosine range < 0.08 AND mean cosine > 0.42 — ordinal test has near-zero variance")
            with col_m3:
                n_multi = (pl_full["inflation_signals"] >= 3).sum()
                st.metric("Flagged by ≥ 3 methods",
                          n_multi,
                          help="Papers flagged by the primary criterion + at least 2 of the 3 corpus-level methods")

            # Residual plot
            fig_res = go.Figure()
            for v, col in VERDICT_COLORS.items():
                sub = pl_full[pl_full["verdict"] == v]
                if len(sub) == 0:
                    continue
                year_col   = sub.get("year", pd.Series([""] * len(sub))).astype(str)
                hover = sub.apply(
                    lambda r: _fmt_author(r.get("authors",""), r.get("year","")), axis=1
                )
                fig_res.add_trace(go.Scatter(
                    x=sub["mean_cosine"],
                    y=sub["residual"],
                    mode="markers",
                    name=v,
                    marker=dict(size=7, color=col, opacity=0.75),
                    customdata=hover,
                    hovertemplate="<b>%{customdata}</b><br>mean cosine %{x:.3f}<br>residual %{y:.3f}<extra></extra>"
                ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#475569",
                              annotation_text="corpus trend", annotation_font_color="#94a3b8")
            fig_res.add_hline(
                y=float(np.percentile(residuals, 80)),
                line_dash="dot", line_color="#a855f7",
                annotation_text="80th pct residual", annotation_font_color="#a855f7"
            )
            fig_res.update_layout(
                title="Residuals from corpus regression (mean cosine → mean |β|)",
                xaxis_title="Mean cosine", yaxis_title="Residual (observed − predicted mean |β|)",
                height=340,
                legend=dict(font=dict(color=_FONT_COL, size=11),
                            bgcolor="rgba(0,0,0,0.3)", bordercolor="#475569", borderwidth=1),
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig_res, use_container_width=True)
            st.caption(
                "Papers above the purple dotted line show larger effects than the corpus "
                "regression predicts for their cosine level. These are the strongest inflation "
                "candidates. The threshold adapts automatically as the corpus grows."
            )

            # Multi-flag table
            multi_flagged = pl_full[pl_full["inflation_signals"] >= 3].sort_values(
                "inflation_signals", ascending=False
            )
            if len(multi_flagged) > 0:
                with st.expander(f"Papers flagged by ≥ 3 methods ({len(multi_flagged)})"):
                    display_cols = ["authors", "year", "mean_cosine", "mean_abs_beta",
                                    "cosine_range", "verdict", "inflation_signals"]
                    display_cols = [c for c in display_cols if c in multi_flagged.columns]
                    st.dataframe(
                        multi_flagged[display_cols].rename(columns={
                            "mean_cosine": "mean cosine",
                            "mean_abs_beta": "mean |β|",
                            "cosine_range": "cosine range",
                            "inflation_signals": "# flags"
                        }),
                        hide_index=True, use_container_width=True
                    )

    # ── Verdict distribution ──────────────────────────────────────────────
    st.subheader("Verdict Distribution")
    v_counts = summary_df["verdict"].value_counts()
    v_order  = ["Semantically Structured", "Partially Structured",
                 "Empirically Independent", "Semantic Inflation"]
    v_labels = [v for v in v_order if v in v_counts.index]
    v_vals   = [v_counts.get(v, 0) for v in v_labels]
    v_colors = [VERDICT_COLORS[v] for v in v_labels]
    fig_vc = go.Figure(go.Bar(
        x=v_labels, y=v_vals,
        marker_color=v_colors,
        text=v_vals, textposition="auto"
    ))
    fig_vc.update_layout(
        xaxis_title="Verdict", yaxis_title="Papers",
        height=280, showlegend=False, **_BASE_LAYOUT
    )
    st.plotly_chart(fig_vc, use_container_width=True)

    # ── A>B distribution ──────────────────────────────────────────────────
    st.subheader("A>B Concordance Distribution")
    col_l, col_r = st.columns(2)

    with col_l:
        ab_df = summary_df.dropna(subset=["ab_rate"])
        if len(ab_df):
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=ab_df["ab_rate"] * 100,
                nbinsx=20,
                marker_color="#60a5fa",
                name="Papers"
            ))
            fig2.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                           annotation_text="50% chance",
                           annotation_position="top left",
                           annotation_font=dict(color="#94a3b8", size=12))
            mean_pct = ab_vals.mean() * 100
            fig2.add_vline(x=mean_pct, line_dash="dot", line_color="#f59e0b",
                           annotation_text=f"mean {mean_pct:.1f}%",
                           annotation_position="top right",
                           annotation_font=dict(color="#f59e0b", size=12))
            fig2.update_layout(
                xaxis_title="A>B concordance (%)",
                yaxis_title="Number of papers",
                height=320,
                showlegend=False,
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Top and bottom 5 papers as a compact table
            label_col = "authors" if "authors" in ab_df.columns else ab_df.index.name
            ab_display = ab_df.copy()
            ab_display["Author (year)"] = ab_display.apply(
                lambda r: _fmt_author(r.get("authors",""), r.get("year","")), axis=1
            )
            ab_display["A>B %"] = (ab_display["ab_rate"] * 100).round(1)
            ab_sorted = ab_display.sort_values("ab_rate", ascending=False)
            top5 = ab_sorted.head(5)[["Author (year)", "A>B %"]]
            bot5 = ab_sorted.tail(5)[["Author (year)", "A>B %"]].iloc[::-1]
            t1, t2 = st.columns(2)
            with t1:
                st.caption("**Highest concordance**")
                st.dataframe(top5, hide_index=True, use_container_width=True)
            with t2:
                st.caption("**Lowest concordance**")
                st.dataframe(bot5, hide_index=True, use_container_width=True)

            # Full sortable table in expander
            with st.expander(f"All {len(ab_display)} papers — click to expand"):
                full = ab_sorted[["Author (year)", "A>B %"]].reset_index(drop=True)
                st.dataframe(full, hide_index=True, use_container_width=True)

    with col_r:
        # ABC pass rate distribution
        abc_df = summary_df.dropna(subset=["abc_rate"])
        abc_df = abc_df[abc_df["abc_total"] > 0]
        if len(abc_df):
            st.markdown("**A>B>C Mediation Gradient**")
            abc_pass_mean = abc_df["abc_rate"].mean()
            st.metric("Mean pass rate", f"{abc_pass_mean*100:.1f}%",
                      help="Corpus baseline: 54.1%")
            fig3 = go.Figure(go.Histogram(
                x=abc_df["abc_rate"] * 100,
                nbinsx=10,
                marker_color="#818cf8"
            ))
            fig3.add_vline(x=50, line_dash="dash", line_color="#94a3b8",
                           annotation_text="50%")
            fig3.update_layout(
                xaxis_title="A>B>C pass rate (%)",
                yaxis_title="Papers",
                height=280,
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No papers with mediation chains yet.")

    # ── Corpus composition ────────────────────────────────────────────────
    st.subheader("Corpus Composition")
    col_a, col_b = st.columns(2)

    with col_a:
        if "model_type" in summary_df.columns:
            # Clean up raw model_type strings for display
            label_map = {
                "mediation":            "Mediation",
                "direct_only":          "Direct only",
                "moderation":           "Moderation",
                "mediated_moderation":  "Mediated moderation",
                "other":                "Other"
            }
            mt = summary_df["model_type"].dropna().map(
                lambda x: label_map.get(str(x).strip(), str(x).replace("_", " ").title())
            )
            mc = mt.value_counts()
            fig4 = go.Figure(go.Pie(
                labels=mc.index, values=mc.values,
                hole=0.4,
                marker_colors=["#60a5fa","#818cf8","#4ade80","#f59e0b","#f87171"],
                textfont=dict(size=13, color="#e0e0e0"),
                insidetextfont=dict(size=13, color="#e0e0e0"),
                outsidetextfont=dict(size=13, color="#e0e0e0")
            ))
            fig4.update_layout(
                title=dict(text="Model type", font=dict(color="#e0e0e0", size=14)),
                height=300,
                legend=dict(
                    font=dict(color="#e0e0e0", size=12),
                    bgcolor="rgba(0,0,0,0.3)",
                    bordercolor="#475569",
                    borderwidth=1
                ),
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        if "year" in summary_df.columns:
            yc = summary_df["year"].dropna().astype(str).str[:4]
            yc = yc[yc.str.isnumeric()].astype(int).value_counts().sort_index()
            fig5 = go.Figure(go.Bar(
                x=yc.index, y=yc.values,
                marker_color="#60a5fa"
            ))
            fig5.update_layout(
                title="Papers by decade",
                xaxis_title="Year", yaxis_title="Papers",
                height=280,
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig5, use_container_width=True)

    # ── Signed vs unsigned gap ────────────────────────────────────────────
    if signed_rho is not None and unsigned_rho is not None:
        st.subheader("Signed vs Unsigned Spearman Gap")
        gap = signed_rho - unsigned_rho
        c1, c2, c3 = st.columns(3)
        c1.metric("Signed ρ",   f"{signed_rho:.3f}")
        c2.metric("Unsigned ρ", f"{unsigned_rho:.3f}")
        c3.metric("Gap (signed − unsigned)", f"{gap:+.3f}",
                  help="Positive gap means directionality is better encoded "
                       "in language than magnitude — as Smedslund\'s theory predicts.")
        st.caption(
            "Smedslund\'s framework predicts the signed correlation will exceed "
            "the unsigned: construct definitions encode the *direction* of "
            "relationships more reliably than their magnitude. "
            f"Current gap: {gap:+.3f} ({'✓ consistent with theory' if gap > 0 else '✗ inconsistent with theory'})."
        )

    # ── Empirical R² distribution ────────────────────────────────────────
    r2_data = summary_df["avg_empirical_r2"].dropna()
    if len(r2_data) >= 3:
        st.subheader("Empirical R² Distribution")
        col_g1, col_g2 = st.columns([1, 2])

        with col_g1:
            # Gauge showing corpus mean vs Smedslund benchmark
            corpus_mean_r2 = float(r2_data.mean())
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=corpus_mean_r2,
                number={"valueformat": ".3f", "font": {"color": _FONT_COL, "size": 40}},
                gauge={
                    "axis": {"range": [0, 1],
                             "tickcolor": _FONT_COL,
                             "tickfont": {"color": _FONT_COL}},
                    "bar": {"color": "#60a5fa"},
                    "bgcolor": "#334155",
                    "steps": [
                        {"range": [0, SMEDSLUND_R2_BENCH], "color": "#1e293b"},
                        {"range": [SMEDSLUND_R2_BENCH, 1], "color": "#164e63"}
                    ],
                    "threshold": {
                        "line": {"color": "#f59e0b", "width": 3},
                        "thickness": 0.85,
                        "value": SMEDSLUND_R2_BENCH
                    }
                },
                title={"text": f"Corpus mean R²<br><sub style='color:#94a3b8'>Smedslund benchmark = {SMEDSLUND_R2_BENCH} (amber line)</sub>",
                       "font": {"color": _FONT_COL, "size": 13}}
            ))
            fig_g.update_layout(
                height=260,
                paper_bgcolor=_DARK_BG,
                font=dict(color=_FONT_COL),
                margin=dict(l=20, r=20, t=60, b=10)
            )
            st.plotly_chart(fig_g, use_container_width=True)
            n_above = int((r2_data >= SMEDSLUND_R2_BENCH).sum())
            st.caption(
                f"{n_above} of {len(r2_data)} papers ({100*n_above/len(r2_data):.0f}%) "
                f"exceed the Smedslund benchmark of {SMEDSLUND_R2_BENCH}."
            )

        with col_g2:
            # Histogram of R² values across papers
            fig_r2h = go.Figure(go.Histogram(
                x=r2_data,
                nbinsx=20,
                marker_color="#60a5fa",
                name="Papers"
            ))
            fig_r2h.add_vline(
                x=SMEDSLUND_R2_BENCH, line_dash="dash", line_color="#f59e0b",
                annotation_text=f"Benchmark {SMEDSLUND_R2_BENCH}",
                annotation_position="top right",
                annotation_font=dict(color="#f59e0b", size=12)
            )
            fig_r2h.add_vline(
                x=r2_data.mean(), line_dash="dot", line_color="#60a5fa",
                annotation_text=f"Mean {r2_data.mean():.3f}",
                annotation_position="top left",
                annotation_font=dict(color="#60a5fa", size=12)
            )
            fig_r2h.update_layout(
                xaxis_title="Average empirical R²",
                yaxis_title="Number of papers",
                height=260,
                showlegend=False,
                **_BASE_LAYOUT
            )
            st.plotly_chart(fig_r2h, use_container_width=True)


    # ── Cosine → R² scatter and ratio panel ──────────────────────────────
    if "mean_cosine" in summary_df.columns:
        r2_cos_df = summary_df.dropna(subset=["avg_empirical_r2", "mean_cosine"])
        # Exclude papers whose mean |β| exceeds the ceiling
        if "mean_abs_beta" in r2_cos_df.columns:
            r2_cos_df = r2_cos_df[
                r2_cos_df["mean_abs_beta"].isna() |
                (r2_cos_df["mean_abs_beta"] <= BETA_CEILING)
            ]
        if len(r2_cos_df) >= 10:
            st.subheader("Semantic Cosine vs Reported R²")
            st.caption(
                "Each dot is one paper. Tests whether the semantic level of a paper’s "
                "constructs predicts how much variance its model explains — independent "
                "of the ordinal within-paper signal. Pearson r = 0.335, p < 0.001."
            )

            col_sc1, col_sc2 = st.columns([3, 2])

            with col_sc1:
                # Scatter: mean cosine vs R²
                fig_cr = go.Figure()
                for v, col in VERDICT_COLORS.items():
                    sub = r2_cos_df[r2_cos_df.get("verdict", pd.Series([""] * len(r2_cos_df))) == v]                           if "verdict" in r2_cos_df.columns else pd.DataFrame()
                    if len(sub) == 0:
                        sub = r2_cos_df  # fallback: plot all in one colour
                        col = "#60a5fa"
                    hover = sub.apply(
                        lambda r: _fmt_author(r.get("authors",""), r.get("year","")), axis=1
                    )
                    fig_cr.add_trace(go.Scatter(
                        x=sub["mean_cosine"], y=sub["avg_empirical_r2"],
                        mode="markers",
                        name=v,
                        marker=dict(size=7, color=col, opacity=0.75),
                        customdata=hover,
                        hovertemplate="<b>%{customdata}</b><br>"
                                      "mean cosine %{x:.3f}<br>"
                                      "R² %{y:.3f}<extra></extra>"
                    ))
                    if len(sub) == r2_cos_df.shape[0]:
                        break  # only plot once in fallback mode

                # OLS trend
                mc_r2 = r2_cos_df["mean_cosine"].values
                r2_v  = r2_cos_df["avg_empirical_r2"].values
                slope_cr, intercept_cr = np.polyfit(mc_r2, r2_v, 1)
                xs_cr = np.linspace(mc_r2.min(), mc_r2.max(), 50)
                r_cr, p_cr = stats.spearmanr(mc_r2, r2_v)

                fig_cr.add_trace(go.Scatter(
                    x=xs_cr, y=slope_cr * xs_cr + intercept_cr,
                    mode="lines", name="OLS trend", showlegend=False,
                    line=dict(color="#94a3b8", dash="dash", width=1.5)
                ))

                # Smedslund benchmark line
                fig_cr.add_hline(
                    y=SMEDSLUND_R2_BENCH, line_dash="dot", line_color="#f59e0b",
                    annotation_text=f"Smedslund benchmark {SMEDSLUND_R2_BENCH}",
                    annotation_position="top right",
                    annotation_font=dict(color="#f59e0b", size=11)
                )

                fig_cr.add_annotation(
                    x=0.01, y=0.97, xref="paper", yref="paper",
                    text=f"Spearman ρ = {r_cr:.3f}   p = {p_cr:.3e}   n = {len(r2_cos_df)} papers",
                    showarrow=False,
                    font=dict(color="#94a3b8", size=12),
                    align="left", bgcolor="rgba(0,0,0,0.3)", borderpad=4
                )

                fig_cr.update_layout(
                    xaxis_title="Mean cosine similarity (definition level)",
                    yaxis_title="Reported R²",
                    height=360,
                    legend=dict(font=dict(color=_FONT_COL, size=11),
                                bgcolor="rgba(0,0,0,0.3)",
                                bordercolor="#475569", borderwidth=1),
                    **_BASE_LAYOUT
                )
                st.plotly_chart(fig_cr, use_container_width=True)

            with col_sc2:
                # R²/cosine² ratio stability across percentiles
                st.markdown("**R² / cosine² ratio across percentiles**")
                st.caption(
                    "A stable ratio means R² scales as a near-constant multiple of "
                    "cosine² throughout the distribution — confirming the factor "
                    "loading derivation holds empirically."
                )
                pcts  = [10, 25, 50, 75, 90]
                ratios, cos_pcts, r2_pcts = [], [], []
                for p_val in pcts:
                    c_p = float(np.percentile(r2_cos_df["mean_cosine"], p_val))
                    r_p = float(np.percentile(r2_cos_df["avg_empirical_r2"], p_val))
                    ratio = r_p / (c_p ** 2) if c_p > 0 else None
                    cos_pcts.append(c_p)
                    r2_pcts.append(r_p)
                    ratios.append(ratio)

                mean_ratio = float(np.nanmean(ratios))

                fig_ratio = go.Figure()
                fig_ratio.add_trace(go.Scatter(
                    x=[f"{p}th" for p in pcts],
                    y=ratios,
                    mode="markers+lines",
                    marker=dict(size=10, color="#818cf8"),
                    line=dict(color="#818cf8", width=2),
                    name="R² / cosine²"
                ))
                fig_ratio.add_hline(
                    y=mean_ratio, line_dash="dash", line_color="#f59e0b",
                    annotation_text=f"Mean ratio {mean_ratio:.2f}",
                    annotation_font=dict(color="#f59e0b", size=11)
                )
                fig_ratio.update_layout(
                    xaxis_title="Percentile",
                    yaxis_title="R² / cosine²",
                    yaxis=dict(range=[0, max(ratios) * 1.3] if ratios else [0, 3]),
                    height=200,
                    showlegend=False,
                    **_BASE_LAYOUT
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

                # Key metrics
                m_a, m_b = st.columns(2)
                m_a.metric("Mean R²/cos² ratio",
                           f"{mean_ratio:.2f}",
                           help="Stable across percentiles — R² ≈ 1.63 × mean_cosine²")
                pred_smed = slope_cr * 0.49 + intercept_cr
                m_b.metric("Predicted R² at cosine 0.49",
                           f"{pred_smed:.3f}",
                           help="Cosine 0.49 = 0.70², Smedslund’s loading ceiling. "
                                f"Smedslund benchmark = {SMEDSLUND_R2_BENCH}")

                ols_text = (
                    f"**OLS:** R² = {slope_cr:.3f} × cosine + {intercept_cr:.3f}  \n\n"
                    f"At corpus mean cosine 0.41: predicted R² = {slope_cr*0.41+intercept_cr:.3f}  \n\n"
                    f"At Smedslund cosine 0.49: predicted R² = {pred_smed:.3f}"
                )
                st.markdown(ols_text)

    # ── Download (admin only) ────────────────────────────────────────────
    st.divider()
    st.subheader("Corpus Data")
    st.markdown(
        "The underlying dataset is available to researchers on request. "
        "Contact **Ketil Arnulf** at BI Norwegian Business School "
        "(ketil.arnulf@bi.no)."
    )
    with st.expander("Admin download (password required)"):
        admin_pw = st.text_input("Admin password", type="password", key="admin_pw")
        correct_pw = st.secrets.get("ADMIN_PASSWORD", "") or os.environ.get("ADMIN_PASSWORD", "")
        if admin_pw and correct_pw and admin_pw == correct_pw:
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "Download all pairs (CSV)",
                    data=pairs_df.to_csv(index=False),
                    file_name="pooled_db_pathB_web.csv",
                    mime="text/csv",
                    help="Compatible with Jupyter pipeline pooled_db_pathB.csv format"
                )
            with col_dl2:
                st.download_button(
                    "Download paper summaries (CSV)",
                    data=summary_df.to_csv(index=False),
                    file_name="batch_theory_summary_web.csv",
                    mime="text/csv",
                    help="Compatible with Jupyter pipeline batch_theory_summary.csv format"
                )
        elif admin_pw:
            st.error("Incorrect password.")


def main():
    page, ant_key, oai_key = sidebar()
    # Track whether the user supplied their own keys (vs host keys)
    host_ant_check, _ = _get_host_keys()
    ant_key_is_user = bool(ant_key) and (ant_key != host_ant_check)

    if page == "Corpus Dashboard":
        show_dashboard()
        return

    if page == "How to Read Results":
        show_guide()
        return

    st.title("Semantic Predetermination Detector")
    st.markdown(
        "Upload an empirical psychology paper to test whether its findings "
        "are predictable from the semantic content of its construct definitions."
    )

    if not ant_key or not oai_key:
        st.info("👈  Enter both API keys in the sidebar to begin.")
        return

    # ── Retrieve a past analysis ──────────────────────────────────────────
    with st.expander("📋 Retrieve a past analysis from the database"):
        st.caption(
            "Search by author family name, keyword from the title, or publication year. "
            "Selecting a paper regenerates the full report instantly — no API calls needed."
        )
        search_q = st.text_input("Search author / title / year", key="retrieve_search",
                                  placeholder="e.g. Andreassen  or  2008  or  leadership")
        if search_q and len(search_q.strip()) >= 2:
            hits = _search_theory_supabase(search_q.strip())
            if not hits:
                st.info("No matching papers found in the database.")
            else:
                options = {
                    f"{_fmt_author(h.get('authors',''), h.get('year',''))} — {h.get('title','')[:55]}": h["file"]
                    for h in hits
                }
                selected_label = st.selectbox(
                    f"{len(hits)} paper(s) found — select to load:",
                    list(options.keys()), key="retrieve_select"
                )
                if st.button("🔄 Load this paper's report", key="retrieve_btn"):
                    selected_file = options[selected_label]
                    with st.spinner("Fetching from database…"):
                        retrieved_theory = _fetch_theory_supabase(selected_file)
                    if retrieved_theory is None:
                        st.error("Could not retrieve theory data for this paper.")
                    else:
                        # Use OpenAI key if we have one; cache avoids the call when available
                        _retr_oai = oai_key or ""
                        # Recompute Stage 2 (uses cache if embedded, else needs oai_key)
                        try:
                            with st.spinner("Recomputing semantic analysis…"):
                                retrieved_stage2 = run_stage2(
                                    retrieved_theory, _retr_oai,
                                    log=lambda m: None   # silent
                                )
                            retrieved_verdict = compute_verdict(retrieved_stage2)
                            # Store in session state to trigger the results display below
                            st.session_state["theory"]       = retrieved_theory
                            st.session_state["stage2"]       = retrieved_stage2
                            st.session_state["verdict_data"] = retrieved_verdict
                            st.session_state["mean_cosine_paper"] = retrieved_verdict[5]
                            st.session_state["mean_abs_beta_paper"] = retrieved_verdict[6]
                            st.session_state.pop("db_msg", None)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Stage 2 recompute failed: {e}")
        st.caption(
            "Only papers submitted with the ‘Contribute results’ option ticked "
            "are stored in the database."
        )

    st.divider()

    st.warning(
        "**Eligible studies** are quantitative empirical papers with directional "
        "hypotheses between distinct constructs and reported effect sizes (β, r, "
        "path coefficients). "
        "**Conceptual papers, meta-analyses, pure scale validation studies, and "
        "qualitative research will not pass the eligibility screening** and the "
        "analysis will stop at that point. When in doubt, check the "
        "*How to Read Results* page for eligibility criteria."
    )

    uploaded = st.file_uploader(
        "Upload research article (PDF)", type=["pdf"],
        help="The paper should be an empirical study with directional hypotheses "
             "and reported effect sizes."
    )
    if uploaded is None:
        return

    pdf_bytes = uploaded.read()

    st.info(
        "⏱ **Analysis takes 1–3 minutes** per paper — Claude reads the full PDF "
        "and OpenAI computes embeddings. Please keep this tab open while it runs."
    )

    save_to_db = st.checkbox(
        "Contribute results to the corpus after analysis",
        value=True,
        help="Saves this paper's results to the shared Supabase corpus (when online) "
             "or to local CSV files on your Mac. Contributions help grow the global "
             "benchmark that other researchers compare against."
    )

    if st.button("Analyse Paper", type="primary", use_container_width=True):
        for k in ("theory", "stage2", "verdict_data"):
            st.session_state.pop(k, None)

        # ── Stage 0 ───────────────────────────────────────────────────────
        with st.status("Starting analysis...", expanded=True) as status:
            logs = []
            log_box = st.empty()

            def log(msg):
                logs.append(msg)
                log_box.code("\n".join(logs), language=None)

            # Pre-screen
            v0, reason0 = prescreening(pdf_bytes)
            log(f"[Stage 0] {v0}: {reason0}")

            # ── Stage 1 ───────────────────────────────────────────────────
            status.update(label="Extracting theory with Claude…")
            try:
                theory = extract_theory(pdf_bytes, ant_key, log)
                st.session_state["theory"] = theory
            except Exception as e:
                status.update(label="Extraction failed", state="error")
                st.error(f"Stage 1 error: {e}")
                return

            elig = theory.get("eligibility", {})
            if not elig.get("eligible", True):
                status.update(label="Paper ineligible", state="error")
                log(f"[Ineligible] {elig.get('exclusion_reason','')}")
                return

            # ── Stage 2 ───────────────────────────────────────────────────
            status.update(label="Running semantic analysis…")
            try:
                stage2 = run_stage2(theory, oai_key, log)
                if stage2 is None:
                    status.update(label="Too few constructs", state="error")
                    return
                st.session_state["stage2"] = stage2
            except Exception as e:
                status.update(label="Stage 2 failed", state="error")
                st.error(f"Stage 2 error: {e}")
                return

            verdict_data = compute_verdict(stage2)
            st.session_state["verdict_data"] = verdict_data
            # Store mean_cosine for DB save (verdict_data[5])
            st.session_state["mean_cosine_paper"] = verdict_data[5]
            st.session_state["mean_abs_beta_paper"] = verdict_data[6]

            # ── Increment free-access usage counter if applicable ──────
            if not ant_key_is_user:
                _increment_daily_usage()

            # ── Local DB save (opt-in) ─────────────────────────────────
            if save_to_db:
                status.update(label="Saving to local database…")
                try:
                    pairs_added, already_existed, backend, n_papers, n_pairs = save_to_local_db(
                        theory, stage2, uploaded.name
                    )
                    if already_existed:
                        log(f"[DB] Paper already in {backend} database — not duplicated.")
                    else:
                        log(f"[DB] Saved {pairs_added} pairs to {backend}. "
                            f"Corpus now: {n_papers} papers, {n_pairs} pairs.")
                    st.session_state["db_msg"] = (pairs_added, already_existed, backend, n_papers, n_pairs)
                except Exception as e:
                    log(f"[DB] Save failed: {e}")

            status.update(label="Analysis complete ✓", state="complete")

    # ── Results display ───────────────────────────────────────────────────────
    if "theory" not in st.session_state:
        return

    theory = st.session_state["theory"]
    stage2 = st.session_state.get("stage2")
    meta   = theory.get("study_metadata", {})
    elig   = theory.get("eligibility", {})

    st.divider()

    # DB save confirmation
    if "db_msg" in st.session_state:
        pairs_added, already_existed, backend, n_papers, n_pairs = st.session_state["db_msg"]
        if already_existed:
            st.info(f"📂 This paper was already in the {backend} database and was not duplicated.")
        else:
            st.success(
                f"📂 Results saved to **{backend}**.  "
                f"**{pairs_added} construct pairs** added.  "
                f"Growing corpus now contains **{n_papers} papers** "
                f"and **{n_pairs} pairs** total."
            )

    # Paper header
    st.subheader(meta.get("title", "Untitled"))
    st.caption(
        f"**{meta.get('authors','')}** · "
        f"*{meta.get('journal','')}* · "
        f"{meta.get('year','')}"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Study type",    meta.get("study_type", "—").title())
    c2.metric("N respondents", meta.get("n_respondents") or "—")
    c3.metric("Model type",    meta.get("model_type", "—").replace("_", " ").title())
    c4.metric("Constructs",    len(theory.get("constructs", [])))

    # Ineligibility (reached only via JSON already in session from a prior run)
    if not elig.get("eligible", True):
        st.error(f"**Ineligible:** {elig.get('exclusion_reason','')}")
        st.caption(f"Category: {elig.get('exclusion_category','unknown')}")
        return

    # Verdict banner
    if stage2 and "verdict_data" in st.session_state:
        label, icon, color, signals, reasons, mean_cos, mean_beta = st.session_state["verdict_data"]
        inflation = label == "SEMANTIC INFLATION"
        extra_note = (
            f"  mean cosine {mean_cos:.3f} · mean |β| {mean_beta:.3f}"
            if inflation and mean_cos is not None else ""
        )
        st.markdown(
            f'<div style="background:{color}22;border-left:5px solid {color};'
            f'padding:0.9em 1.4em;border-radius:6px;margin:0.8em 0">'
            f'<span style="font-size:1.35em;font-weight:700;color:{color}">'
            f'{icon} {label}</span>'
            f'<span style="font-size:0.85em;color:#94a3b8;margin-left:1em">'
            f'({signals}/3 ordinal signals{extra_note})</span></div>',
            unsafe_allow_html=True
        )
        with st.expander("Verdict rationale"):
            for r in reasons:
                st.markdown(f"- {r}")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_extract, tab_analysis, tab_json = st.tabs(
        ["📖 Extraction", "📊 Semantic Analysis", "🗂 Raw JSON"]
    )

    # ── Tab 1: Extraction ─────────────────────────────────────────────────────
    with tab_extract:
        st.subheader("Extracted Constructs & Definitions")
        for c in theory.get("constructs", []):
            role_badge = c.get("role", "")
            with st.expander(f"**{c['name']}**  —  *{role_badge}*"):
                st.markdown(c.get("theoretical_definition", "*(no definition)*"))
                col_a, col_b, col_c = st.columns(3)
                col_a.caption(f"📍 Source: {c.get('definition_source','—')}")
                col_b.caption(f"📋 Items: {c.get('item_availability','—').replace('_',' ')}")
                col_c.caption(f"📏 Scale: {c.get('scale_name','—') or '—'}")

        if theory.get("mediation_chains"):
            st.subheader("Mediation Chains")
            for ch in theory["mediation_chains"]:
                st.markdown(
                    f"**{ch['predictor']}** → **{ch['mediator']}** → "
                    f"**{ch['outcome']}**  "
                    f"<span style='color:#94a3b8'>({ch.get('type','').replace('_',' ')})</span>",
                    unsafe_allow_html=True
                )

        st.subheader("Hypothesised Relationships")
        rels = theory.get("relationships", [])
        if rels:
            rel_df = pd.DataFrame([{
                "From":      r.get("from", ""),
                "To":        r.get("to", ""),
                "β":         r.get("effect_size"),
                "Type":      r.get("effect_type", ""),
                "Step":      r.get("regression_step"),
                "Sig.":      "✓" if r.get("significant") else "✗"
            } for r in rels])
            st.dataframe(rel_df, use_container_width=True, hide_index=True)
        else:
            st.info("No relationships extracted.")

        # Validation warnings
        for w in theory.get("_validation", {}).get("warnings", []):
            st.warning(w)
        for e in theory.get("_validation", {}).get("errors", []):
            st.error(e)

    # ── Tab 2: Semantic analysis ──────────────────────────────────────────────
    with tab_analysis:
        if not stage2:
            st.info("Stage 2 results not available.")
        else:
            ab  = stage2["ab"]
            abc = stage2["abc"]
            sp  = stage2["spearman"]

            # Key metrics row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "A>B Concordance",
                f"{100*ab['rate']:.1f}%" if ab["rate"] is not None else "—",
                help="% of pair-comparisons where higher cosine → larger |β|. "
                     "Corpus benchmark: 60.8%"
            )
            m2.metric(
                "A>B>C Pass rate",
                f"{abc['pass']}/{abc['total']}  ({100*abc['rate']:.1f}%)"
                if abc["rate"] is not None else "N/A",
                help="Mediation chains passing the semantic gradient test. "
                     "Corpus benchmark: 54.1%"
            )
            m3.metric(
                "Within-study ρ (signed)",
                f"{sp['signed_rho']:.3f}" if sp["signed_rho"] is not None else "—",
                delta=f"p = {sp['signed_p']:.3f}" if sp["signed_p"] is not None else None,
                help=f"Spearman between cosine and signed β. n = {sp['n_pairs']} pairs. "
                     "Corpus pooled ρ = 0.259."
            )
            m4.metric(
                "Mean Empirical R²",
                f"{stage2['avg_r2']:.3f}" if stage2["avg_r2"] is not None else "—",
                help=f"Average explained variance. Smedslund benchmark = {SMEDSLUND_R2_BENCH}"
            )

            st.divider()

            # Cosine heatmap (full width)
            st.plotly_chart(
                plot_cosine_heatmap(stage2["constructs"], stage2["cosine_matrix"]),
                use_container_width=True
            )

            # A>B bar + scatter side by side
            col_l, col_r = st.columns(2)
            with col_l:
                if ab["rate"] is not None:
                    st.plotly_chart(plot_ab_bar(ab), use_container_width=True)
                else:
                    st.info("A>B concordance not computable (fewer than 2 pairs).")
            with col_r:
                fig_sc = plot_scatter(stage2["pair_data"])
                if fig_sc:
                    st.plotly_chart(fig_sc, use_container_width=True)

            # A>B>C chains
            if abc["results"]:
                st.plotly_chart(plot_abc_chains(abc["results"]),
                                use_container_width=True)

                st.subheader("Mediation Chain Detail")
                abc_df = pd.DataFrame([{
                    "Chain":      r["chain"],
                    "Type":       r.get("chain_type", "").replace("_", " "),
                    "cos(A,B)":   f"{r['cos_ab']:.3f}",
                    "cos(B,C)":   f"{r['cos_bc']:.3f}",
                    "cos(A,C)":   f"{r['cos_ac']:.3f}",
                    "Gradient":   "✓ PASS" if r["passes"] else "✗ FAIL"
                } for r in abc["results"]])
                st.dataframe(abc_df, use_container_width=True, hide_index=True)

            # R² gauge
            if stage2["avg_r2"] is not None:
                st.plotly_chart(plot_r2_gauge(stage2["avg_r2"]),
                                use_container_width=True)

            # Data quality warnings
            # Flag suspiciously large effect sizes in this paper
            suspicious_pairs = [
                p for p in stage2.get("pair_data", [])
                if abs(p.get("unsigned_effect", 0)) > BETA_CEILING
            ]
            if suspicious_pairs:
                sp_labels = ", ".join(
                    f"{p['from']} → {p['to']} (|β|={p['unsigned_effect']:.2f})"
                    for p in suspicious_pairs
                )
                st.warning(
                    f"⚠ **{len(suspicious_pairs)} relationship(s) have |β| > {BETA_CEILING}** "
                    f"and are likely non-standardised coefficients (odds ratios, logistic β, or "
                    f"unstandardised OLS): {sp_labels}. "
                    f"These are excluded from the Spearman and concordance calculations above "
                    f"but retained in the raw database."
                )

            if stage2.get("cosine_range") is not None and stage2["cosine_range"] < 0.05:
                st.warning(
                    f"⚠ Compressed cosine space (range = {stage2['cosine_range']:.3f}). "
                    "All constructs are very close in embedding space — "
                    "Spearman and concordance estimates may be noise-dominated."
                )
            if sp["n_pairs"] < 5:
                st.info(
                    f"ℹ {sp['n_pairs']} construct pairs extracted. "
                    "Within-study Spearman is indicative only with fewer than 5 pairs. "
                    "The A>B concordance rate is the primary within-study criterion."
                )

            # Pair data table
            with st.expander("All construct pairs"):
                pair_df = pd.DataFrame([{
                    "From":    p["from"],
                    "To":      p["to"],
                    "Cosine":  f"{p['cosine']:.4f}",
                    "Signed β": p["signed_effect"],
                    "|β|":     p["unsigned_effect"],
                    "Type":    p.get("effect_type", ""),
                    "Step":    p.get("step")
                } for p in stage2["pair_data"]])
                st.dataframe(pair_df, use_container_width=True, hide_index=True)

    # ── Tab 3: Raw JSON ───────────────────────────────────────────────────────
    with tab_json:
        st.subheader("Theory extraction (Stage 1 output)")
        theory_display = copy.deepcopy(theory)
        theory_display.pop("_validation", None)
        st.json(theory_display, expanded=False)

        if stage2:
            st.subheader("Semantic analysis (Stage 2 output)")
            stage2_display = copy.deepcopy(stage2)
            stage2_display.pop("cosine_matrix", None)  # large — shown as heatmap
            st.json(stage2_display, expanded=False)

        # Download button for the full theory JSON
        fname = uploaded.name.replace(".pdf","") if uploaded else "theory1"
        st.download_button(
            label="⬇️  Download full theory JSON (constructs, definitions, relationships)",
            data=json.dumps(theory_display, indent=2),
            file_name=f"{fname}_theory1.json",
            mime="application/json",
            use_container_width=True,
            type="primary"
        )
        st.caption(
            "The JSON contains all extracted construct definitions, hypotheses, "
            "mediation chains, and relationships. Compatible with the Jupyter pipeline."
        )


if __name__ == "__main__":
    main()
