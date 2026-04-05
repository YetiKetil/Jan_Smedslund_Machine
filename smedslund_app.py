"""
Jan Smedslund Semantic Predetermination Detector
=================================================
Developed by Ketil Arnulf · BI Norwegian Business School
In memory of Jan Smedslund (1924–2024)

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

    log(f"Fetching embeddings for {len(names)} constructs (text-embedding-3-large)...")
    embeddings = get_embeddings(defs, openai_key)

    # Full cosine matrix
    n = len(names)
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

    if n_pairs >= 3:
        cosines          = [p["cosine"] for p in pair_data]
        signed_effects   = [p["signed_effect"] for p in pair_data]
        unsigned_effects = [p["unsigned_effect"] for p in pair_data]
        cosine_range     = max(cosines) - min(cosines)

        sr = stats.spearmanr(cosines, signed_effects)
        ur = stats.spearmanr(cosines, unsigned_effects)
        signed_rho,   signed_p   = float(sr.statistic), float(sr.pvalue)
        unsigned_rho, unsigned_p = float(ur.statistic), float(ur.pvalue)
        log(f"Spearman: signed ρ={signed_rho:.3f} (p={signed_p:.3f}), "
            f"unsigned ρ={unsigned_rho:.3f} (p={unsigned_p:.3f}), n={n_pairs} pairs.")
    else:
        log(f"Only {n_pairs} pairs — Spearman not computed (need ≥3).")

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

def compute_verdict(stage2):
    """
    Three signals; each above its threshold scores 1 point.
    2–3 signals → Semantically Structured
    1 signal     → Partially Structured
    0 signals    → Empirically Independent
    """
    ab_rate  = stage2["ab"]["rate"]
    abc_rate = stage2["abc"]["rate"]
    rho      = stage2["spearman"]["signed_rho"]
    n_pairs  = stage2["spearman"]["n_pairs"]

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

    if signals >= 2:
        label, icon, color = "SEMANTICALLY STRUCTURED",   "🔴", "#ef4444"
    elif signals == 1:
        label, icon, color = "PARTIALLY STRUCTURED",      "🟡", "#f59e0b"
    else:
        label, icon, color = "EMPIRICALLY INDEPENDENT",   "🟢", "#22c55e"

    return label, icon, color, signals, reasons


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



# ── Local database accumulation ──────────────────────────────────────────────
#
# Saves results to two CSV files in the same folder as this script,
# exactly matching the format of pooled_db_pathB.csv and
# batch_theory_summary.csv from the Jupyter pipeline.
# Safe to re-run: duplicate study_ids are not appended twice.

import os

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

def save_to_local_db(theory, stage2, source_filename):
    """
    Append this paper's results to the two local CSV databases.
    Returns (pairs_added, already_existed) tuple.
    """
    meta     = theory.get("study_metadata", {})
    authors  = meta.get("authors", "Unknown")
    year     = meta.get("year", "")
    study_id = f"{authors.split(',')[0].strip()} ({year})"

    # pooled_db_pathB.csv
    pooled = _load_or_create(POOLED_DB_PATH, POOLED_COLS)
    already_in_pooled = study_id in pooled["study_id"].values if len(pooled) else False

    pairs_added = 0
    if not already_in_pooled:
        new_rows = [{
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
        if new_rows:
            pooled = pd.concat([pooled, pd.DataFrame(new_rows)], ignore_index=True)
            pooled.to_csv(POOLED_DB_PATH, index=False)
            pairs_added = len(new_rows)

    # batch_theory_summary.csv
    summary = _load_or_create(SUMMARY_PATH, SUMMARY_COLS)
    already_in_summary = source_filename in summary["file"].values if len(summary) else False

    if not already_in_summary:
        ab  = stage2.get("ab", {})
        abc = stage2.get("abc", {})
        sp  = stage2.get("spearman", {})
        new_row = {
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
            "status":           "analysed"
        }
        summary = pd.concat([summary, pd.DataFrame([new_row])], ignore_index=True)
        summary.to_csv(SUMMARY_PATH, index=False)

    return pairs_added, already_in_pooled


# ── Streamlit UI ──────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("## 🔬 Jan Smedslund\nSemantic Detector")
        st.caption("*In memory of Jan Smedslund (1929–2026)*")
        st.divider()

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
                "Corpus benchmark (103 studies): signed ρ = 0.259, "
                "A>B concordance 60.8%, A>B>C 54.1%.\n\n"
                "*In memory of Jan Smedslund (1929–2026)*"
            )
        st.caption("Developed by Ketil Arnulf · BI Norwegian Business School")
    return ant_key, oai_key


def main():
    ant_key, oai_key = sidebar()

    st.title("Semantic Predetermination Detector")
    st.markdown(
        "Upload an empirical psychology paper to test whether its findings "
        "are predictable from the semantic content of its construct definitions."
    )

    if not ant_key or not oai_key:
        st.info("👈  Enter both API keys in the sidebar to begin.")
        return

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
        "Save results to local database after analysis",
        value=True,
        help="Appends this paper's results to pooled_db_pathB.csv and "
             "batch_theory_summary.csv in the same folder as smedslund_app.py. "
             "These files are compatible with your existing Jupyter pipeline."
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

            # ── Local DB save (opt-in) ─────────────────────────────────
            if save_to_db:
                status.update(label="Saving to local database…")
                try:
                    pairs_added, already_existed = save_to_local_db(
                        theory, stage2, uploaded.name
                    )
                    if already_existed:
                        log("[DB] Paper already in database — not duplicated.")
                    else:
                        log(f"[DB] Saved {pairs_added} pairs to pooled_db_pathB.csv "
                            f"and 1 row to batch_theory_summary.csv.")
                    st.session_state["db_msg"] = (pairs_added, already_existed)
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
        pairs_added, already_existed = st.session_state["db_msg"]
        if already_existed:
            st.info("📂 This paper was already in the local database and was not duplicated.")
        else:
            # Count current corpus size
            pooled_n = 0
            summary_n = 0
            try:
                import os
                if os.path.exists(POOLED_DB_PATH):
                    pooled_n = len(pd.read_csv(POOLED_DB_PATH))
                if os.path.exists(SUMMARY_PATH):
                    summary_n = len(pd.read_csv(SUMMARY_PATH))
            except Exception:
                pass
            st.success(
                f"📂 Results saved to local database.  "
                f"**{pairs_added} pairs** added.  "
                f"Database now contains **{summary_n} papers** and "
                f"**{pooled_n} construct pairs** total."
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
        label, icon, color, signals, reasons = st.session_state["verdict_data"]
        st.markdown(
            f'<div style="background:{color}22;border-left:5px solid {color};'
            f'padding:0.9em 1.4em;border-radius:6px;margin:0.8em 0">'
            f'<span style="font-size:1.35em;font-weight:700;color:{color}">'
            f'{icon} {label}</span>'
            f'<span style="font-size:0.85em;color:#94a3b8;margin-left:1em">'
            f'({signals}/3 signals above threshold)</span></div>',
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
        st.download_button(
            label="Download theory JSON",
            data=json.dumps(theory_display, indent=2),
            file_name=f"{uploaded.name.replace('.pdf','')}_theory1.json"
                      if uploaded else "theory1.json",
            mime="application/json"
        )


if __name__ == "__main__":
    main()
