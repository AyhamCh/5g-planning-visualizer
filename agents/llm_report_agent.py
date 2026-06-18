"""
LLM Report Agent — Génération rapport final 5G
================================================
- Lit le fichier txt combiné des agents
- Interroge Ollama (Qwen2.5:3b) en local
- Génère un PDF structuré avec ReportLab

Prérequis :
    ollama pull qwen2.5:3b
    pip install reportlab requests
"""

import requests
import json
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
INPUT_TXT    = "outputs/agents_summary.txt"       # ton fichier combiné
OUTPUT_PDF   = "outputs/rapport_final_5g.pdf"

# ─────────────────────────────────────────────
# COULEURS
# ─────────────────────────────────────────────

C_BLUE_DARK  = HexColor("#0f3460")
C_RED        = HexColor("#e94560")
C_BLUE_MID   = HexColor("#16213e")
C_GREY_LIGHT = HexColor("#f5f5f5")
C_GREY_MED   = HexColor("#cccccc")
C_GREEN      = HexColor("#27ae60")
C_ORANGE     = HexColor("#e67e22")


# ─────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────

SYSTEM_CONTEXT = """Tu es un expert en déploiement de réseaux 5G.
Tu analyses des données techniques issues d'agents d'analyse géospatiale
et tu rédiges des rapports professionnels clairs et structurés en français.
Réponds UNIQUEMENT en JSON valide, sans texte avant ou après.
"""

def build_prompt_executive_summary(txt: str) -> str:
    return f"""{SYSTEM_CONTEXT}

Voici les résultats combinés de 4 agents d'analyse 5G :
{txt}

Génère un résumé exécutif JSON avec exactement cette structure :
{{
  "titre_zone": "nom descriptif de la zone analysée",
  "contexte": "2-3 phrases décrivant le contexte urbain de la zone",
  "problematique_principale": "1 phrase résumant le problème réseau principal",
  "chiffres_cles": [
    {{"label": "...", "valeur": "...", "unite": "..."}},
    {{"label": "...", "valeur": "...", "unite": "..."}},
    {{"label": "...", "valeur": "...", "unite": "..."}},
    {{"label": "...", "valeur": "...", "unite": "..."}},
    {{"label": "...", "valeur": "...", "unite": "..."}}
  ],
  "conclusion_executive": "2-3 phrases de conclusion pour un décideur"
}}"""


def build_prompt_sites(txt: str) -> str:
    return f"""{SYSTEM_CONTEXT}

Voici les résultats des agents 5G incluant le rapport SitePlacementAgent :
{txt}

"Pour CHAQUE site mentionné dans le rapport SitePlacementAgent, génère une analyse JSON. Inclus TOUS les sites sans exception :"
{{
  "sites": [
    {{
      "numero": 1,
      "priorite": "CRITIQUE" ou "HAUTE" ou "MOYENNE",
      "type_site": "type technique",
      "urban_class": "classe urbaine",
      "score": "valeur numérique",
      "deficit_gbps": "valeur",
      "justification_technique": "2-3 phrases de justification technique détaillée",
      "consignes_deploiement": [
        "consigne 1 concrète et actionnable",
        "consigne 2 concrète et actionnable",
        "consigne 3 concrète et actionnable"
      ],
      "risques": "1 phrase sur les risques ou contraintes de déploiement"
    }}
  ]
}}"""


def build_prompt_recommendations(txt: str) -> str:
    return f"""{SYSTEM_CONTEXT}

Voici les résultats des agents 5G :
{txt}

Génère des recommandations stratégiques JSON :
{{
  "recommandations": [
    {{
      "categorie": "Court terme (0-3 mois)",
      "actions": [
        "action 1",
        "action 2",
        "action 3"
      ]
    }},
    {{
      "categorie": "Moyen terme (3-12 mois)",
      "actions": [
        "action 1",
        "action 2",
        "action 3"
      ]
    }},
    {{
      "categorie": "Long terme (12+ mois)",
      "actions": [
        "action 1",
        "action 2"
      ]
    }}
  ],
  "note_finale": "1-2 phrases de clôture professionnelle"
}}"""


def build_prompt_one_site(site_raw: str, site_num: int) -> str:
    return f"""{SYSTEM_CONTEXT}

Voici les données du site #{site_num} issu du rapport SitePlacementAgent :
{site_raw}

Génère une analyse JSON pour CE site uniquement :
{{
  "numero": {site_num},
  "priorite": "CRITIQUE" ou "HAUTE" ou "MOYENNE",
  "type_site": "type technique du site",
  "urban_class": "classe urbaine",
  "score": "valeur du score composite",
  "deficit_gbps": "valeur du déficit",
  "justification_technique": "2-3 phrases de justification technique détaillée",
  "consignes_deploiement": [
    "consigne 1 concrète et actionnable",
    "consigne 2 concrète et actionnable",
    "consigne 3 concrète et actionnable"
  ],
  "risques": "1 phrase sur les risques ou contraintes de déploiement"
}}"""

# ─────────────────────────────────────────────
# APPEL OLLAMA
# ─────────────────────────────────────────────

def call_ollama(prompt: str, label: str = "") -> dict:
    """Appelle Ollama et retourne le JSON parsé."""
    print(f"  → LLM [{label}]...", end=" ", flush=True)

    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.3,
            "num_predict": 1500,
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Ollama non accessible. Lance : ollama serve"
        )

    raw = resp.json().get("response", "")

    print("\n")
    print("=" * 80)
    print(f"RAW RESPONSE [{label}]")
    print("=" * 80)
    print(raw)
    print("=" * 80)
    print("\n")

    # Extraire le JSON (ignorer texte parasite)
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"Pas de JSON dans la réponse LLM [{label}]:\n{raw[:300]}")

    json_str = raw[start:end]
    try:
        result = json.loads(json_str)
        print("OK")
        return result
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON invalide [{label}]: {e}\n{json_str[:300]}")


# ─────────────────────────────────────────────
# CONSTRUCTION PDF
# ─────────────────────────────────────────────

def build_pdf(exec_data: dict, sites_data: dict, reco_data: dict, output_path: str):
    """Construit le PDF final avec ReportLab."""

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Rapport Final 5G",
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Styles personnalisés ──────────────────

    s_title = ParagraphStyle(
        "MainTitle",
        fontSize=22, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=6,
    )
    s_subtitle = ParagraphStyle(
        "SubTitle",
        fontSize=11, textColor=C_GREY_MED, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=4,
    )
    s_section = ParagraphStyle(
        "SectionHead",
        fontSize=13, textColor=white, fontName="Helvetica-Bold",
        alignment=TA_LEFT, spaceBefore=4, spaceAfter=4,
        leftIndent=0,
    )
    s_body = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=9.5, textColor=black,
        leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
    )
    s_body_bold = ParagraphStyle(
        "BodyBold", parent=s_body,
        fontName="Helvetica-Bold", textColor=C_BLUE_DARK,
    )
    s_bullet = ParagraphStyle(
        "Bullet", parent=s_body,
        leftIndent=14, bulletIndent=4, spaceBefore=2, spaceAfter=2,
    )
    s_small = ParagraphStyle(
        "Small", parent=s_body,
        fontSize=8, textColor=HexColor("#666666"),
    )

    def section_header(title: str):
        """Bloc titre de section avec fond coloré."""
        tbl = Table([[Paragraph(title, s_section)]], colWidths=[17*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C_BLUE_DARK),
            ("ROUNDEDCORNERS", [4]),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        return tbl

    def priority_badge(p: str):
        colors = {"CRITIQUE": C_RED, "HAUTE": C_ORANGE, "MOYENNE": C_GREEN}
        c = colors.get(p.upper(), C_BLUE_DARK)
        st = ParagraphStyle("Badge", fontSize=8, textColor=white,
                            fontName="Helvetica-Bold", alignment=TA_CENTER)
        tbl = Table([[Paragraph(p.upper(), st)]], colWidths=[3*cm])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), c),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        return tbl

    # ══════════════════════════════════════════
    # PAGE DE COUVERTURE
    # ══════════════════════════════════════════

    cover_table = Table(
        [[Paragraph("RAPPORT FINAL — DÉPLOIEMENT 5G", s_title)],
         [Paragraph(exec_data.get("titre_zone", "Analyse Réseau"), s_subtitle)],
         [Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", s_subtitle)]],
        colWidths=[17*cm]
    )
    cover_table.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_BLUE_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(cover_table)
    story.append(Spacer(1, 0.6*cm))

    # Badge "Confidentiel / Usage Interne"
    badge_tbl = Table(
        [[Paragraph("USAGE INTERNE — STAGE AMARIS 2026", s_small)]],
        colWidths=[17*cm]
    )
    badge_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_GREY_LIGHT),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("LINEBELOW",     (0,0), (-1,-1), 1, C_GREY_MED),
    ]))
    story.append(badge_tbl)
    story.append(Spacer(1, 0.8*cm))

    # ══════════════════════════════════════════
    # 1. RÉSUMÉ EXÉCUTIF
    # ══════════════════════════════════════════

    story.append(section_header("1.  RÉSUMÉ EXÉCUTIF"))
    story.append(Spacer(1, 0.3*cm))

    story.append(Paragraph(exec_data.get("contexte", ""), s_body))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"<b>Problématique principale :</b> {exec_data.get('problematique_principale','')}",
        s_body
    ))
    story.append(Spacer(1, 0.4*cm))

    # Chiffres clés — tableau 5 colonnes
    kpis = exec_data.get("chiffres_cles", [])
    if kpis:
        kpi_header = [Paragraph("<b>Indicateur</b>", s_body_bold),
                      Paragraph("<b>Valeur</b>",      s_body_bold),
                      Paragraph("<b>Unité</b>",        s_body_bold)]
        kpi_rows = [kpi_header] + [
            [Paragraph(k.get("label",""), s_body),
             Paragraph(f"<b>{k.get('valeur','')}</b>", s_body_bold),
             Paragraph(k.get("unite",""), s_small)]
            for k in kpis
        ]
        kpi_tbl = Table(kpi_rows, colWidths=[9*cm, 4*cm, 4*cm])
        kpi_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C_BLUE_MID),
            ("TEXTCOLOR",     (0,0), (-1,0), white),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_GREY_LIGHT, white]),
            ("GRID",          (0,0), (-1,-1), 0.3, C_GREY_MED),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ]))
        story.append(kpi_tbl)
        story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph(exec_data.get("conclusion_executive",""), s_body))
    story.append(Spacer(1, 0.6*cm))

    # ══════════════════════════════════════════
    # 2. SITES RECOMMANDÉS
    # ══════════════════════════════════════════

    story.append(section_header("2.  SITES RECOMMANDÉS — ANALYSE & CONSIGNES"))
    story.append(Spacer(1, 0.4*cm))

    for site in sites_data.get("sites", []):
        num      = site.get("numero", "?")
        priorite = site.get("priorite", "MOYENNE")
        stype    = site.get("type_site", "")
        uc       = site.get("urban_class", "")
        score    = site.get("score", "")
        deficit  = site.get("deficit_gbps", "")

        # En-tête site
        header_data = [
            [Paragraph(f"<b>Site #{num}</b>", s_body_bold),
             priority_badge(priorite),
             Paragraph(f"<b>{stype}</b> · {uc}", s_body),
             Paragraph(f"Score : <b>{score}</b> | Déficit : <b>{deficit} Gbps</b>", s_small)],
        ]
        h_tbl = Table(header_data, colWidths=[2.5*cm, 3*cm, 6.5*cm, 5*cm])
        h_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_GREY_LIGHT),
            ("LINEBELOW",     (0,0), (-1,-1), 1.5, C_BLUE_DARK),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(h_tbl)
        story.append(Spacer(1, 0.15*cm))

        # Justification
        story.append(Paragraph(
            f"<b>Justification technique :</b> {site.get('justification_technique','')}",
            s_body
        ))
        story.append(Spacer(1, 0.15*cm))

        # Consignes de déploiement
        story.append(Paragraph("<b>Consignes de déploiement :</b>", s_body_bold))
        for consigne in site.get("consignes_deploiement", []):
            story.append(Paragraph(f"• {consigne}", s_bullet))

        # Risques
        risques = site.get("risques", "")
        if risques:
            story.append(Paragraph(
                f"<b>⚠ Risques / contraintes :</b> {risques}", s_small
            ))

        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_GREY_MED))
        story.append(Spacer(1, 0.3*cm))

    # ══════════════════════════════════════════
    # 3. RECOMMANDATIONS STRATÉGIQUES
    # ══════════════════════════════════════════

    story.append(PageBreak())
    story.append(section_header("3.  RECOMMANDATIONS STRATÉGIQUES"))
    story.append(Spacer(1, 0.4*cm))

    reco_colors = [C_RED, C_ORANGE, C_BLUE_DARK]
    for i, reco in enumerate(reco_data.get("recommandations", [])):
        cat     = reco.get("categorie", "")
        actions = reco.get("actions", [])
        col     = reco_colors[i % len(reco_colors)]

        # Titre catégorie
        cat_style = ParagraphStyle(
            f"CatStyle{i}", fontSize=10, textColor=white,
            fontName="Helvetica-Bold", alignment=TA_LEFT,
        )
        cat_tbl = Table([[Paragraph(cat, cat_style)]], colWidths=[17*cm])
        cat_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), col),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        story.append(cat_tbl)

        for action in actions:
            story.append(Paragraph(f"→  {action}", s_bullet))
        story.append(Spacer(1, 0.3*cm))

    # Note finale
    note = reco_data.get("note_finale", "")
    if note:
        story.append(Spacer(1, 0.3*cm))
        note_tbl = Table(
            [[Paragraph(f"<i>{note}</i>", s_body)]],
            colWidths=[17*cm]
        )
        note_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_GREY_LIGHT),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("LINERIGHT",     (0,0), (0,-1), 3, C_BLUE_DARK),
        ]))
        story.append(note_tbl)

    # ── Pied de page (footer manuel) ─────────
    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=C_BLUE_DARK))
    story.append(Spacer(1, 0.2*cm))
    footer_style = ParagraphStyle(
        "Footer", fontSize=7.5, textColor=HexColor("#888888"),
        alignment=TA_CENTER,
    )
    story.append(Paragraph(
        f"Rapport généré automatiquement par LLM Report Agent — "
        f"Pipeline 5G AI — Stage Amaris 2026 — {datetime.now().strftime('%d/%m/%Y')}",
        footer_style
    ))

    doc.build(story)
    print(f"\n  ✓ PDF généré : {output_path}")



def extract_sites_blocks(txt: str) -> list[str]:
    """Extrait chaque bloc Site #XX du rapport SitePlacementAgent."""
    import re
    blocks = re.split(r'(?=\s*Site #\d+)', txt)
    return [b.strip() for b in blocks if re.match(r'\s*Site #\d+', b)]

# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────


def main():
    print("=" * 60)
    print("  LLM Report Agent — Pipeline 5G")
    print("=" * 60)

    # 1. Lecture du fichier txt combiné
    txt_path = Path(INPUT_TXT)
    if not txt_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {txt_path}")

    txt = txt_path.read_text(encoding="utf-8")
    # Limiter la taille pour le modèle 3B (context window ~4k tokens)
    txt_trimmed = txt[:6000]
    print(f"  Fichier lu : {len(txt)} caractères (trimmed: {len(txt_trimmed)})")

    # 2. Appels LLM
    print("\n[ GÉNÉRATION LLM ]")
    exec_data  = call_ollama(build_prompt_executive_summary(txt_trimmed), "Résumé exécutif")
    site_blocks = extract_sites_blocks(txt)
    sites_list = []
    for i, block in enumerate(site_blocks, 1):
        site_result = call_ollama(build_prompt_one_site(block, i), f"Site #{i}")
        sites_list.append(site_result)
    sites_data = {"sites": sites_list}
    reco_data  = call_ollama(build_prompt_recommendations(txt_trimmed),   "Recommandations")

    # 3. Construction PDF
    print("\n[ CONSTRUCTION PDF ]")
    Path(OUTPUT_PDF).parent.mkdir(parents=True, exist_ok=True)
    build_pdf(exec_data, sites_data, reco_data, OUTPUT_PDF)

    print("\n" + "=" * 60)
    print(f"  TERMINÉ → {OUTPUT_PDF}")
    print("=" * 60)


if __name__ == "__main__":
    main()