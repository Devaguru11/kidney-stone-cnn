# api/unified_report.py
# Generates a clinical PDF report for all 3 model results

import io, base64, datetime
from pathlib import Path

def generate_report(result: dict, image_bytes: bytes, filename: str) -> bytes:
    """Generate a clinical HTML→PDF report. Returns PDF bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        return _generate_pdf_reportlab(result, image_bytes, filename)
    except ImportError:
        return _generate_html_report(result, image_bytes, filename)


def _generate_pdf_reportlab(result, image_bytes, filename):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # Colors
    NAVY   = colors.HexColor('#0D1B2A')
    BLUE   = colors.HexColor('#2D9CDB')
    TEAL   = colors.HexColor('#00D4AA')
    RED    = colors.HexColor('#FF4B6E')
    ORANGE = colors.HexColor('#FF8C42')
    GRAY   = colors.HexColor('#5A7A99')

    RISK_COLORS = {
        'HIGH':   RED,
        'MEDIUM': ORANGE,
        'LOW':    TEAL,
        'NORMAL': TEAL,
    }

    # Custom styles
    title_style = ParagraphStyle('title', fontSize=22, textColor=colors.white,
                                  fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=4)
    sub_style   = ParagraphStyle('sub',   fontSize=10, textColor=BLUE,
                                  fontName='Helvetica', alignment=TA_CENTER, spaceAfter=2)
    h2_style    = ParagraphStyle('h2',    fontSize=13, textColor=NAVY,
                                  fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle('body',  fontSize=9,  textColor=GRAY,
                                  fontName='Helvetica', spaceAfter=4, leading=14)
    mono_style  = ParagraphStyle('mono',  fontSize=8,  textColor=GRAY,
                                  fontName='Courier', spaceAfter=2)

    now = datetime.datetime.now()

    # ── Header banner ──
    header_data = [[
        Paragraph('🔬 NephroScan AI', title_style),
        Paragraph('Unified Clinical Report', sub_style),
    ]]
    header_table = Table([[
        Paragraph('<font color="white"><b>NephroScan AI</b></font>', ParagraphStyle('ht', fontSize=20, textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_CENTER)),
    ]], colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), NAVY),
        ('TOPPADDING',    (0,0), (-1,-1), 18),
        ('BOTTOMPADDING', (0,0), (-1,-1), 18),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ROUNDEDCORNERS', [8]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph('Unified Kidney Analysis Report — 3 AI Models', ParagraphStyle('sub2', fontSize=10, textColor=BLUE, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*cm))

    # ── Report meta ──
    meta_data = [
        ['Report Date:', now.strftime('%B %d, %Y')],
        ['Report Time:', now.strftime('%H:%M:%S')],
        ['Image File:', filename],
        ['Models Used:', 'v1 Stone Detector · v2 4-Class · v3 Cancer Detector'],
    ]
    meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME',  (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME',  (1,0), (1,-1), 'Helvetica'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (0,-1), NAVY),
        ('TEXTCOLOR', (1,0), (1,-1), GRAY),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', thickness=1, color=BLUE, spaceAfter=10))

    # ── Risk Level ──
    risk      = result.get('risk_level', 'NORMAL')
    risk_color= RISK_COLORS.get(risk, TEAL)
    risk_messages = {
        'HIGH':   'High risk finding detected. Immediate clinical review and specialist referral recommended.',
        'MEDIUM': 'Moderate risk finding. Clinical follow-up and further investigation recommended.',
        'LOW':    'Low risk finding. Routine monitoring and follow-up advised.',
        'NORMAL': 'No significant finding detected. Routine follow-up recommended.',
    }
    risk_table = Table([[
        Paragraph(f'<b>RISK LEVEL: {risk}</b>', ParagraphStyle('risk', fontSize=14, textColor=colors.white, fontName='Helvetica-Bold', alignment=TA_CENTER)),
    ]], colWidths=[17*cm])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), risk_color),
        ('TOPPADDING',    (0,0), (-1,-1), 12),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ('ROUNDEDCORNERS',[6]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(risk_messages[risk], ParagraphStyle('riskmsg', fontSize=9, textColor=GRAY, fontName='Helvetica', alignment=TA_CENTER)))
    story.append(Spacer(1, 0.4*cm))
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E0E6ED'), spaceAfter=10))

    # ── 3 Model Results ──
    v1 = result.get('v1', {})
    v2 = result.get('v2', {})
    v3 = result.get('v3', {})

    story.append(Paragraph('Model Results', h2_style))

    model_rows = [
        ['Model', 'Prediction', 'Confidence', 'Details'],
        [
            'v1 · Stone Detector',
            'Stone Detected' if v1.get('has_stone') else 'No Stone',
            f"{v1.get('confidence',0)*100:.1f}%",
            f"P(stone)={v1.get('probabilities',{}).get('stone',0)*100:.1f}%  P(no_stone)={v1.get('probabilities',{}).get('no_stone',0)*100:.1f}%",
        ],
        [
            'v2 · 4-Class Classifier',
            v2.get('prediction','').upper(),
            f"{v2.get('confidence',0)*100:.1f}%",
            f"Normal={v2.get('probabilities',{}).get('normal',0)*100:.1f}%  Cyst={v2.get('probabilities',{}).get('cyst',0)*100:.1f}%  Stone={v2.get('probabilities',{}).get('stone',0)*100:.1f}%  Tumour={v2.get('probabilities',{}).get('tumour',0)*100:.1f}%",
        ],
        [
            'v3 · Cancer Detector',
            'CANCER SUSPECTED' if v3.get('is_cancer') else 'Not Cancer',
            f"{v3.get('confidence',0)*100:.1f}%",
            f"P(cancer)={v3.get('cancer_prob',0)*100:.1f}%  P(not_cancer)={v3.get('probabilities',{}).get('not_cancer',0)*100:.1f}%",
        ],
    ]

    col_widths = [3.5*cm, 3.5*cm, 2.5*cm, 7.5*cm]
    model_table = Table(model_rows, colWidths=col_widths)
    model_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), NAVY),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',     (0,1), (-1,-1), GRAY),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#E0E6ED')),
        ('TOPPADDING',    (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,-1), 7),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        # Color code v1 result
        ('TEXTCOLOR', (1,1), (1,1), RED if v1.get('has_stone') else TEAL),
        ('FONTNAME',  (1,1), (1,1), 'Helvetica-Bold'),
        # Color code v2 result
        ('TEXTCOLOR', (1,2), (1,2),
            RED if v2.get('prediction')=='tumour' else
            ORANGE if v2.get('prediction')=='stone' else
            BLUE if v2.get('prediction')=='cyst' else TEAL),
        ('FONTNAME',  (1,2), (1,2), 'Helvetica-Bold'),
        # Color code v3 result
        ('TEXTCOLOR', (1,3), (1,3), RED if v3.get('is_cancer') else TEAL),
        ('FONTNAME',  (1,3), (1,3), 'Helvetica-Bold'),
    ]))
    story.append(model_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Clinical Notes ──
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E0E6ED'), spaceAfter=10))
    story.append(Paragraph('Clinical Notes', h2_style))

    notes = [
        ('v2 Classifier', v2.get('clinical_note', '')),
        ('v3 Cancer Screen', v3.get('clinical_note', '')),
    ]
    for label, note in notes:
        if note:
            story.append(Paragraph(f'<b>{label}:</b> {note}', body_style))

    story.append(Spacer(1, 0.3*cm))

    # ── Model Performance ──
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E0E6ED'), spaceAfter=10))
    story.append(Paragraph('Model Performance Summary', h2_style))

    perf_rows = [
        ['Model',              'Architecture',      'Accuracy', 'AUC'],
        ['v1 Stone Detector',  'EfficientNet-B4',  '99.2%',    '0.999'],
        ['v2 4-Class',         'EfficientNet-B4',  '97.0%',    '0.998'],
        ['v3 Cancer Detector', 'EfficientNet-B4',  '99.6%',    '0.9999'],
    ]
    perf_table = Table(perf_rows, colWidths=[4.5*cm, 4.5*cm, 4*cm, 4*cm])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1E2D3D')),
        ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 8),
        ('FONTNAME',      (0,1), (-1,-1), 'Helvetica'),
        ('TEXTCOLOR',     (0,1), (-1,-1), GRAY),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('GRID',          (0,0), (-1,-1), 0.5, colors.HexColor('#E0E6ED')),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING',   (0,0), (-1,-1), 8),
        ('ALIGN',         (2,0), (-1,-1), 'CENTER'),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Disclaimer ──
    story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#E0E6ED'), spaceAfter=10))
    story.append(Paragraph(
        '<b>⚠️ Disclaimer:</b> This report is generated by an AI system for research and educational '
        'purposes only. It is not a substitute for professional medical diagnosis. All findings must '
        'be reviewed and confirmed by a qualified healthcare professional before any clinical decisions '
        'are made. NephroScan AI is not approved for clinical use.',
        ParagraphStyle('disclaimer', fontSize=8, textColor=GRAY, fontName='Helvetica',
                      borderColor=ORANGE, borderWidth=1, borderPadding=8,
                      backColor=colors.HexColor('#FFF8E1'))
    ))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f'Generated by NephroScan AI v3.0  ·  {now.strftime("%Y-%m-%d %H:%M:%S")}  ·  3-Model Unified Pipeline',
        ParagraphStyle('footer', fontSize=7, textColor=colors.HexColor('#3A5A79'),
                      fontName='Helvetica', alignment=TA_CENTER)
    ))

    doc.build(story)
    return buf.getvalue()


def _generate_html_report(result, image_bytes, filename):
    """Fallback: return HTML as bytes if reportlab not installed."""
    now  = datetime.datetime.now().strftime('%B %d, %Y %H:%M:%S')
    v1   = result.get('v1', {})
    v2   = result.get('v2', {})
    v3   = result.get('v3', {})
    risk = result.get('risk_level', 'NORMAL')
    risk_colors = {'HIGH':'#FF4B6E','MEDIUM':'#FF8C42','LOW':'#00D4AA','NORMAL':'#00D4AA'}
    rc = risk_colors.get(risk, '#00D4AA')

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"/>
    <title>NephroScan Report</title>
    <style>body{{font-family:Arial,sans-serif;max-width:800px;margin:40px auto;color:#333;}}
    h1{{background:#0D1B2A;color:white;padding:20px;border-radius:8px;text-align:center;}}
    .risk{{background:{rc};color:white;padding:12px;border-radius:6px;text-align:center;font-size:18px;font-weight:bold;margin:20px 0;}}
    table{{width:100%;border-collapse:collapse;margin:16px 0;}}
    th{{background:#1E2D3D;color:white;padding:10px;text-align:left;}}
    td{{padding:9px;border-bottom:1px solid #eee;}}
    tr:nth-child(even){{background:#f8f9fa;}}
    .disclaimer{{background:#FFF8E1;border:1px solid #FFE082;padding:12px;border-radius:6px;font-size:12px;color:#555;}}
    </style></head><body>
    <h1>🔬 NephroScan AI — Clinical Report</h1>
    <p style="text-align:center;color:#666">{now} · {filename}</p>
    <div class="risk">RISK LEVEL: {risk}</div>
    <h2>Model Results</h2>
    <table><tr><th>Model</th><th>Prediction</th><th>Confidence</th></tr>
    <tr><td>v1 Stone Detector</td><td>{'Stone Detected' if v1.get('has_stone') else 'No Stone'}</td><td>{v1.get('confidence',0)*100:.1f}%</td></tr>
    <tr><td>v2 4-Class</td><td>{v2.get('prediction','').upper()}</td><td>{v2.get('confidence',0)*100:.1f}%</td></tr>
    <tr><td>v3 Cancer</td><td>{'Cancer Suspected' if v3.get('is_cancer') else 'Not Cancer'}</td><td>{v3.get('confidence',0)*100:.1f}%</td></tr>
    </table>
    <h2>Clinical Notes</h2>
    <p>{v2.get('clinical_note','')}</p>
    <p>{v3.get('clinical_note','')}</p>
    <div class="disclaimer">⚠️ This report is for research purposes only and is not a substitute for professional medical diagnosis.</div>
    </body></html>"""
    return html.encode('utf-8')