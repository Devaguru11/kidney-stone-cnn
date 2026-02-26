# api/report.py
# Clinical PDF Report Generator
# Called by POST /report — receives image + prediction result, returns PDF

import io
import json
import base64
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

router = APIRouter()


def build_pdf(image_bytes: bytes, result: dict) -> io.BytesIO:
    """Build the clinical PDF report and return as BytesIO buffer."""

    buffer = io.BytesIO()
    page_w, page_h = A4  # 595 x 842 points

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
    )

    # ── Colours ───────────────────────────────────────────────────
    dark_blue  = colors.HexColor('#1A3A5C')
    mid_blue   = colors.HexColor('#2D6EA8')
    light_blue = colors.HexColor('#D6EAF8')
    red        = colors.HexColor('#C0392B')
    light_red  = colors.HexColor('#FADBD8')
    green      = colors.HexColor('#1E8449')
    light_green= colors.HexColor('#D5F5E3')
    gray       = colors.HexColor('#F4F6F7')
    mid_gray   = colors.HexColor('#AAB7B8')
    white      = colors.white
    black      = colors.HexColor('#1C2833')

    # ── Styles ────────────────────────────────────────────────────
    normal = ParagraphStyle('normal',
        fontName='Helvetica', fontSize=10, textColor=black,
        leading=14, spaceAfter=4)

    bold = ParagraphStyle('bold',
        fontName='Helvetica-Bold', fontSize=10, textColor=black,
        leading=14, spaceAfter=4)

    center = ParagraphStyle('center',
        fontName='Helvetica', fontSize=10, textColor=black,
        alignment=TA_CENTER, leading=14)

    small = ParagraphStyle('small',
        fontName='Helvetica', fontSize=8, textColor=mid_gray,
        alignment=TA_CENTER, leading=11)

    small_italic = ParagraphStyle('small_italic',
        fontName='Helvetica-Oblique', fontSize=8, textColor=mid_gray,
        alignment=TA_CENTER, leading=12)

    # ── Report metadata ───────────────────────────────────────────
    report_id  = datetime.now().strftime('%Y%m%d_%H%M%S')
    now        = datetime.now()
    date_str   = now.strftime('%B %d, %Y')
    time_str   = now.strftime('%I:%M:%S %p')

    is_stone   = result.get('prediction') == 'stone'
    confidence = result.get('confidence', 0)
    prob_stone = result.get('probability_stone', 0)
    prob_no    = result.get('probability_no_stone', 0)
    model_ver  = result.get('model_version', 'efficientnet_b4_v1')
    threshold  = result.get('threshold_used', 0.5)
    gradcam_b64= result.get('gradcam_heatmap', None)

    content_width = page_w - 30 * mm  # usable width

    elements = []

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — Header Banner
    # ═══════════════════════════════════════════════════════════════
    header_data = [[
        Paragraph(
            '<font color="white" size="16"><b>🫘 NephroScan AI</b></font>'
            '<br/><font color="#A8D8EA" size="9">Kidney Stone Detection System — Clinical Report</font>',
            ParagraphStyle('hdr', fontName='Helvetica-Bold', fontSize=16,
                           textColor=white, alignment=TA_LEFT, leading=22)
        ),
        Paragraph(
            f'<font color="#A8D8EA" size="8">Report ID</font><br/>'
            f'<font color="white" size="10"><b>{report_id}</b></font>',
            ParagraphStyle('hdr_r', fontName='Helvetica', fontSize=10,
                           textColor=white, alignment=TA_RIGHT, leading=16)
        )
    ]]

    header_table = Table(header_data, colWidths=[content_width * 0.65, content_width * 0.35])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), dark_blue),
        ('TEXTCOLOR',  (0, 0), (-1, -1), white),
        ('TOPPADDING',    (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING',   (0, 0), (0, -1), 14),
        ('RIGHTPADDING',  (-1, 0), (-1, -1), 14),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 6 * mm))

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — Report Info Table
    # ═══════════════════════════════════════════════════════════════
    info_label = ParagraphStyle('info_lbl', fontName='Helvetica-Bold',
                                fontSize=9, textColor=mid_blue)
    info_val   = ParagraphStyle('info_val', fontName='Helvetica',
                                fontSize=9, textColor=black)

    info_data = [
        [
            Paragraph('Date',          info_label), Paragraph(date_str,   info_val),
            Paragraph('Time',          info_label), Paragraph(time_str,   info_val),
        ],
        [
            Paragraph('Model Version', info_label), Paragraph(model_ver,  info_val),
            Paragraph('Threshold',     info_label), Paragraph(str(threshold), info_val),
        ],
    ]

    col = content_width / 4
    info_table = Table(info_data, colWidths=[col * 0.7, col * 1.3, col * 0.7, col * 1.3])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), gray),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [gray, light_blue]),
        ('GRID',       (0, 0), (-1, -1), 0.5, mid_gray),
        ('TOPPADDING',    (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LEFTPADDING',   (0, 0), (-1, -1), 10),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 10),
        ('VALIGN',     (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 6 * mm))

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — Verdict
    # ═══════════════════════════════════════════════════════════════
    verdict_text  = 'STONE DETECTED'      if is_stone else 'NO STONE DETECTED'
    verdict_icon  = '⚠️'                  if is_stone else '✅'
    verdict_color = red                   if is_stone else green
    verdict_bg    = light_red             if is_stone else light_green
    verdict_border= red                   if is_stone else green

    verdict_style = ParagraphStyle('verdict',
        fontName='Helvetica-Bold', fontSize=22,
        textColor=verdict_color, alignment=TA_CENTER, leading=28)

    conf_style = ParagraphStyle('conf',
        fontName='Helvetica-Bold', fontSize=32,
        textColor=verdict_color, alignment=TA_CENTER, leading=38)

    verdict_data = [[
        Paragraph(f'{verdict_icon}  {verdict_text}', verdict_style),
        Paragraph(f'{confidence * 100:.2f}%', conf_style),
    ]]

    verdict_table = Table(verdict_data,
                          colWidths=[content_width * 0.6, content_width * 0.4])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), verdict_bg),
        ('LINEABOVE',     (0, 0), (-1, 0),  2, verdict_border),
        ('LINEBELOW',     (0, -1), (-1, -1), 2, verdict_border),
        ('LINEBEFORE',    (0, 0), (0, -1),  2, verdict_border),
        ('LINEAFTER',     (-1, 0), (-1, -1), 2, verdict_border),
        ('TOPPADDING',    (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
        ('LEFTPADDING',   (0, 0), (-1, -1), 16),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 16),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(verdict_table)
    elements.append(Spacer(1, 6 * mm))

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4 — Confidence Scores
    # ═══════════════════════════════════════════════════════════════
    conf_lbl = ParagraphStyle('conf_lbl', fontName='Helvetica-Bold',
                              fontSize=9, textColor=mid_blue, alignment=TA_CENTER)
    conf_val_red = ParagraphStyle('conf_val_r', fontName='Helvetica-Bold',
                                  fontSize=18, textColor=red, alignment=TA_CENTER)
    conf_val_grn = ParagraphStyle('conf_val_g', fontName='Helvetica-Bold',
                                  fontSize=18, textColor=green, alignment=TA_CENTER)

    scores_data = [[
        Paragraph('STONE PROBABILITY',    conf_lbl),
        Paragraph('NO STONE PROBABILITY', conf_lbl),
    ],[
        Paragraph(f'{prob_stone * 100:.2f}%', conf_val_red),
        Paragraph(f'{prob_no    * 100:.2f}%', conf_val_grn),
    ]]

    half = content_width / 2
    scores_table = Table(scores_data, colWidths=[half, half])
    scores_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (0, -1), light_red),
        ('BACKGROUND',    (1, 0), (1, -1), light_green),
        ('GRID',          (0, 0), (-1, -1), 0.5, mid_gray),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    elements.append(scores_table)
    elements.append(Spacer(1, 6 * mm))

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5 — Scan Images
    # ═══════════════════════════════════════════════════════════════
    img_label_style = ParagraphStyle('img_lbl', fontName='Helvetica-Bold',
                                     fontSize=9, textColor=mid_blue,
                                     alignment=TA_CENTER)

    img_size = 75 * mm  # image display size

    # Original scan
    try:
        scan_buf = io.BytesIO(image_bytes)
        scan_img = Image(scan_buf, width=img_size, height=img_size)
    except Exception:
        scan_img = Paragraph('[Image unavailable]', center)

    # Grad-CAM heatmap
    if gradcam_b64:
        try:
            heatmap_bytes = base64.b64decode(gradcam_b64)
            heatmap_buf   = io.BytesIO(heatmap_bytes)
            heatmap_img   = Image(heatmap_buf, width=img_size, height=img_size)
        except Exception:
            heatmap_img = Paragraph('[Heatmap decode error]', center)
    else:
        heatmap_img = Paragraph(
            '<font color="#AAB7B8">Grad-CAM heatmap<br/>not available</font>',
            ParagraphStyle('na', fontName='Helvetica', fontSize=9,
                           textColor=mid_gray, alignment=TA_CENTER, leading=13)
        )

    images_data = [
        [Paragraph('Original Scan', img_label_style),
         Paragraph('AI Focus Area (Grad-CAM)', img_label_style)],
        [scan_img, heatmap_img],
        [Paragraph('<font color="#AAB7B8" size="8">Uploaded kidney scan image</font>',
                   ParagraphStyle('cap', fontName='Helvetica-Oblique', fontSize=8,
                                  textColor=mid_gray, alignment=TA_CENTER)),
         Paragraph('<font color="#AAB7B8" size="8">Red regions = AI focus areas</font>',
                   ParagraphStyle('cap2', fontName='Helvetica-Oblique', fontSize=8,
                                  textColor=mid_gray, alignment=TA_CENTER))],
    ]

    images_table = Table(images_data, colWidths=[half, half])
    images_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, 0),  light_blue),
        ('BACKGROUND',    (0, 1), (-1, 1),  gray),
        ('BACKGROUND',    (0, 2), (-1, 2),  gray),
        ('GRID',          (0, 0), (-1, -1), 0.5, mid_gray),
        ('ALIGN',         (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',        (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING',    (0, 0), (-1, 0),  8),
        ('BOTTOMPADDING', (0, 0), (-1, 0),  8),
        ('TOPPADDING',    (0, 1), (-1, 1),  10),
        ('BOTTOMPADDING', (0, 1), (-1, 1),  10),
        ('TOPPADDING',    (0, 2), (-1, 2),  6),
        ('BOTTOMPADDING', (0, 2), (-1, 2),  6),
    ]))
    elements.append(images_table)
    elements.append(Spacer(1, 8 * mm))

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6 — Footer / Disclaimer
    # ═══════════════════════════════════════════════════════════════
    disclaimer_data = [[
        Paragraph(
            'This report is generated by NephroScan AI for informational purposes only.<br/>'
            'It is not a substitute for professional medical diagnosis.<br/>'
            'Always consult a qualified radiologist or urologist for clinical decisions.<br/>'
            f'<br/><font size="7">Generated: {date_str} at {time_str} &nbsp;·&nbsp; '
            f'Report ID: {report_id} &nbsp;·&nbsp; NephroScan AI v1.0</font>',
            small_italic
        )
    ]]

    footer_table = Table(disclaimer_data, colWidths=[content_width])
    footer_table.setStyle(TableStyle([
        ('BACKGROUND',    (0, 0), (-1, -1), gray),
        ('LINEABOVE',     (0, 0), (-1, 0),  1, mid_blue),
        ('TOPPADDING',    (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING',   (0, 0), (-1, -1), 12),
        ('RIGHTPADDING',  (0, 0), (-1, -1), 12),
    ]))
    elements.append(footer_table)

    # ── Build PDF ─────────────────────────────────────────────────
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ── FastAPI Endpoint ──────────────────────────────────────────────

@router.post('/report', tags=['Report'])
async def generate_report(
    file:   UploadFile = File(..., description='Original kidney scan image'),
    result: str        = Form(..., description='Prediction result JSON string'),
):
    """
    Generate a clinical PDF report for a kidney stone prediction.

    - **file**: The same image that was sent to /predict
    - **result**: The JSON response from /predict as a string
    """
    try:
        result_dict = json.loads(result)
    except json.JSONDecodeError:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail='Invalid result JSON')

    try:
        image_bytes = await file.read()
        pdf_buffer  = build_pdf(image_bytes, result_dict)
        report_id   = datetime.now().strftime('%Y%m%d_%H%M%S')

        return StreamingResponse(
            pdf_buffer,
            media_type='application/pdf',
            headers={
                'Content-Disposition': f'attachment; filename=nephroscan_{report_id}.pdf'
            }
        )
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f'PDF generation error: {str(e)}')