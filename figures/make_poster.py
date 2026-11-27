"""Showcase poster: fills the course template in place with our text, figures and a pipeline diagram.
Usage: python figures/make_poster.py  (run make_poster_figures.py first) -> Docs/Poster/Quality-Aware Slit-Lamp Grading - Poster.pptx
"""
import copy
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSTER = os.path.join(ROOT, "Docs", "Poster")
TEMPLATE = os.path.join(POSTER, "Topics_in_AI_for_Healthcare_Project_showcase_Nov20th_Event_Poster_Template.pptx")
OUT = os.path.join(POSTER, "Quality-Aware Slit-Lamp Grading - Poster.pptx")
FIG = os.path.join(ROOT, "figures")
INK, MUTED = RGBColor(0x1F, 0x23, 0x28), RGBColor(0x59, 0x63, 0x6E)
BLUE, ORANGE = RGBColor(0x2F, 0x6F, 0xB3), RGBColor(0xD9, 0x82, 0x2B)


def set_text(shape, paras, size=None):
    """Replace the text, keeping the template's first-paragraph format. paras: list of str or (str, bold)."""
    tf = shape.text_frame
    tf.word_wrap = True
    first = tf.paragraphs[0]._p
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    proto = copy.deepcopy(first)
    for r in proto.findall(qn("a:r"))[1:] + proto.findall(qn("a:br")):
        proto.remove(r)
    first.getparent().remove(first)
    for item in paras:
        text, bold = (item, None) if isinstance(item, str) else item
        p = copy.deepcopy(proto)
        tf._txBody.append(p)
        runs = p.findall(qn("a:r"))
        if not runs:
            continue
        runs[0].find(qn("a:t")).text = text
    for p, item in zip(tf.paragraphs, paras):
        bold = None if isinstance(item, str) else item[1]
        for r in p.runs:
            if size:
                r.font.size = Pt(size)
            if bold is not None:
                r.font.bold = bold
        p.space_after = Pt(8)


def box(slide, x, y, w, h, text, fill, size=22, color=INK, bold=False):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    s.adjustments[0] = 0.12
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = RGBColor(0xC9, 0xD1, 0xD9)
    s.line.width = Pt(1.5)
    tf = s.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size if i == 0 else size - 4)
        r.font.bold = bold or i == 0
        r.font.color.rgb = color if i == 0 else MUTED
    return s


def arrow(slide, x1, y1, x2, y2):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    c.line.color.rgb = MUTED
    c.line.width = Pt(3)
    ln = c.line._get_or_add_ln()
    tail = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
    ln.append(tail)


def picture(slide, path, x, y, w=None, h=None):
    return slide.shapes.add_picture(path, Cm(x), Cm(y), Cm(w) if w else None, Cm(h) if h else None)


def main():
    prs = Presentation(TEMPLATE)
    s = prs.slides[0]
    sh = {x.name: x for x in s.shapes}   # template names are unique except "Rounded Rectangle 84" (unused here)

    set_text(sh["TextBox 7"], ["Does knowing image quality help grade slit-lamp photos?"])
    set_text(sh["TextBox 4"], ["We built a model that scores 9 photo defects and tested it two ways: as an extra input "
                               "to a disease grader, and as a gatekeeper that decides which photos get graded."])
    set_text(sh["TextBox 2"], [("Team:  Abhay Sharma (2025201014)  ·  Shaik Areesh (2026909008)  ·  IIIT Hyderabad", True)])

    # the problem
    set_text(sh["TextBox 17"], [
        "Slit-lamp photos from busy clinics and screening camps are often blurred, dark, overexposed or blocked by the eyelid.",
        "AI graders for cataract and corneal disease are trained on good photos and ignore quality.",
        "No public slit-lamp dataset has quality labels.",
    ], size=24)
    picture(s, os.path.join(FIG, "update3", "fig_real_problem.png"), 2.0, 21.2, w=21.8)
    t = s.shapes.add_textbox(Cm(2.0), Cm(26.1), Cm(21.8), Cm(0.8))
    t.text_frame.text = "Real photos from the public datasets we used"
    t.text_frame.paragraphs[0].runs[0].font.size = Pt(16)
    t.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED

    # our approach
    set_text(sh["TextBox 32"], [
        ("The challenges", True),
        "No quality labels exist. Labelling thousands of photos ourselves would not be clinically sound, "
        "and one good / bad cut-off would be arbitrary.",
        "So we predict how strong each of 9 defects is (0 to 1), learned from clean photos with defects "
        "added by code at known strength.",
        ("Data", True),
        "8,263 clean photos from 5 public datasets → 33,052 training images (each clean photo + 3 degraded copies).",
        "Grading tasks: nuclear cataract grade (1,190 photos, 112 people) and corneal ulcer type, SUSTech-SYSU (708 photos).",
        ("Hypothesis", True),
        "A grader that is told the photo's quality grades degraded photos better.",
        ("How we tested it", True),
        "Graders trained on  A: clean photos,  B: + degraded copies,  C: + degraded copies + quality scores. "
        "Control: C with the quality scores shuffled. 3 seeds; 95 % bootstrap CIs over eyes; metric QWK.",
    ], size=24)

    # our solution: text + pipeline diagram in place of the template's visual placeholder
    set_text(sh["TextBox 24"], [
        "A quality module scores every photo for 9 defects before any diagnosis is made. We used the scores in two ways: "
        "as an extra input to the grader (our hypothesis), and as a gatekeeper that only lets good photos through.",
    ], size=26)
    s.shapes._spTree.remove(sh["Flowchart: Process 51"]._element)
    y0 = 19.0
    picture(s, os.path.join(FIG, "update3", "fig_real_problem.png"), 26.9, y0 + 2.6, h=4.6).crop_right = 0
    pic = s.shapes[-1]
    pic.crop_left, pic.crop_right = 0.5, 0.25   # one photo from the strip
    pic.width = Cm(6.0)
    box(s, 34.6, y0 + 2.4, 9.4, 5.0, "Quality module\nSwin-T · 9 defect scores", RGBColor(0xFB, 0xEB, 0xD9))
    box(s, 47.4, y0 + 0.2, 15.2, 4.4, "① Extra input\ngrader sees photo + scores → hypothesis", RGBColor(0xE6, 0xEE, 0xF7))
    box(s, 47.4, y0 + 5.4, 15.2, 5.6, "② Gatekeeper\ngood photo → graded by VisionFM\npoor photo → retake, with the defect named",
        RGBColor(0xE6, 0xEE, 0xF7))
    arrow(s, 33.0, y0 + 4.9, 34.5, y0 + 4.9)
    arrow(s, 44.1, y0 + 4.2, 47.3, y0 + 2.4)
    arrow(s, 44.1, y0 + 5.6, 47.3, y0 + 8.2)

    # how it works
    set_text(sh["TextBox 10"], [
        ("Quality module", True),
        "Swin-T trained on the 33,052 images. Defect detection (AUROC) and strength ranking (Spearman):",
    ], size=20)
    rows = [("Test defects", "AUROC", "Spearman"), ("same code as training", "0.998", "0.976"),
            ("held-out code", "0.995", "0.908"), ("fresh code, written after training", "0.953", "0.767")]
    tb = s.shapes.add_table(4, 3, Cm(26.3), Cm(41.6), Cm(16.9), Cm(3.6)).table
    tb.columns[0].width, tb.columns[1].width, tb.columns[2].width = Cm(9.7), Cm(3.4), Cm(3.8)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tb.cell(i, j)
            c.text = v
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xEE, 0xF1, 0xF4) if i == 0 else RGBColor(0xFF, 0xFF, 0xFF)
            f = c.text_frame.paragraphs[0].runs[0].font
            f.size, f.bold, f.color.rgb = Pt(17), i == 0, INK
    picture(s, os.path.join(FIG, "update3", "fig_defects.png"), 26.3, 45.8, w=8.6)
    t = s.shapes.add_textbox(Cm(35.2), Cm(45.6), Cm(8.2), Cm(9))
    t.text_frame.word_wrap = True
    for i, (head, body) in enumerate([("Grader", "VisionFM slit-lamp ViT-B/16, fine-tuned at 384 × 512."),
                                      ("Defects", "defocus, motion blur, over- / underexposure, uneven light, "
                                                  "haze, glare, occlusion, noise.")]):
        p = t.text_frame.paragraphs[0] if i == 0 else t.text_frame.add_paragraph()
        r = p.add_run()
        r.text, r.font.bold, r.font.size, r.font.color.rgb = head, True, Pt(20), INK
        p = t.text_frame.add_paragraph()
        r = p.add_run()
        r.text, r.font.size, r.font.color.rgb = body, Pt(18), INK
        p.space_after = Pt(10)

    # what we learned: hypothesis failed -> gatekeeper
    set_text(sh["TextBox 86"], [
        ("1. The hypothesis did not hold", True),
        "Quality scores added nothing: C ≈ B ≈ shuffled.",
    ], size=19)
    sh["TextBox 86"].height = Cm(2.2)
    picture(s, os.path.join(FIG, "poster", "fig_hypothesis.png"), 46.2, 39.4, h=6.6)
    t = s.shapes.add_textbox(Cm(45.6), Cm(46.0), Cm(16.9), Cm(2.2))
    t.text_frame.word_wrap = True
    for i, (txt, b) in enumerate([("2. So we used quality as a gatekeeper", True),
                                  ("Grade only the best-quality photos, refer the rest:", False)]):
        p = t.text_frame.paragraphs[0] if i == 0 else t.text_frame.add_paragraph()
        r = p.add_run()
        r.text, r.font.bold, r.font.size, r.font.color.rgb = txt, b, Pt(19), INK
    picture(s, os.path.join(FIG, "poster", "fig_gating.png"), 45.4, 48.2, h=7.5)

    # impact and what's next (right column, also carries the gating takeaway)
    set_text(sh["TextBox 59"], [
        ("What the gatekeeper gives", True),
        "Keeping the best 50 % raises QWK by +0.03 to +0.08 over grading everything (CIs above 0; "
        "not significant on fresh defect code).",
        "The grader's own confidence is as good or better, but the quality module works before grading, "
        "does not depend on the grader, and says why a photo was rejected.",
        ("Use", True),
        "Check each photo on the spot in screening camps and ask for a retake while the patient is still there.",
        ("Next", True),
        "Validate on real photos with expert quality labels; combine quality with grader confidence.",
    ], size=21)
    sh["TextBox 59"].height = Cm(20.5)
    set_text(sh["TextBox 43"], ["IIIT Hyderabad; Topics in AI for Healthcare instructors; Ada HPC cluster, IIIT Hyderabad."],
             size=17)
    set_text(sh["TextBox 44"], [
        "Datasets: SLID, SLID-E, Nuclear Cataract, SUSTech-SYSU, Infectious Keratitis (public).",
        "VisionFM: Qiu et al., NEJM AI 2024.  Swin Transformer: Liu et al., ICCV 2021.",
    ], size=15)
    prs.save(OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
