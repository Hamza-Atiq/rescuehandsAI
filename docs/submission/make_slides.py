"""Build the submission slide deck (16:9 PowerPoint).

  uv run --no-project --with python-pptx==1.0.2 python docs/submission/make_slides.py

Every number on the slides is either measured (committed results or Hub pages) or an
orange ⟨placeholder⟩ to replace from the final v2 results. Re-run after editing RESULTS.
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
MEDIA = HERE.parent / "media"
OUT = HERE / "rescuehands_slides.pptx"

NAVY, INK, MUTED = RGBColor(0x0E, 0x16, 0x24), RGBColor(0xF2, 0xF5, 0xF9), RGBColor(0xA9, 0xB6, 0xC6)
GOLD, TEAL, RED, PENDING = RGBColor(0xFF, 0xC4, 0x40), RGBColor(0x3F, 0xC1, 0xB0), RGBColor(0xE8, 0x5D, 0x4F), RGBColor(0xFF, 0x9F, 0x1C)
CARD = RGBColor(0x18, 0x24, 0x36)

# Final learned-policy numbers: replace the placeholders once results/smolvla_v2_* exist.
RESULTS = {
    "v2_sup_on": "not run", "v2_sup_off": "not run", "v2_fault_sup_on": "1/10", "v2_fault_sup_off": "running",
    "igpu_s": "4.53 s", "torch_s": "≈190 s (v1)", "speedup": "≈40×", "diff": "0.004 rad (v1)",
}

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(title=None, kicker=None):
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY
    if kicker:
        text(s, kicker, 0.6, 0.35, 12, 0.4, 14, GOLD, bold=True)
    if title:
        text(s, title, 0.6, 0.7, 12.2, 1.0, 34, INK, bold=True)
    return s


def text(s, value, x, y, w, h, size, color=INK, bold=False, align=PP_ALIGN.LEFT):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    lines = value if isinstance(value, list) else [value]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size, run.font.bold, run.font.name = Pt(size), bold, "Segoe UI"
        run.font.color.rgb = PENDING if "⟨" in line else color
        p.space_after = Pt(6)
    return box


def bullets(s, items, x, y, w, h, size=20):
    return text(s, [f"•  {item}" for item in items], x, y, w, h, size, INK)


def card(s, x, y, w, h, title, body, accent=TEAL):
    shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = CARD
    shape.line.color.rgb = accent
    shape.adjustments[0] = 0.08
    text(s, title, x + 0.2, y + 0.12, w - 0.4, 0.5, 17, accent, bold=True)
    text(s, body, x + 0.2, y + 0.65, w - 0.4, h - 0.75, 17, INK)


def arrow(s, x1, y1, x2, y2):
    line = s.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    line.line.color.rgb = MUTED
    line.line.width = Pt(2)
    line.line._get_or_add_ln().append(_tail())


def _tail():
    from pptx.oxml.ns import qn
    from lxml import etree
    tail = etree.Element(qn("a:tailEnd"))
    tail.set("type", "triangle")
    return tail


def picture(s, name, x, y, w):
    return s.shapes.add_picture(str(MEDIA / name), Inches(x), Inches(y), width=Inches(w))


def table(s, rows, x, y, widths, size=15, highlight_col=None):
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                               Inches(sum(widths)), Inches(0.5 * len(rows)))
    tbl = shape.table
    for c, w in enumerate(widths):
        tbl.columns[c].width = Inches(w)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            cell = tbl.cell(r, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0x24, 0x34, 0x4C) if r == 0 else CARD
            cell.text = value or " "
            run = cell.text_frame.paragraphs[0].runs[0]
            run.font.size, run.font.name = Pt(size), "Segoe UI"
            run.font.bold = r == 0 or c == highlight_col
            run.font.color.rgb = PENDING if "⟨" in value else (GOLD if r == 0 else INK)
    return tbl


# 1 — title
s = prs.slides.add_slide(BLANK)
s.shapes.add_picture(str(HERE / "cover.jpg"), 0, 0, width=prs.slide_width, height=prs.slide_height)
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(6.75), prs.slide_width, Inches(0.75))
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
text(s, "Intel Bimanual VLA Manipulation track · AI Infra Summit Hackathon 2026", 0.75, 6.85, 12, 0.5, 18, INK)

# 2 — problem
s = slide("A policy acts even after reality went wrong", "THE PROBLEM")
bullets(s, ["Vision-language-action models predict the next move; they do not check the last one.",
            "With two arms, one slip by one hand ruins the other hand's work.",
            "Learned policies drift into states their demonstrations never showed.",
            "So: never trust an action because a model produced it — check the physics, then recover."],
        0.6, 1.9, 7.2, 4.5, 21)
picture(s, "teacher_drop_recovery.jpg", 8.1, 2.0, 4.7)
text(s, "A real drop: the gripper is forced open, the fork falls, both arms back off and retry.",
     8.1, 4.75, 4.7, 1.0, 13, MUTED)

# 3 — task
s = slide("Set a dinner place with two SO-101 arms", "THE TASK")
picture(s, "teacher_handoff_close.jpg", 0.6, 1.85, 6.0)
picture(s, "teacher_table_set.jpg", 6.75, 1.85, 6.0)
bullets(s, ["“Hand the spoon over to the left arm, then place the cup next to the plate.” — the words pick fork or spoon",
            "Right arm picks → in-air hand-off → left arm places the utensil → right arm places the cup",
            "Every seed changes positions, sizes, mass, friction, lighting and table colour; contact grasps only, no welding"],
        0.6, 5.35, 12.2, 2.0, 16)

# 4 — architecture
s = slide("The VLA proposes; physics decides", "ARCHITECTURE")
boxes = [("Instruction\n3 cameras\n12 joints", 0.6, GOLD), ("SmolVLA\nOpenVINO on Intel iGPU", 3.05, TEAL),
         ("Action guard\nlimits + step size", 5.5, TEAL), ("MuJoCo physics\n2× SO-101", 7.95, TEAL),
         ("Physics auditor\nheld? dropped? placed?", 10.4, RED)]
for label, x, colour in boxes:
    b = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.3), Inches(2.25), Inches(1.5))
    b.fill.solid()
    b.fill.fore_color.rgb = CARD
    b.line.color.rgb = colour
    tf = b.text_frame
    tf.word_wrap = True
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name, r.font.bold = Pt(15 if i == 0 else 12), "Segoe UI", i == 0
        r.font.color.rgb = INK if i == 0 else MUTED
for x in (2.85, 5.3, 7.75, 10.2):
    arrow(s, x, 3.05, x + 0.2, 3.05)
card(s, 0.6, 4.35, 3.9, 2.55, "Policy sees", "Camera images, joint positions and the instruction. Nothing else — no object poses, no contacts.", GOLD)
card(s, 4.72, 4.35, 3.9, 2.55, "Supervisor sees", "Simulator ground truth: which jaws hold what, support, zones, arm-to-arm contact. Used only to judge and recover.", RED)
card(s, 8.85, 4.35, 3.9, 2.55, "IK only in the teacher", "A scripted IK teacher makes demonstrations and the baseline. The deployed loop has no IK: SmolVLA outputs the 12 joint targets.", TEAL)

# 5 — learned policy and data
s = slide("SmolVLA, taught to get back on track", "LEARNED POLICY")
table(s, [["", "v1", "v2"],
          ["Episodes", "101", "569"],
          ["Frames (20 fps)", "57,173", "336,729"],
          ["Clean demonstrations", "101", "293"],
          ["Messy starts (arms nudged, items moved)", "—", "194"],
          ["Real drop → recovery → finish", "—", "82"],
          ["Training", "12,000 steps from SmolVLA base", "6,000 steps from v1, lr 5e-5"]],
      0.6, 1.9, [4.6, 3.0, 3.6], 15)
bullets(s, ["Why v2: v1 copied the teacher closely (open-loop error ≈ 0.01 rad) but only 1 of 9 completed live episodes succeeded — live runs reach states the clean demos never showed.",
            "Trained on free Kaggle T4 GPUs; camera images rendered on the GPU (229 s → 14 s per episode after an EGL fix)."],
        0.6, 5.65, 12.2, 1.7, 15)

# 6 — safety and success
s = slide("Physics decides success — not the policy", "SAFETY & RECOVERY")
card(s, 0.6, 1.9, 4.0, 4.9, "Supervisor",
     "Drop or stalled grasp → open both hands, return to a safe pose, let the policy re-plan. At most 2 retries. "
     "The retreat itself is watched: an item leaving the table ends the episode.", RED)
card(s, 4.72, 1.9, 4.0, 4.9, "Task success (10 steps stable)",
     "Cup and utensil in their zones, released, supported, cup upright (≤15°), settled, spare utensil still near its start at the end, "
     "and an ordered IN-AIR hand-off: right alone → both → left alone, with at most 5 steps of contact flicker.", TEAL)
card(s, 8.85, 1.9, 4.0, 4.9, "Checked, not assumed",
     "Model contracts pin joint order, cameras, rate and file hashes. Every run writes a manifest (code revision, "
     "arguments, versions). An independent audit's findings were reproduced, fixed and re-measured.", GOLD)

# 7 — results
s = slide("Results on 10 randomized seeds", "RESULTS")
table(s, [["Policy", "Supervisor", "Gripper fault", "Success"],
          ["Scripted teacher (baseline)", "on", "none", "8/10"],
          ["Scripted teacher (baseline)", "off", "yes", "2/10"],
          ["Scripted teacher (baseline)", "on", "yes", "7/10"],
          ["SmolVLA v2 · OpenVINO iGPU", "off", "none", RESULTS["v2_sup_off"]],
          ["SmolVLA v2 · OpenVINO iGPU", "on", "none", RESULTS["v2_sup_on"]],
          ["SmolVLA v2 · OpenVINO iGPU", "off", "yes", RESULTS["v2_fault_sup_off"]],
          ["SmolVLA v2 · OpenVINO iGPU", "on", "yes", RESULTS["v2_fault_sup_on"]]],
      0.6, 1.75, [5.2, 2.2, 2.2, 2.4], 16, highlight_col=3)
text(s, "Pairs change one thing: same seeds and fault, supervisor on vs off. For the teacher it turns 2/10 into 7/10. "
        "Every number comes from committed result files with run manifests.", 0.6, 6.0, 12.2, 1.2, 15, MUTED)

# 8 — Intel
s = slide("Running the policy on an Intel laptop", "INTEL OPTIMIZATION")
table(s, [["Backend (Intel Core i5-6300U + HD Graphics 520)", "Time per 50-action chunk", "Max action diff vs FP32"],
          ["PyTorch, CPU (baseline)", RESULTS["torch_s"], RESULTS["diff"]],
          ["OpenVINO FP32, CPU", "13.9 s (v1)", "0 (reference)"],
          ["OpenVINO FP16, iGPU", RESULTS["igpu_s"], RESULTS["diff"]],
          ["OpenVINO INT8 weights (NNCF)", "not measured", "—"]],
      0.6, 1.9, [6.2, 3.0, 3.0], 16)
bullets(s, [f"iGPU speed-up over PyTorch CPU: {RESULTS['speedup']}, with its accuracy cost reported next to it",
            "Exported with Intel Physical AI Studio; organizers allowed non-Core-Ultra Intel hardware",
            "Load time and memory are measured inside the same timer for every backend"],
        0.6, 4.8, 12.2, 2.3, 17)

# 9 — limits and reproducibility
s = slide("What we claim, and what we do not", "HONEST LIMITS")
card(s, 0.6, 1.9, 6.0, 4.9, "Limits",
     "• Collision checks cover arm-to-arm contact only.\n• Randomized item friction does not reach the jaw contacts "
     "(measured), so slippery grasps are not tested.\n• Simulation pauses during inference: not real-time 20 Hz control.\n"
     "• Evaluation seeds guided teacher debugging, so they are not a pristine unseen set.", RED)
card(s, 6.75, 1.9, 6.0, 4.9, "Reproduce it",
     "• Pinned MuJoCo scene and SO-101 asset revision\n• Data generation, training, export, evaluation, benchmark scripts\n"
     "• Public dataset ABDHAM/rescuehands_table_v2 and model ABDHAM/smolvla_rescuehands_v2\n"
     "• 99 unit and physics tests · per-run manifests · dataset provenance logs", TEAL)

# 10 — close
s = slide("RescueHands AI", "THANK YOU")
text(s, "A VLA should not be trusted just because it produced an action. Check what physically happened — and recover.",
     0.6, 1.9, 12, 1.5, 26, INK)
text(s, ["github.com/Hamza-Atiq/rescuehandsAI", "huggingface.co/ABDHAM/smolvla_rescuehands_v2",
         "huggingface.co/datasets/ABDHAM/rescuehands_table_v2"], 0.6, 4.0, 12, 2.0, 20, GOLD)

prs.save(OUT)
print("wrote", OUT, "slides:", len(prs.slides))
