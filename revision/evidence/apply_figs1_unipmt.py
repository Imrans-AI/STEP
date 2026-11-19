# -*- coding: utf-8 -*-
"""
Figure S1: swap in the re-rendered dataset-flow diagram (now four sources, with the
per-source contribution to the pool) and rewrite its caption, which still described
three databases and the submitted counts although the image had already been rebuilt.
Mechanics are those of apply_swap_figures.py: bytes replaced in place, drawing extent
rescaled so Word does not stretch the new aspect ratio.
"""
import io
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, 'revision/evidence')
from PIL import Image                                          # noqa: E402
from docx_tools import Docx                                    # noqa: E402

Image.MAX_IMAGE_PIXELS = None
BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
SI = BASE / 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
SRC = BASE / 'revision/evidence/figures/Figure_S1_dataset_flow_v2.png'
PART = 'word/media/image1.png'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WPNS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

d = Docx(str(SI))

OLD_CAP = (
    "Supplementary Figure S1 | Dataset construction and epitope-disjoint evaluation "
    "design. Provenance and processing pipeline for the STEP dataset. TCRβ–epitope "
    "pairs were compiled from three public databases (VDJdb, McPAS-TCR, IEDB), then "
    "merged, filtered (human; MHC class I; CDR3β 8–20 aa; epitopes 8–15 aa), "
    "and deduplicated to 99,323 unique positive pairs spanning 1,181 epitopes (93,654 "
    "unique CDR3β). Pairs were partitioned by an epitope-disjoint split — no "
    "epitope appears in both training and test, yielding 600 training and 581 held-out "
    "test epitopes."
)
NEW_CAP = (
    "Supplementary Figure S1 | Dataset construction and epitope-disjoint evaluation "
    "design. Provenance and processing pipeline for the STEP dataset. TCRβ–epitope "
    "pairs were compiled from four public sources — VDJdb, McPAS-TCR, IEDB and "
    "UniPMT, the last of which was not named in the submitted figure — then merged, "
    "filtered (human; MHC class I; CDR3β 8–20 aa; epitopes 8–15 aa), and "
    "deduplicated to 99,643 unique positive pairs spanning 1,281 epitopes (93,654 unique "
    "CDR3β), of which VDJdb supplies 82,617 pairs, UniPMT 8,256, McPAS-TCR 6,585 and "
    "IEDB 2,185. UniPMT contributes 522 of the 1,281 distinct epitopes, more than any "
    "other source, and supplied 210 of the 581 epitopes in the submitted test set; only "
    "its positive pairs were imported. Whitespace normalisation then merges four "
    "contaminated spellings, leaving 1,276 epitopes. Pairs were partitioned by an "
    "epitope-disjoint split — no epitope appears in more than one partition — "
    "yielding 600 training, 100 validation and 576 held-out test epitopes."
)
full = d.full_text()
assert OLD_CAP[:120] in full, 'caption head not found'
hits = d.find(OLD_CAP[:120])
assert len(hits) == 1, hits
d.replace_in_paragraph(hits[0], OLD_CAP, NEW_CAP, label='Fig S1 caption: four sources')

# --------------------------------------------------------------- swap the image
rels = d.blobs['word/_rels/document.xml.rels'].decode()
rid2t = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
rid = {f'word/{v}': k for k, v in rid2t.items()}[PART]

old_img = Image.open(io.BytesIO(d.blobs[PART]))
new_bytes = SRC.read_bytes()
new_img = Image.open(io.BytesIO(new_bytes))
d.blobs[PART] = new_bytes

done = 0
for blip in d.tree.iter(f'{{{A}}}blip'):
    if blip.get(f'{{{R}}}embed') != rid:
        continue
    anc = blip
    while anc is not None and not anc.tag.endswith('}inline') \
            and not anc.tag.endswith('}anchor'):
        anc = anc.getparent()
    assert anc is not None
    for ext in list(anc.iter(f'{{{WPNS}}}extent')) + list(anc.iter(f'{{{A}}}ext')):
        cx = ext.get('cx')
        if cx is None:
            continue
        ext.set('cy', str(int(int(cx) * new_img.height / new_img.width)))
        done += 1
assert done >= 1
print(f'image1.png {old_img.size} -> {new_img.size}, {done} extents rescaled')
d.save()

with zipfile.ZipFile(SI) as z:
    assert z.testzip() is None
    assert z.read(PART) == new_bytes
chk = Docx(str(SI))
t = chk.full_text()
assert 'UniPMT' in t and '99,643 unique positive pairs' in t and 'three public databases' not in t
print('Fig S1 caption and image verified')
