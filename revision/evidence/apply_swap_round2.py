"""Swap the rebuilt Figure 5 and the two Fig S2 panels into the working documents."""
import io, re, sys, zipfile
from pathlib import Path
sys.path.insert(0, 'revision/evidence')
from PIL import Image
from docx_tools import Docx, W

Image.MAX_IMAGE_PIXELS = None
BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
FIG = BASE / 'revision/evidence/figures'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
WPNS = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'

JOBS = {
    'revision/working/Manuscript_fv_1_REVISED.docx': {
        'word/media/image5.png': FIG / 'Figure_5_rebuilt_full.png'},
    'revision/working/STEP_Supplementary_FINAL_REVISED.docx': {
        'word/media/image2.png': FIG / 'Figure_S2a_rebuilt_6RSY.png',
        'word/media/image3.png': FIG / 'Figure_S2b_rebuilt_1OGA.png'},
}
for doc, mapping in JOBS.items():
    dd = Docx(str(BASE / doc))
    rels = dd.blobs['word/_rels/document.xml.rels'].decode()
    rid2t = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    t2rid = {f'word/{v}': k for k, v in rid2t.items()}
    for part, src in mapping.items():
        old = Image.open(io.BytesIO(dd.blobs[part]))
        nb = open(src, 'rb').read()
        new = Image.open(io.BytesIO(nb))
        dd.blobs[part] = nb
        rid = t2rid[part]
        done = 0
        for blip in dd.tree.iter(f'{{{A}}}blip'):
            if blip.get(f'{{{R}}}embed') != rid:
                continue
            anc = blip
            while anc is not None and not anc.tag.endswith('}inline') \
                    and not anc.tag.endswith('}anchor'):
                anc = anc.getparent()
            for ext in list(anc.iter(f'{{{WPNS}}}extent')) + list(anc.iter(f'{{{A}}}ext')):
                cx = ext.get('cx')
                if cx:
                    ext.set('cy', str(int(int(cx) * new.height / new.width)))
                    done += 1
        assert done >= 1
        print(f'  {Path(doc).name}: {part} {old.size} -> {new.size}, {done} extents')
    dd.save()

for doc, mapping in JOBS.items():
    p = BASE / doc
    with zipfile.ZipFile(p) as z:
        assert z.testzip() is None
        for part, src in mapping.items():
            assert z.read(part) == open(src, 'rb').read()
    dd = Docx(str(p))
    t = [dd.ptext(x) for x in dd.paragraphs()]
    n = sum(1 for _ in dd.tree.iter(f'{{{A}}}blip'))
    caps = [x for x in t if x.startswith('Figure ') or x.startswith('Supplementary Figure ')]
    print(f'{p.name}: {n} images, {len(caps)} captions, {len(t)} paragraphs')
print('swaps verified byte-for-byte')
