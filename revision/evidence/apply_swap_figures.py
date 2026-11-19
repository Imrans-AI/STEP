"""
Item 3 + 4: the Fig 3i caption line, and swapping the regenerated images into
the working .docx files.

Each image part is replaced in place, and the drawing's extent is rescaled so
the new aspect ratio is honoured: Word stretches an image to the stored cx/cy,
so replacing bytes alone would distort any figure whose proportions changed.
Width is kept, height recomputed.
"""
import io
import sys
import zipfile
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
        'word/media/image3.png': FIG / 'Figure_3_rebuilt_full.png',
        'word/media/image5.png': FIG / 'Figure_5_rebuilt_full.png',
    },
    'revision/working/STEP_Supplementary_FINAL_REVISED.docx': {
        'word/media/image1.png': FIG / 'Figure_S1_dataset_flow_v2.png',
        'word/media/image4.png': FIG / 'Figure_S3_per_epitope_v2.png',
        'word/media/image8.png': FIG / 'Figure_S7_SHAP_corrected_labels.png',
    },
}

# ------------------------------------------------- item 3: the Fig 3i caption
d = Docx(str(BASE / 'revision/working/Manuscript_fv_1_REVISED.docx'))
d.replace(
    'clear separation between binding (coral) and non-binding (steel blue) '
    'clusters is consistent with the high melanoma ROC-AUC (0.910; silhouette '
    'score = 0.37).',
    'clear separation between binding (coral) and non-binding (steel blue) '
    'clusters is consistent with the high melanoma ROC-AUC (0.910; silhouette '
    'score = 0.37). Panel i derives from the originally submitted model and was '
    'not regenerated under the corrected pipeline; every other panel is rebuilt '
    'from the retrained models.',
    expected=1, label='Fig 3i caption line')
d.save()
print('Fig 3i caption line added')

# ------------------------------------------------- item 4: swap the images
for doc, mapping in JOBS.items():
    path = BASE / doc
    dd = Docx(str(path))
    rels = dd.blobs['word/_rels/document.xml.rels'].decode()
    import re
    rid2t = dict(re.findall(r'Id="([^"]+)"[^>]*Target="([^"]+)"', rels))
    t2rid = {f'word/{v}': k for k, v in rid2t.items()}

    for part, src in mapping.items():
        assert part in dd.blobs, f'{part} not in {doc}'
        old = Image.open(io.BytesIO(dd.blobs[part]))
        new_bytes = open(src, 'rb').read()
        new = Image.open(io.BytesIO(new_bytes))
        dd.blobs[part] = new_bytes
        rid = t2rid[part]
        # rescale the drawing extent: keep cx, recompute cy from the new ratio
        done = 0
        for blip in dd.tree.iter(f'{{{A}}}blip'):
            if blip.get(f'{{{R}}}embed') != rid:
                continue
            anc = blip
            while anc is not None and not anc.tag.endswith('}inline') \
                    and not anc.tag.endswith('}anchor'):
                anc = anc.getparent()
            assert anc is not None, 'no wp:inline/anchor ancestor'
            for ext in list(anc.iter(f'{{{WPNS}}}extent')) + \
                    list(anc.iter(f'{{{A}}}ext')):
                cx = ext.get('cx')
                if cx is None:
                    continue
                ext.set('cy', str(int(int(cx) * new.height / new.width)))
                done += 1
        assert done >= 1, f'no extent found for {part}'
        print(f'  {doc.split("/")[-1]}: {part} {old.size} -> {new.size} '
              f'({len(new_bytes)/1e6:.1f} MB), {done} extents rescaled')
    dd.save()

# ------------------------------------------------- verify
for doc, mapping in JOBS.items():
    path = BASE / doc
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        for part, src in mapping.items():
            assert z.read(part) == open(src, 'rb').read(), f'{part} not swapped'
    dd = Docx(str(path))
    t = [dd.ptext(p) for p in dd.paragraphs()]
    nimg = sum(1 for _ in dd.tree.iter(f'{{{A}}}blip'))
    caps = [x for x in t if x.startswith('Figure ') or
            x.startswith('Supplementary Figure ')]
    print(f'{path.name}: {nimg} images, {len(caps)} figure captions, '
          f'{len(t)} paragraphs')
print('all swaps verified byte-for-byte')
