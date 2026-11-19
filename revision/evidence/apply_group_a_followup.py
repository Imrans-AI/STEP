"""Three locations the group-A script did not reach."""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
PM = '±'
d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]

# Discussion still carried the v1 MART-1 figure and the v1 per-epitope fractions
d.replace('with the MART-1/Melan-A antigen at 0.983',
          f'with the MART-1/Melan-A antigen AAGIGILTV at 0.960 {PM} 0.040',
          expected=1, label='B4 Discussion, MART-1')
d.replace('Per-epitope analysis shows 66% of evaluated epitopes exceed ROC-AUC 0.7 '
          'and 43% exceed 0.9',
          'Per-epitope analysis shows 64% of evaluated epitopes exceed ROC-AUC 0.7 '
          'and 43% exceed 0.9',
          expected=1, label='A4 Discussion, per-epitope fractions')

# Methods 2.9 — the two variants that were not regenerated
d.replace(
    'All threshold-dependent metrics were reported at each variant’s '
    'Youden’s J optimal threshold.',
    'The full and no-structure variants were retrained under the corrected pipeline '
    'with three seeds each, and their threshold-dependent metrics are reported at a '
    'Youden’s J threshold fitted on the validation set (Section 2.6). The '
    '300-epitope and biological-negative variants were not regenerated under that '
    'pipeline: both require their own training splits, which were not rebuilt from '
    'the repaired data. They are reported as originally computed, at a threshold '
    'fitted on the test set, and are not comparable term by term with the other two.',
    expected=1, label='Methods 2.9, variants not regenerated')

d.save()

d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
import difflib
for k, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'\n===== [{k}] =====')
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, b, a, autojunk=False).get_opcodes():
            if tag == 'equal':
                continue
            if b[i1:i2]:
                print(f'  - {b[i1:i2]}')
            if a[j1:j2]:
                print(f'  + {a[j1:j2]}')
full = '\n'.join(after)
for bad in ('0.983', '581 ', 'ECE 0.33', 'PENDING: R1-7'):
    assert bad not in full, f'still present: {bad!r}'
print('\nchecks pass; paragraphs', len(after), '| abstract words', len(after[10].split()))
import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK')
