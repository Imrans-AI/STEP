"""
Rewrite the Discussion clause attributing the DAISY margin to the contact prior.

The margin is real: STEP(seed 42) - DAISY = +0.0421, epitope-level cluster
bootstrap P = 2.80e-3, 95% CI [+0.0158, +0.0692] over 576 test epitopes
(R1-5_delong.md §5). What produces it is not established: R2-4_prior_controls.md
places the prior's own contribution within seed variance.

Read-only on revision/as_submitted/.
"""
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
EM, RSQ = '—', '’'

OLD = ('The 5.4-percentage-point margin over DAISY indicates that the contact-prior '
       'formulation contributes information beyond physicochemical features alone, '
       f'although retrained controls {EM} no prior, uniform, shuffled and '
       f'length-only, three seeds each {EM} place the prior{RSQ}s own contribution '
       'to accuracy within seed variance, so that margin is not attributable to the '
       'prior on present evidence.')

NEW = (f'On the corrected benchmark STEP{RSQ}s margin over DAISY is 4.2 percentage '
       'points (epitope-level cluster bootstrap P = 2.8 × 10⁻³, 95% CI '
       '+1.6 to +6.9 points across 576 test epitopes), so the margin itself is '
       'robust to resampling the epitopes rather than the rows. Its source, however, '
       f'is not established: retrained controls {EM} no prior, uniform, shuffled and '
       f'length-only, three seeds each {EM} place the structural prior{RSQ}s own '
       'contribution to accuracy within seed variance, so the present ablation does '
       'not isolate what produces the margin.')

d = Docx(MS)
before = [d.ptext(p) for p in d.paragraphs()]
d.replace(OLD, NEW, expected=1, label='Discussion, DAISY-margin attribution')
d.save()

d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
assert len(after) == len(before)
for i, (b, a) in enumerate(zip(before, after)):
    if b != a:
        print(f'=== paragraph [{i}] ===\n- {OLD}\n\n+ {NEW}\n')
assert 'indicates that the contact-prior formulation contributes information' \
    not in '\n'.join(after)
print('old attribution removed; paragraphs', len(after))

import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK')
