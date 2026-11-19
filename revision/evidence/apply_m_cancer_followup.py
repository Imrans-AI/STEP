"""
One location missed by apply_m1_m2_m5_cancer.py: the nearest-neighbour paragraph
restates the overlap-free pair of numbers a second time.
"""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
d = Docx(MS)
d.replace('performance falls to 0.680 (melanoma) and 0.615 (lung)',
          'performance falls to 0.610 (melanoma) and 0.465 (lung)',
          expected=1, label='cancer Results, nearest-neighbour restatement')
d.save()

d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
full = '\n'.join(after)
for bad in ('ERGO-LSTM (0.589)', 'ERGO-AE (0.570)', 'TEINet (0.493)',
            'ATM-TCR (0.479)', 'source of much of the gain', '5.4-percentage-point',
            '0.680', '0.615', 'every pairwise comparison significant by DeLong',
            'ROC-AUC of 0.719'):
    assert bad not in full, f'still present: {bad!r}'
print('all superseded strings removed')
assert sum(t.strip() == '0.913' for t in after) == 1, 'Table 1 cell touched'
print("Table 1's DAISY seen-pair 0.913 cell untouched; 0.910 occurrences:",
      full.count('0.910'))
import re
print('markers:', len(re.findall(r'\[\[[^\]]+\]\]', full)))
import zipfile
with zipfile.ZipFile(MS) as z:
    assert z.testzip() is None
print('zip OK, paragraphs', len(after))
