"""
Item 1: call the attribution method what the code computes.

SCORECAM_4QOK.py and the Fig 5/6 scripts compute gradient x input on the
physicochemical interaction map - the gradient of the predicted probability with
respect to the input map, multiplied elementwise by the map, summed over the
five channels, rectified, max-normalised. The manuscript described Score-CAM,
including its nine-step algorithm with equations, which is a different method.

Consequences handled here:
  * the nine algorithm paragraphs (2.12) are Score-CAM's and are removed;
  * reference 50 (Wang et al., Score-CAM) loses its only citation, so the entry
    is deleted and references 51-57 renumber to 50-56, in the list and in every
    in-text citation.
"""
import re
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx, W, WT

WR, WRPR, WVA = f'{{{W}}}r', f'{{{W}}}rPr', f'{{{W}}}vertAlign'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'
GRAD = 'gradient × input'

d = Docx(MS)
ps = list(d.paragraphs())
before = [d.ptext(p) for p in ps]


def is_sup(r):
    pr = r.find(WRPR)
    if pr is None:
        return False
    va = pr.find(WVA)
    return va is not None and va.get(f'{{{W}}}val') == 'superscript'


# ---------------------------------------------- 2.12 heading and description
d.replace_in_paragraph(147, '2.12 Score-CAM Structural Visualization',
                       '2.12 Attribution Maps and Structural Projection',
                       label='2.12 heading')

p148 = ps[148]
runs = [r for r in p148.iter(WR)]
sup50 = [r for r in runs if is_sup(r)
         and ''.join(x.text or '' for x in r.iter(WT)) == '50']
assert len(sup50) == 1, 'ref 50 citation not where expected'
sup50[0].getparent().remove(sup50[0])
ts = [t for r in runs if r is not sup50[0] for t in r.iter(WT)]
ts[0].text = (
    'Attribution maps were computed as ' + GRAD + ' on the five-channel '
    'physicochemical interaction map: the gradient of the predicted binding '
    'probability with respect to the input map was multiplied elementwise by the '
    'map, summed over the five channels, rectified and divided by its maximum, '
    'giving one non-negative score per TCR-position / epitope-position pair. The '
    'submitted version of this manuscript described the procedure as Score-CAM; '
    'the implementation that produced the figures computes ' + GRAD + ', and the '
    'description is corrected here to match the code.')
for t in ts[1:]:
    t.text = ''

# ---------------------------------------------- delete the Score-CAM algorithm
for i in range(157, 148, -1):
    ps[i].getparent().remove(ps[i])
print('removed 9 algorithm paragraphs (149-157)')

# ---------------------------------------------- rename in prose and captions
d.replace('used ablation, SHAP and Score-CAM analyses',
          f'used ablation, SHAP and {GRAD} attribution analyses',
          expected=1, label='Introduction')
d.replace('we applied Score-CAM to the final convolutional layer',
          f'we computed {GRAD} attribution on the physicochemical interaction map',
          expected=1, label='Results, lead-in')
d.replace('Figure 5 | Score-CAM structural visualization of learned '
          'TCR–epitope interactions. a, Score-CAM activation map',
          'Figure 5 | Attribution maps and structural projection of learned '
          f'TCR–epitope interactions. a, {GRAD.capitalize()} attribution map',
          expected=1, label='Fig 5 caption')
d.replace('b, STEP Score-CAM contact scores for the same pair',
          'b, STEP attribution scores for the same pair',
          expected=1, label='Fig 6 caption')
d.replace('Additional Score-CAM examples', 'Additional attribution-map examples',
          expected=1, label='Results, pointer to Fig S2')
d.replace('Two interpretability analyses converge on the same conclusion: '
          'Score-CAM attention localizes',
          'Two interpretability analyses converge on the same conclusion: '
          'attribution localizes',
          expected=1, label='Discussion')

# ---------------------------------------------- drop reference 50, renumber
ps = list(d.paragraphs())
texts = [d.ptext(p) for p in ps]
ref50 = [i for i, x in enumerate(texts) if x.strip().startswith('(50)Wang, H.')]
assert len(ref50) == 1, ref50
ps[ref50[0]].getparent().remove(ps[ref50[0]])
print('deleted reference entry (50) Wang et al., Score-CAM')

ps = list(d.paragraphs())
texts = [d.ptext(p) for p in ps]
for n in range(51, 58):
    hit = [i for i, x in enumerate(texts) if x.strip().startswith(f'({n})')]
    assert len(hit) == 1, (n, hit)
    d.replace_in_paragraph(hit[0], f'({n})', f'({n - 1})',
                           label=f'reference list {n}->{n-1}')

# in-text superscript citations
changed = 0
for p in d.paragraphs():
    for r in p.iter(WR):
        if not is_sup(r):
            continue
        ts = [t for t in r.iter(WT)]
        txt = ''.join(t.text or '' for t in ts)
        if not re.fullmatch(r'[0-9,–—\-\s]+', txt or 'x'):
            continue
        new = re.sub(r'\d+', lambda m: str(int(m.group()) - 1)
                     if 51 <= int(m.group()) <= 57 else m.group(), txt)
        if new != txt:
            ts[0].text = new
            for t in ts[1:]:
                t.text = ''
            changed += 1
print(f'renumbered {changed} superscript citation runs')
# PyMOL's citation is plain text, as ref 41's is
d.replace('rendering with PyMOL51.', 'rendering with PyMOL50.',
          expected=1, label='PyMOL citation 51->50')
d.save()

# ---------------------------------------------------------------- supplementary
s = Docx(SI)
s.replace('Supplementary Figure S2 | Additional Score-CAM structural examples. '
          'a, Score-CAM visualization',
          'Supplementary Figure S2 | Additional attribution-map examples. '
          f'a, {GRAD.capitalize()} attribution map',
          expected=1, label='SI Fig S2 caption a')
s.replace('b, Score-CAM visualization', 'b, Attribution map',
          expected=1, label='SI Fig S2 caption b')
s.save()

# ---------------------------------------------------------------- verify
d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
full = '\n'.join(after)
assert 'Score-CAM' not in full.replace(
    'described the procedure as Score-CAM', ''), 'Score-CAM still present'
s2 = Docx(SI)
assert 'Score-CAM' not in '\n'.join(s2.ptext(p) for p in s2.paragraphs())
refs = [x for x in after if re.match(r'^\(\d+\)', x.strip())]
print(f'\nreference entries: {len(refs)} (was 57); last = {refs[-1][:60]}')
nums = [int(re.match(r'^\((\d+)\)', x.strip()).group(1)) for x in refs]
assert nums == list(range(1, 57)), 'reference numbering broken'
print('reference list numbered 1-56 contiguously')
print('paragraphs', len(after), '(was', len(before), ')')
import zipfile
for f in (MS, SI):
    with zipfile.ZipFile(f) as z:
        assert z.testzip() is None
print('zips OK')
