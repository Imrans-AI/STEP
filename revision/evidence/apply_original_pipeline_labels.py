# -*- coding: utf-8 -*-
"""
Three small corrections.

1. Supplementary Figures S10 and S11 are labelled "original pipeline", as Table 2's
   bio-negative and 300-epitope rows are.  S10's cross-reference also had to be
   repaired: it pointed the reader at overlap-free values "reported in the main
   text" that the main text no longer reports.
2. Section 2.1's cross-source check labels its 0.755 as the original-pipeline
   figure, which Section 2.5 already declares for that analysis.
3. Reference 52 (PyMOL) becomes a superscript citation run.  It is the last plain-text
   citation in the manuscript; the run carrying it is split in three so that only the
   number takes vertAlign="superscript", cloned from an existing citation run rather
   than hand-built.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
XS = '{http://www.w3.org/XML/1998/namespace}space'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'

# ============================================================ 1. S10 and S11 ===
s = Docx(SI)
s.replace(
    "versus 0.680 and 0.615 on the pair-level overlap-free subsets reported in the main "
    "text.",
    "versus 0.680 and 0.615 on the pair-level overlap-free subsets.",
    expected=1, label='S10: drop the stale cross-reference')
s.replace(
    "leaving the novel-identity bins without negatives.",
    "leaving the novel-identity bins without negatives. All values in this figure are "
    "original-pipeline figures and were not regenerated under the corrected pipeline "
    "(Section 2.5); for the overlap-free subsets the corrected pipeline gives 0.591 "
    "± 0.019 and 0.477 ± 0.027, which are the values reported in the main text.",
    expected=1, label='S10: original-pipeline label')
s.replace(
    "Dashed line marks perfect calibration.",
    "Dashed line marks perfect calibration. All values in this figure are "
    "original-pipeline figures and were not regenerated under the corrected pipeline "
    "(Section 2.5); on the unseen-epitope test set the retrained models give ECE "
    "0.254 ± 0.054 and Brier 0.283 ± 0.033, as reported in the main text.",
    expected=1, label='S11: original-pipeline label')
s.save()

# ========================================== 2. the cross-source check's 0.755 ==
d = Docx(MS)
d.replace(
    "within 0.02 of the full model (0.755).",
    "within 0.02 of the full model as that pipeline computed it (0.755); both are "
    "original-pipeline figures, this cross-source analysis not having been reconstructed "
    "under the corrected pipeline (Section 2.5).",
    expected=1, label='2.1: label the cross-source 0.755')

# ====================================================== 3. reference 52 ========
ps = list(d.paragraphs())
hits = [i for i, p in enumerate(ps) if 'PyMOL52' in d.ptext(p)]
assert len(hits) == 1, hits
para = ps[hits[0]]

# an existing superscript citation run to clone the properties from
exemplar = None
for p in ps:
    for t in d.texts(p):
        if (t.text or '').strip() == '51':
            r = t.getparent()
            rpr = r.find(W + 'rPr')
            if rpr is not None and rpr.find(W + 'vertAlign') is not None:
                exemplar = r
                break
    if exemplar is not None:
        break
assert exemplar is not None, 'no superscript citation run to clone'

target = None
for t in d.texts(para):
    if 'PyMOL52' in (t.text or ''):
        target = t
        break
assert target is not None
run = target.getparent()
head, tail = (target.text or '').split('PyMOL52', 1)

target.text = head + 'PyMOL'
target.set(XS, 'preserve')

sup = copy.deepcopy(exemplar)
st = sup.find(W + 't')
st.text = '52'
st.set(XS, 'preserve')

rest = copy.deepcopy(run)
rt = rest.find(W + 't')
rt.text = tail
rt.set(XS, 'preserve')

run.addnext(rest)
run.addnext(sup)
d.log.append(('ref 52 superscript', 'PyMOL52', 'PyMOL + superscript 52', 1))
d.save()

# ------------------------------------------------------------------ verify ----
for label, old, new, n in list(s.log) + list(d.log):
    print(f"  OK  {label}: {n}")
chk = Docx(MS)
cps = list(chk.paragraphs())
p = [q for q in cps if 'PyMOL52' in chk.ptext(q)]
assert len(p) == 1, 'visible text must still read PyMOL52'
sup_runs = 0
for t in chk.texts(p[0]):
    if (t.text or '') == '52':
        rpr = t.getparent().find(W + 'rPr')
        va = rpr.find(W + 'vertAlign') if rpr is not None else None
        if va is not None and va.get(W + 'val') == 'superscript':
            sup_runs += 1
assert sup_runs == 1, sup_runs
print('  ref 52 now carries vertAlign="superscript" in its own run')
schk = Docx(SI)
f = schk.full_text()
assert f.count('original-pipeline figures') == 2 and 'reported in the main text.' in f
print('  S10 and S11 labelled')
