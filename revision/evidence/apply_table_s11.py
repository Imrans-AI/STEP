# -*- coding: utf-8 -*-
"""
Supplementary Table S11 rebuilt with the fourth source.

Methods 2.1 now attributes the pooled positives to four sources, so the table it
cites could not keep a three-source breakdown with 6,583 pairs (6.7%) booked as
"Unattributed" — those are UniPMT's. Composition measured here, from
data/processed/positive_pairs_multisource_v2.tsv joined to
data/splits_600_epitope_v2/train.tsv on the normalised (cdr3, epitope) key:

  source         pool pairs   train positives   % of train   pool epitopes
  VDJdb              82,581            81,868        85.7%             386
  UniPMT              8,234             5,667         5.9%             522
  McPAS-TCR           6,562             6,254         6.5%             291
  IEDB                2,185             1,739         1.8%             382
  in >= 2 sources        40                40        < 0.1%              -
  not attributable        -                 3        < 0.1%              -
  total              99,602            95,571         100%           1,281

Epitope counts are per source in the pre-normalisation spellings and sum to more
than 1,281, because an epitope may be recorded by several sources.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
XS = '{http://www.w3.org/XML/1998/namespace}space'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'

ROWS = [
    ("Source", "Pool positive pairs", "Train positives", "% of train positives",
     "Distinct epitopes in pool"),
    ("VDJdb", "82,581", "81,868", "85.7%", "386"),
    ("UniPMT", "8,234", "5,667", "5.9%", "522"),
    ("McPAS-TCR", "6,562", "6,254", "6.5%", "291"),
    ("IEDB", "2,185", "1,739", "1.8%", "382"),
    ("In ≥ 2 sources", "40", "40", "< 0.1%", "—"),
    ("Not attributable†", "—", "3", "< 0.1%", "—"),
    ("Total", "99,602", "95,571", "100%", "1,281"),
]

d = Docx(SI)
cap_i = d.find("Supplementary Table S11 | Source-database composition")
assert len(cap_i) == 1, cap_i
cap_p = list(d.paragraphs())[cap_i[0]]
OLD_CAP = d.ptext(cap_p)
NEW_CAP = (
    "Supplementary Table S11 | Source composition of the pooled positive pairs and of "
    "the training positives. Attribution is taken from the multi-source file from which "
    "the splits are built; a pair carried by more than one source is counted once, in "
    "the “In ≥ 2 sources” row. UniPMT was not named in the submitted "
    "manuscript, where its pairs appeared as the 6.7% booked to “unattributed”. "
    "Distinct-epitope counts are per source, in the spellings used before normalisation, "
    "and sum to more than the pool total of 1,281 because an epitope may be recorded by "
    "several sources. †Three training pairs cannot be matched back to the "
    "multi-source file, their keys having changed in the whitespace repair of Section 2.1."
)
d.replace_in_paragraph(cap_i[0], OLD_CAP, NEW_CAP, label='Table S11 caption')

# the table immediately follows its caption in the body
body = cap_p.getparent()
nxt = cap_p.getnext()
while nxt is not None and not nxt.tag.endswith('}tbl'):
    nxt = nxt.getnext()
assert nxt is not None, 'Table S11 not found after its caption'
tbl = nxt
trs = tbl.findall(W + 'tr')
print(f"Table S11 has {len(trs)} rows, target {len(ROWS)}")
while len(tbl.findall(W + 'tr')) < len(ROWS):
    trs = tbl.findall(W + 'tr')
    trs[-2].addnext(copy.deepcopy(trs[-2]))          # clone a body row
trs = tbl.findall(W + 'tr')
assert len(trs) == len(ROWS), (len(trs), len(ROWS))

for tr, vals in zip(trs, ROWS):
    tcs = tr.findall(W + 'tc')
    assert len(tcs) == len(vals), (len(tcs), len(vals))
    for tc, text in zip(tcs, vals):
        cps = tc.findall(W + 'p')
        for extra in cps[1:]:
            tc.remove(extra)
        runs = list(cps[0].iter(W + 'r'))
        for r in runs[1:]:
            r.getparent().remove(r)
        t = runs[0].find(W + 't')
        t.text = text
        t.set(XS, 'preserve')

d.save()
for label, old, new, n in d.log:
    print(f"  OK  {label}: {n}")

chk = Docx(SI)
tb = None
for t in chk.tree.iter(W + 'tbl'):
    txt = ''.join(x.text or '' for x in t.iter(W + 't'))
    if txt.startswith('SourcePool positive pairs'):
        tb = t
assert tb is not None
for tr in tb.findall(W + 'tr'):
    print(" | ".join(''.join(x.text or '' for x in tc.iter(W + 't'))
                     for tc in tr.findall(W + 'tc')))
# The only surviving "6.7%" is the new caption's own back-reference to the
# submitted figure; the "Unattributed" row it described is gone.
full = chk.full_text()
assert 'Unattributed*' not in full
assert full.count('6.7%') == 1 and 'booked to \u201cunattributed\u201d' in full
print("\nTable S11 rebuilt; the only 6.7% left is the caption's back-reference")
