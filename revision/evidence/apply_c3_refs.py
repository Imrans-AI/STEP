"""
C3: resolve [[NEW REF 1]], [[NEW REF 2]] and [[PENDING: R1-1a]].

Reference order. [[NEW REF 1]] sits in 2.3.1's provenance paragraph BEFORE that
paragraph's Kyte-Doolittle citation, which is now ref 40's first use, so the
flexibility reference becomes 40 and Kyte-Doolittle shifts to 41. [[NEW REF 2]]
is first cited in 2.3.5 and becomes 42. Old 41-56 therefore become 43-58.
"""
import re
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR, WRPR, WVA, WP = f'{{{W}}}r', f'{{{W}}}rPr', f'{{{W}}}vertAlign', f'{{{W}}}p'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'

d = Docx(MS)
ps = list(d.paragraphs())
texts = [d.ptext(p) for p in ps]


def is_sup(r):
    pr = r.find(WRPR)
    if pr is None:
        return False
    va = pr.find(WVA)
    return va is not None and va.get(f'{{{W}}}val') == 'superscript'


# ------------------------------------------------ renumber old 41-56 -> 43-58
changed = 0
refstart = [i for i, x in enumerate(texts) if x.strip() == 'References'][0]
for i, p in enumerate(ps):
    if i >= refstart:
        break
    for r in p.iter(WR):
        if not is_sup(r):
            continue
        ts = [t for t in r.iter(WT)]
        txt = ''.join(t.text or '' for t in ts)
        if not re.fullmatch(r'[0-9,–—\-\s]+', txt or 'x'):
            continue
        new = re.sub(r'\d+', lambda m: str(int(m.group()) + 2)
                     if 41 <= int(m.group()) <= 56 else m.group(), txt)
        if new != txt:
            ts[0].text = new
            for t in ts[1:]:
                t.text = ''
            changed += 1
print(f'superscript runs renumbered: {changed}')
# ResNet's 41 was converted to a superscript run earlier, so the loop above
# already moved it; PyMOL's citation is still plain text, as recorded.
d.replace('PyMOL50', 'PyMOL52', expected=1, label='PyMOL 50->52')
# Kyte-Doolittle 40 -> 41 (single superscript occurrence, in 2.3.1)
for p in ps[:refstart]:
    for r in p.iter(WR):
        if is_sup(r):
            ts = [t for t in r.iter(WT)]
            if ''.join(t.text or '' for t in ts) == '40':
                ts[0].text = '41'
                print('Kyte-Doolittle citation 40 -> 41')

# reference list: 41-56 -> 43-58, descending so numbers do not collide
texts = [d.ptext(p) for p in d.paragraphs()]
ps = list(d.paragraphs())
for n in range(56, 40, -1):
    hit = [i for i, x in enumerate(texts) if x.strip().startswith(f'({n})')]
    assert len(hit) == 1, (n, hit)
    d.replace_in_paragraph(hit[0], f'({n})', f'({n + 2})', label=f'ref {n}->{n+2}')
texts = [d.ptext(p) for p in d.paragraphs()]
hit = [i for i, x in enumerate(texts) if x.strip().startswith('(40)')]
assert len(hit) == 1
d.replace_in_paragraph(hit[0], '(40)', '(41)', label='ref 40->41')

# ------------------------------------------------ insert the two new entries
ps = list(d.paragraphs())
texts = [d.ptext(p) for p in ps]
anchor41 = [i for i, x in enumerate(texts) if x.strip().startswith('(41)Kyte')][0]
model = ps[anchor41]


def ref_para(text):
    p = etree.fromstring(etree.tostring(model))
    ts = [t for t in p.iter(WT)]
    ts[0].text = text
    for t in ts[1:]:
        t.text = ''
    return p


FLEX = ('(40)Bhaskaran, R.; Ponnuswamy, P. K. Dynamics of Amino Acid Residues in '
        'Globular Proteins. Int. J. Pept. Protein Res. 1984, 24 (2), 180'
        '–191. https://doi.org/10.1111/j.1399-3011.1984.tb00944.x.')
PEP = ('(42)Larralde, M. peptides.py: Physicochemical Properties, Indices and '
       'Descriptors for Amino-Acid Sequences, version 0.5.0; a Python port of the '
       'R package Peptides. https://github.com/althonos/peptides.py.')
model.addprevious(ref_para(FLEX))
ps = list(d.paragraphs())
texts = [d.ptext(p) for p in ps]
a43 = [i for i, x in enumerate(texts) if x.strip().startswith('(43)')][0]
ps[a43].addprevious(ref_para(PEP))
print('inserted reference entries 40 and 42')

# ------------------------------------------------ resolve the markers
d.replace('of Bhaskaran and Ponnuswamy [[NEW REF 1]], used on its native range',
          'of Bhaskaran and Ponnuswamy', expected=1, label='NEW REF 1 prose')
for p in d.paragraphs():
    if 'of Bhaskaran and Ponnuswamy' in d.ptext(p) and '[[' not in d.ptext(p):
        runs = [r for r in p.iter(WR)]
        host = next(r for r in runs
                    if 'of Bhaskaran and Ponnuswamy' in
                    ''.join(x.text or '' for x in r.iter(WT)))
        ts = [t for t in host.iter(WT)]
        txt = ts[0].text
        i = txt.index('of Bhaskaran and Ponnuswamy') + len('of Bhaskaran and Ponnuswamy')
        head, tail = txt[:i], txt[i:]
        ts[0].text = head
        sup = next(r for q in d.paragraphs() for r in q.iter(WR)
                   if is_sup(r) and ''.join(x.text or '' for x in r.iter(WT)) == '41')
        import copy
        rs = etree.Element(WR, nsmap=host.nsmap)
        pr = sup.find(WRPR)
        if pr is not None:
            rs.append(copy.deepcopy(pr))
        t = etree.SubElement(rs, WT)
        t.text = '40'
        rt = etree.Element(WR, nsmap=host.nsmap)
        prb = host.find(WRPR)
        if prb is not None:
            rt.append(copy.deepcopy(prb))
        t2 = etree.SubElement(rt, WT)
        t2.text = ', used on its native range' + tail
        t2.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        host.addnext(rt)
        host.addnext(rs)
        break
print('[[NEW REF 1]] resolved to 40')

d.replace('the peptides package [[NEW REF 2]];',
          'the peptides package;', expected=1, label='NEW REF 2 prose')
d.replace('[[PENDING: R1-1a — new Supplementary table of the five per-residue '
          'scales]]', 'Supplementary Table S12', expected=1,
          label='R1-1a marker -> Table S12')
d.save()

d2 = Docx(MS)
after = [d2.ptext(p) for p in d2.paragraphs()]
full = '\n'.join(after)
assert '[[' not in full, re.findall(r'\[\[[^\]]+\]\]', full)
refs = [x for x in after if re.match(r'^\(\d+\)', x.strip())]
nums = [int(re.match(r'^\((\d+)\)', x.strip()).group(1)) for x in refs]
assert nums == list(range(1, 59)), (len(nums), [n for i, n in enumerate(nums) if n != i + 1][:5])
print(f'reference list now 1-{len(nums)} contiguous; no markers left')
