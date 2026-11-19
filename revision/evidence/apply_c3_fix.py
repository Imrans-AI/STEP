"""
Two citations the C3 script dropped: ref 40 (flexibility) lost both its
superscript and the clause that followed it, and ref 42 (peptides) was never
inserted after the marker was removed.
"""
import copy
import sys

sys.path.insert(0, 'revision/evidence')
from lxml import etree

from docx_tools import Docx, W, WT

WR, WRPR, WVA = f'{{{W}}}r', f'{{{W}}}rPr', f'{{{W}}}vertAlign'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
d = Docx(MS)


def is_sup(r):
    pr = r.find(WRPR)
    if pr is None:
        return False
    va = pr.find(WVA)
    return va is not None and va.get(f'{{{W}}}val') == 'superscript'


sup_model = next(r for p in d.paragraphs() for r in p.iter(WR)
                 if is_sup(r) and ''.join(x.text or '' for x in r.iter(WT)) == '28')


def cite(anchor_text, number, tail):
    """Split the run holding anchor_text and insert a superscript citation."""
    for p in d.paragraphs():
        if anchor_text not in d.ptext(p):
            continue
        host = next((r for r in p.iter(WR)
                     if anchor_text in ''.join(x.text or '' for x in r.iter(WT))),
                    None)
        if host is None:
            continue
        ts = [t for t in host.iter(WT)]
        assert len(ts) == 1, 'anchor run is split'
        txt = ts[0].text
        i = txt.index(anchor_text) + len(anchor_text)
        head, rest = txt[:i], txt[i:]
        ts[0].text = head
        rs = etree.Element(WR, nsmap=host.nsmap)
        pr = sup_model.find(WRPR)
        if pr is not None:
            rs.append(copy.deepcopy(pr))
        etree.SubElement(rs, WT).text = str(number)
        rt = etree.Element(WR, nsmap=host.nsmap)
        prb = host.find(WRPR)
        if prb is not None:
            rt.append(copy.deepcopy(prb))
        t2 = etree.SubElement(rt, WT)
        t2.text = tail + rest
        t2.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        host.addnext(rt)
        host.addnext(rs)
        return True
    return False


assert cite('side-chain flexibility index of Bhaskaran and Ponnuswamy', 40,
            ', used on its native range'), 'ref 40 anchor not found'
assert cite('the same implementation of the peptides package', 42, ''), \
    'ref 42 anchor not found'
d.save()

d2 = Docx(MS)
import re
t = [d2.ptext(p) for p in d2.paragraphs()]
full = '\n'.join(t)
for frag in ('Bhaskaran and Ponnuswamy40, used on its native range',
             'implementation of the peptides package42;'):
    assert frag in full, frag
print('both citations in place')
ps = list(d2.paragraphs())
refstart = [i for i, x in enumerate(t) if x.strip() == 'References'][0]
cited = set()
for p in ps[:refstart]:
    segs = []
    for r in p.iter(WR):
        tx = ''.join(x.text or '' for x in r.iter(WT))
        if not tx:
            continue
        s = is_sup(r)
        if segs and segs[-1][0] == s:
            segs[-1][1] += tx
        else:
            segs.append([s, tx])
    for s, tx in segs:
        if s and re.fullmatch(r'[0-9,–-]+', tx.strip() or 'x'):
            for part in tx.strip().split(','):
                m = re.fullmatch(r'(\d+)\s*[–-]\s*(\d+)', part.strip())
                if m:
                    cited.update(range(int(m.group(1)), int(m.group(2)) + 1))
                elif part.strip().isdigit():
                    cited.add(int(part.strip()))
missing = [n for n in range(1, 59) if n not in cited]
body = '\n'.join(t[:refstart])
print('not cited as a superscript run:', missing,
      '-> plain-text hits:',
      {n: len(re.findall(r'[A-Za-z\)\.]' + str(n) + r'(?!\d)', body)) for n in missing})
