"""
R3-8: rewrite the Data and Software Availability statement and Table S10's
commit cell.

The DOI in the statement is a real hyperlink (rId11), so the relationship
target has to move with the visible text or the paragraph would read one DOI
and link to another. Two further DOIs are added as new hyperlinks with new
relationships, cloned from the existing one so they inherit its character
style.
"""
import copy
import re
import sys

sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx, W, XML_SPACE

WT = '{%s}t' % W

R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
SI = 'revision/working/STEP_Supplementary_FINAL_REVISED.docx'

CONCEPT = 'https://doi.org/10.5281/zenodo.21473726'
V2 = 'https://doi.org/10.5281/zenodo.24359126'
V12 = 'https://doi.org/10.5281/zenodo.21834133'
COMMIT = '1370f2806c2f7c0a03512558fdd2e8fdf649d5e0'

MID = (' under the MIT license. The data and trained models are archived at '
       'Zenodo under the concept DOI ')
AFTER_CONCEPT = (', which resolves to the most recent version. The analyses '
                 'reported here use version 2.0.0, ')
AFTER_V2 = (' (19 November 2026), deposited for this revision at code state '
            'commit ' + COMMIT + '. That archive contains the corrected '
            'consensus contact-probability prior, the whitespace-normalised '
            'and similarity-disjoint splits, the eighteen retrained '
            'checkpoints, and the corrected baseline predictions; README.md '
            'and MANIFEST.tsv sit at the record root, the first setting out '
            'what changed from the released version and why, the second '
            'giving the path, size and SHA-256 checksum of every archived '
            'file. The cancer-specific held-out test sets for melanoma and '
            'lung, in both full and overlap-free forms, are unchanged and '
            'remain in version 1.2.0, ')

d = Docx(MS)
p = list(d.paragraphs())[303]
runs = Docx.texts(p)
assert runs[3].text == 'https://doi.org/10.5281/zenodo.21473727', runs[3].text
assert runs[2].text.startswith(' under the MIT license.'), runs[2].text[:40]
assert runs[4].text.startswith('. Physicochemical'), runs[4].text[:40]

# ---- 1. the relationships -------------------------------------------------
rels = d.blobs['word/_rels/document.xml.rels'].decode()
assert rels.count('https://doi.org/10.5281/zenodo.21473727') == 1
rels = rels.replace('https://doi.org/10.5281/zenodo.21473727', CONCEPT)
used = set(re.findall(r'Id="(rId\d+)"', rels))
rid_v2, rid_v12 = 'rId900', 'rId901'
assert rid_v2 not in used and rid_v12 not in used
tmpl = ('<Relationship Id="%s" Type="http://schemas.openxmlformats.org/'
        'officeDocument/2006/relationships/hyperlink" Target="%s" '
        'TargetMode="External"/>')
rels = rels.replace('</Relationships>',
                    tmpl % (rid_v2, V2) + tmpl % (rid_v12, V12) +
                    '</Relationships>')
d.blobs['word/_rels/document.xml.rels'] = rels.encode()

# ---- 2. the visible text --------------------------------------------------
runs[2].text = MID
runs[2].set(XML_SPACE, 'preserve')
runs[3].text = CONCEPT

hl = runs[3].getparent()
while hl is not None and not hl.tag.endswith('}hyperlink'):
    hl = hl.getparent()
assert hl is not None and hl.get('{%s}id' % R) == 'rId11'

run_tmpl = runs[2].getparent()          # a plain <w:r> carrying body style


def plain(text):
    r = copy.deepcopy(run_tmpl)
    t = r.find(WT)
    t.text = text
    t.set(XML_SPACE, 'preserve')
    return r


def link(rid, url):
    h = copy.deepcopy(hl)
    h.set('{%s}id' % R, rid)
    for t in h.iter(WT):
        t.text = url
        t.set(XML_SPACE, 'preserve')
    return h


# inserted in reverse so each lands directly after the concept-DOI hyperlink
for node in (link(rid_v12, V12), plain(AFTER_V2), link(rid_v2, V2),
             plain(AFTER_CONCEPT)):
    hl.addnext(node)

d.save()

# ---- 3. Table S10's commit cell ------------------------------------------
s = Docx(SI)
s.replace('c9e4ea3', '1370f28 (archived in Zenodo v2.0.0)',
          expected=1, label='Table S10 STEP commit')
s.save()

# ---- 4. verify ------------------------------------------------------------
chk = Docx(MS)
t = chk.ptext(list(chk.paragraphs())[303])
print(t)
print()
for needle in (CONCEPT, V2, V12, COMMIT, 'README.md', 'MANIFEST.tsv',
               'eighteen retrained checkpoints'):
    assert needle in t, needle
    print('  present:', needle)
assert '21473727' not in chk.full_text(), 'old version DOI still present'
rels = chk.blobs['word/_rels/document.xml.rels'].decode()
for rid, url in (('rId11', CONCEPT), (rid_v2, V2), (rid_v12, V12)):
    assert 'Id="%s"' % rid in rels and url in rels
    print('  rel ok: %s -> %s' % (rid, url))
assert '21473727' not in rels, 'stale hyperlink target'

chk2 = Docx(SI)
assert 'c9e4ea3' not in chk2.full_text()
assert '1370f28 (archived in Zenodo v2.0.0)' in chk2.full_text()
print('  Table S10 commit cell updated')
print('\nboth documents saved')
