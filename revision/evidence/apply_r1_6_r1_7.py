"""R1-6 and R1-7 — applied to revision/working/Manuscript_fv_1_REVISED.docx.

R1-6  Introduction: one new paragraph placing STEP in the physicochemical
      interaction-map lineage (ImRex -> DAISY -> STEP). ImRex is already in the
      reference list as ref 29 (Moris et al. 2021, Brief. Bioinform. 22(4):bbaa318,
      "...a New Perspective Derived from Image Classification" — the ImRex paper),
      so the citation is reused rather than added; no renumbering.

R1-7  Section 2.8: state that the overlap-free subset removes exact TCR-epitope
      pairs and not epitopes, with the measured epitope-familiarity figures, and
      revise every claim of "novel-antigen generalization" or equivalent.

Figures used in 2.8 are computed below from the splits the manuscript reports
(data/splits_600_epitope/train.tsv, i.e. v1) and asserted before any edit is made.
R2-5_cancer_memorisation.md quotes 12/31 and 66.4% because that analysis runs on
the v2 splits; both are printed by this script.

Read-only on revision/as_submitted/. Nothing else is touched.
"""
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path

from lxml import etree

sys.path.insert(0, str(Path(__file__).parent))
from docx_tools import Docx, W, XML_SPACE  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MS = ROOT / 'revision' / 'working' / 'Manuscript_fv_1_REVISED.docx'
SUBMITTED = ROOT / 'revision' / 'as_submitted' / 'Manuscript_fv_1.docx'
SUBMITTED_MD5 = 'e7b08111d935475b7bbfd3cc02612d33'
WP = f'{{{W}}}p'
WR = f'{{{W}}}r'
WT = f'{{{W}}}t'
WRPR = f'{{{W}}}rPr'
WPPR = f'{{{W}}}pPr'


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


# ---------------------------------------------------------------- measurement
def tsv(p):
    with open(p) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def epitope_familiarity(train_tsv):
    train = {r['antigen.epitope'].strip() for r in tsv(ROOT / train_tsv)}
    out = {'train_epitopes': len(train)}
    for name, path in [('melanoma', 'data/cancer_specific/melanoma_test.tsv'),
                       ('lung', 'data/cancer_specific/lung_test.tsv')]:
        rows = tsv(ROOT / path)
        eps = {r['antigen.epitope'].strip() for r in rows}
        seen = {e for e in eps if e in train}
        n = sum(1 for r in rows if r['antigen.epitope'].strip() in seen)
        out[name] = {'epitopes': len(eps), 'epitopes_seen': len(seen),
                     'rows': len(rows), 'rows_seen': n,
                     'pct_rows_seen': round(100 * n / len(rows), 1)}
    pos = [r for r in tsv(ROOT / 'data/cancer_specific/melanoma_test.tsv')
           if r['label'] == '1']
    mart1 = sum(1 for r in pos
                if r['antigen.epitope'].strip() in ('ELAGIGILTV', 'EAAGIGILTV'))
    out['melanoma_mart1_pct_of_positives'] = round(100 * mart1 / len(pos), 1)
    return out


# ---------------------------------------------------------- paragraph builder
def run_templates(p):
    """A body run and a superscript run cloned from an existing paragraph."""
    body = sup = None
    for r in p.findall(WR):
        rpr = r.find(WRPR)
        is_sup = rpr is not None and rpr.find(f'{{{W}}}vertAlign') is not None
        if r.find(WT) is None:
            continue
        if is_sup and sup is None:
            sup = copy.deepcopy(r)
        elif not is_sup and body is None:
            body = copy.deepcopy(r)
    if body is None or sup is None:
        raise AssertionError('template paragraph lacks a body or superscript run')
    return body, sup


def build_paragraph(template_p, segments):
    """New <w:p> cloned from template_p, carrying `segments` = [(text, sup), ...]."""
    body_t, sup_t = run_templates(template_p)
    p = copy.deepcopy(template_p)
    for child in list(p):
        if child.tag != WPPR:
            p.remove(child)
    for text, is_sup in segments:
        r = copy.deepcopy(sup_t if is_sup else body_t)
        for child in list(r):
            if child.tag != WRPR:
                r.remove(child)
        t = etree.SubElement(r, WT)
        t.text = text
        t.set(XML_SPACE, 'preserve')
        p.append(r)
    return p


# --------------------------------------------------------------------- edits
NEW_INTRO = [
    ("STEP sits within a specific line of this work: representations that encode a "
     "TCR–epitope pair as a two-dimensional physicochemical interaction map rather "
     "than as two independent sequences. ImRex introduced that representation, pairing "
     "each CDR3β residue with each epitope residue and treating the resulting matrix "
     "as an image for convolutional feature extraction", False),
    ("29", True),
    (". DAISY extended it with a hierarchical physicochemical representation and a "
     "condition-adaptive fusion module that combines local interaction features with "
     "sequence-level descriptors", False),
    ("28", True),
    (". STEP adopts both components — a five-channel physicochemical interaction map "
     "(Section 2.3.1) and seven global biophysical descriptors (Section 2.3.5) — and "
     "adds a third: a consensus contact-probability prior derived from 198 experimentally "
     "determined TCR–pMHC complexes, which weights every position of the map by how "
     "often that CDR3β–epitope residue pair is in contact across resolved "
     "structures. Where ImRex and DAISY learn which positions of the interaction map "
     "matter from the training pairs alone, STEP supplies that information from "
     "experimental interface geometry before any training pair is seen.", False),
]

ANCHOR = "These methods established machine learning for antigen-specific TCR prediction"


def section_2_8(fam):
    m, l = fam['melanoma'], fam['lung']
    words = {3: 'Three', 4: 'Four', 5: 'Five', 11: 'Eleven', 12: 'Twelve',
             13: 'Thirteen', 14: 'Fourteen'}
    n = m['epitopes_seen']
    return (
        " This removes exact TCR–epitope pairs, not epitopes: an epitope seen in "
        "training can still appear in the overlap-free subset, paired with a receptor it "
        f"was not seen with. {words[n]} of the {m['epitopes']} melanoma epitopes are "
        "present in the training epitope set — including both MART-1/Melan-A epitopes, "
        f"which together supply {fam['melanoma_mart1_pct_of_positives']:.0f}% of melanoma "
        f"positives — and those {words[n].lower()} carry {m['rows_seen']:,} of the "
        f"{m['rows']:,} "
        f"melanoma rows ({m['pct_rows_seen']}%) [[PENDING: R1-7 — these are the "
        "splits reported here; on the v2 splits the figures are 12 epitopes and 1,050 "
        "rows (66.4%), so update if the cancer evaluation is regenerated on v2]]. "
        f"{words[l['epitopes_seen']]} of the {l['epitopes']} lung epitopes are likewise "
        f"present in training, carrying {l['rows_seen']} of the {l['rows']} rows "
        f"({l['pct_rows_seen']}%). The overlap-free analysis therefore measures prediction "
        "on unseen TCR–epitope pairs under partial epitope familiarity, not on unseen "
        "epitopes.")


def main():
    assert md5(SUBMITTED) == SUBMITTED_MD5, 'submitted manuscript changed'
    before = md5(MS)
    fam_v1 = epitope_familiarity('data/splits_600_epitope/train.tsv')
    fam_v2 = epitope_familiarity('data/splits_600_epitope_v2/train.tsv')
    print('v1 (splits the manuscript reports):', json.dumps(fam_v1, indent=1))
    print('v2 (used by R2-5_cancer_memorisation.md):', json.dumps(fam_v2, indent=1))
    assert fam_v1['melanoma']['epitopes_seen'] == 13
    assert fam_v1['melanoma']['rows_seen'] == 1087
    assert fam_v1['melanoma']['pct_rows_seen'] == 68.7
    assert fam_v2['melanoma']['epitopes_seen'] == 12
    assert fam_v2['melanoma']['pct_rows_seen'] == 66.4

    d = Docx(MS)
    n_before = sum(1 for _ in d.paragraphs())

    # --- R1-6: new Introduction paragraph -----------------------------------
    hits = d.find(ANCHOR)
    assert len(hits) == 1, f'anchor paragraph found {len(hits)} times'
    anchor = list(d.paragraphs())[hits[0]]
    assert 'ImRex' not in d.full_text(), 'ImRex already mentioned'
    anchor.addnext(build_paragraph(anchor, NEW_INTRO))
    print(f'R1-6: inserted new Introduction paragraph after paragraph {hits[0]}')

    # --- R1-7: Methods 2.8 ---------------------------------------------------
    d.replace(
        "96 lung samples (48+/48−). 95% confidence intervals were computed via "
        "1,000-iteration bootstrap.",
        "96 lung samples (48+/48−)." + section_2_8(fam_v1)
        + " 95% confidence intervals were computed via 1,000-iteration bootstrap.",
        expected=1, label='R1-7 §2.8 scope of the overlap-free subset')

    # --- R1-7: claims --------------------------------------------------------
    d.replace("Cancer-specific generalization",
              "Cancer-specific evaluation under partial epitope familiarity",
              expected=1, label='R1-7 Results heading')

    d.replace(
        "the overlap-free estimate of 0.680 therefore represents a more stringent test "
        "of novel-antigen generalization and should be regarded as the conservative "
        "reference for translational claims",
        "the overlap-free estimate of 0.680 therefore represents a more stringent test "
        "— prediction on unseen TCR–epitope pairs, with most epitopes still "
        "familiar from training (Methods 2.8) — and should be regarded as the "
        "conservative reference for translational claims",
        expected=1, label='R1-7 Results [243] novel-antigen generalization')

    d.replace(
        "the full-set cancer AUC primarily reflects recall of training data rather than "
        "de novo generalization",
        "the full-set cancer AUC primarily reflects recall of training data rather than "
        "prediction on unfamiliar material",
        expected=1, label='R1-7 Results [244] de novo generalization')

    d.replace(
        "The overlap-free estimates therefore remain the appropriate measure of "
        "novel-antigen performance.",
        "The overlap-free estimates therefore remain the appropriate measure of "
        "performance on unseen TCR–epitope pairs; because the epitopes themselves are "
        "largely familiar, neither analysis measures generalization to novel antigens.",
        expected=1, label='R1-7 Results [244] novel-antigen performance')

    d.replace(
        "STEP generalized to tumor-associated antigens (melanoma ROC-AUC 0.913; 0.680 on "
        "an overlap-free subset)",
        "On tumor-associated antigens, STEP reached ROC-AUC 0.913 (melanoma), 0.680 with "
        "training pairs removed",
        expected=1, label='R1-7 Abstract')

    d.replace(
        "These findings suggest that structure-informed physicochemical encoding "
        "generalizes to therapeutically relevant antigens, including those underlying "
        "current TCR-based immunotherapies",
        "These findings suggest that structure-informed physicochemical encoding transfers "
        "to therapeutically relevant antigens, including those underlying current "
        "TCR-based immunotherapies",
        expected=1, label='R1-7 Discussion [307] generalizes -> transfers')

    d.replace(
        "indicating that lung generalization claims require larger validation cohorts",
        "indicating that claims for lung require larger validation cohorts",
        expected=1, label='R1-7 Discussion [308] lung generalization claims')

    out = d.save()
    after = md5(out)

    # ------------------------------------------------------------- verification
    d2 = Docx(MS)
    n_after = sum(1 for _ in d2.paragraphs())
    text = d2.full_text()
    assert n_after == n_before + 1, f'paragraph count {n_before} -> {n_after}'
    for gone in ['novel-antigen generalization', 'novel-antigen performance',
                 'de novo generalization', 'Cancer-specific generalization',
                 'STEP generalized to tumor-associated antigens']:
        assert gone not in text, f'still present: {gone!r}'
    for present in ['ImRex introduced that representation',
                    'This removes exact TCR–epitope pairs, not epitopes',
                    'under partial epitope familiarity']:
        assert present in text, f'missing: {present!r}'
    abstract = next(d2.ptext(p) for p in d2.paragraphs()
                    if p is not None and d2.ptext(p).startswith('T-cell receptors (TCRs) recognize'))
    print(f'\nabstract words: {len(abstract.split())}')
    import zipfile
    with zipfile.ZipFile(MS) as z:
        assert z.testzip() is None
    assert md5(SUBMITTED) == SUBMITTED_MD5, 'submitted manuscript changed'

    log = {'file': str(MS.relative_to(ROOT)), 'md5_before': before, 'md5_after': after,
           'paragraphs': [n_before, n_after],
           'abstract_words': len(abstract.split()),
           'familiarity_v1': fam_v1, 'familiarity_v2': fam_v2,
           'edits': [{'label': l, 'old': o[:120], 'new': n[:160], 'n': k}
                     for l, o, n, k in d.log]}
    (Path(__file__).parent / 'r1_6_r1_7_applied.json').write_text(json.dumps(log, indent=1))
    print(f"\n{len(d.log)} replacements + 1 inserted paragraph")
    print(f"md5 {before} -> {after}")


if __name__ == '__main__':
    main()
