"""Six further word-level cuts, to bring the abstract to 245 words.  Claims are
untouched; each cut removes a word that carries none."""
import sys
sys.path.insert(0, 'revision/evidence')
from docx_tools import Docx                                    # noqa: E402

MS = 'revision/working/Manuscript_fv_1_REVISED.docx'
d = Docx(MS)

CUTS = [
    ("pairing therefore underpin", "pairing underpin", 'drop "therefore"'),
    ("scarce beside the available sequence data", "scarce beside sequence data",
     'drop "the available"'),
    ("Here we present STEP", "We present STEP", 'drop "Here"'),
    ("framework that transfers interface knowledge", "framework transferring interface knowledge",
     'contract "that transfers"'),
    ("a shallow, largely hydrophobic interface", "a shallow, hydrophobic interface",
     'drop "largely"'),
    ("Attended interface positions coincide", "Attended positions coincide",
     'drop "interface" (repeated later in the sentence)'),
]
# Confined to the abstract paragraph: several of these strings also occur in the
# Introduction and Results, which must not change.
for old, new, why in CUTS:
    d.replace_in_paragraph(10, old, new, label=why)
d.save()

chk = Docx(MS)
abstract = chk.ptext(list(chk.paragraphs())[10])
words = abstract.split()
print("abstract words (whitespace tokens):", len(words))
print("abstract words (ignoring bare ±):", len([w for w in words if w != '±']))
print()
print(abstract)

# One token over after the six cuts: hyphenate the attributive compound.
d2 = Docx(MS)
d2.replace_in_paragraph(10, "a deep learning framework", "a deep-learning framework",
                        label='hyphenate "deep-learning"')
d2.save()
chk2 = Docx(MS)
ab2 = chk2.ptext(list(chk2.paragraphs())[10])
print("FINAL abstract words:", len(ab2.split()))
