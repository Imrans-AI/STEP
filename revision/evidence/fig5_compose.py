"""
Compose the full Figure 5: panels a and b kept from the submitted image, panels
c and d replaced with the rebuilt 4QOK attribution map and PyMOL render.

Panels a and b are the 3GSN pair, which was never in question - the defect was
confined to the lower row - so they are carried over rather than regenerated.
The crop boundary is measured from the submitted image, not guessed: the row
gap is found as the widest band of near-white rows in the middle third.
"""
from pathlib import Path
import io
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'

z = zipfile.ZipFile(BASE / 'revision/working/Manuscript_fv_1_REVISED.docx')
orig = Image.open(io.BytesIO(z.read('word/media/image5.png'))).convert('RGB')
W, H = orig.size

a = np.asarray(orig.resize((W // 8, H // 8)).convert('L'), dtype=np.float32)
rowmin = a.min(axis=1)
lo, hi = int(0.33 * len(rowmin)), int(0.62 * len(rowmin))
blank = np.where(rowmin[lo:hi] > 250)[0] + lo
runs, cur = [], [blank[0]]
for v in blank[1:]:
    if v == cur[-1] + 1:
        cur.append(v)
    else:
        runs.append(cur); cur = [v]
runs.append(cur)
band = max(runs, key=len)
# cut at the TOP of the gap, not its middle: the submitted panel letters c and d
# sit just below the gap and would otherwise survive the crop and duplicate the
# new ones.
# The submitted panel letters c and d sit ABOVE the widest gap, between two
# narrower ones, so the cut goes above them: the topmost blank run of three or
# more scaled rows in the search window.
first = next(r for r in runs if len(r) >= 3)
cut = int(first[0] * 8)
print(f'  topmost qualifying gap at scaled rows {first[0]}-{first[-1]}')
print(f'row gap found at scaled rows {band[0]}-{band[-1]} -> full-size y = {cut} '
      f'({cut / H:.3f} of height)')

top = orig.crop((0, 0, W, cut))
c = Image.open(OUT / 'Figure_5c_rebuilt_4QOK.png').convert('RGB')
d = Image.open(OUT / 'Figure_5d_rebuilt_4QOK.png').convert('RGB')

half = W // 2
bh = max(int(c.height * (half / c.width)), int(d.height * (half / d.width)))
cr = c.resize((half, int(c.height * half / c.width)), Image.LANCZOS)
dr = d.resize((half, int(d.height * half / d.width)), Image.LANCZOS)
bottom = Image.new('RGB', (W, bh), 'white')
bottom.paste(cr, (0, (bh - cr.height) // 2))
bottom.paste(dr, (half, (bh - dr.height) // 2))

pad = int(0.012 * H)
out = Image.new('RGB', (W, cut + pad + bh), 'white')
out.paste(top, (0, 0))
out.paste(bottom, (0, cut + pad))

# panel letters on the new row, matching the weight of a and b
dr2 = ImageDraw.Draw(out)
size = int(H * 0.030)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                              size)
except OSError:
    font = ImageFont.load_default()
# panel c carries its own letter from matplotlib; only d needs one drawn
dr2.text((half + int(0.012 * W), cut + pad + int(0.02 * H)), 'd', fill='black',
         font=font)

out.save(OUT / 'Figure_5_rebuilt_full.png', optimize=True)
print('composed', out.size, '->', OUT / 'Figure_5_rebuilt_full.png')
