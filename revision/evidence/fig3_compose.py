"""
Append the submitted panel i (melanoma UMAP) under the eight rebuilt panels, so
Figure 3 keeps all nine and the caption's statement about i has something to
point at. Panel i is cropped from the submitted image, not regenerated.
"""
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
BASE = Path('/home/muhammad-imran/PhD_data/TCR_Module_1')
OUT = BASE / 'revision/evidence/figures'

z = zipfile.ZipFile(BASE / 'revision/working/Manuscript_fv_1_REVISED.docx')
orig = Image.open(io.BytesIO(z.read('word/media/image3.png'))).convert('RGB')
W, H = orig.size
print('submitted Fig 3:', orig.size)
# 3 x 3 grid; i is the bottom-right cell
pi = orig.crop((int(W * 2 / 3), int(H * 2 / 3), W, H))
print('panel i crop:', pi.size)

new = Image.open(OUT / 'Figure_3_rebuilt_v2.png').convert('RGB')
NW, NH = new.size
target_w = NW // 4
pir = pi.resize((target_w, int(pi.height * target_w / pi.width)), Image.LANCZOS)

pad = int(0.02 * NH)
out = Image.new('RGB', (NW, NH + pad + pir.height), 'white')
out.paste(new, (0, 0))
out.paste(pir, (0, NH + pad))

d = ImageDraw.Draw(out)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                              int(NH * 0.035))
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                               int(NH * 0.018))
except OSError:
    font = small = ImageFont.load_default()
d.text((int(0.004 * NW), NH + pad), 'i', fill='black', font=font)
d.text((target_w + int(0.02 * NW), NH + pad + int(0.02 * NH)),
       'Panel i is the melanoma UMAP from the originally submitted model;\n'
       'it was not regenerated under the corrected pipeline.',
       fill='#6B7280', font=small)

out.save(OUT / 'Figure_3_rebuilt_full.png', optimize=True)
print('composed', out.size, '->', OUT / 'Figure_3_rebuilt_full.png')
