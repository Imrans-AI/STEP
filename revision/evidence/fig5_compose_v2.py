"""Compose Figure 5 from four rebuilt panels: a and c attention maps, b and d PyMOL."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None
OUT = Path('/home/muhammad-imran/PhD_data/TCR_Module_1/revision/evidence/figures')
panels = {'a': OUT / 'Figure_5a_rebuilt_3GSN.png',
          'b': OUT / 'Figure_5b_rebuilt_3GSN.png',
          'c': OUT / 'Figure_5c_rebuilt_4QOK.png',
          'd': OUT / 'Figure_5d_rebuilt_4QOK.png'}
ims = {k: Image.open(v).convert('RGB') for k, v in panels.items()}
CELL_W = 3000
def fit(im):
    return im.resize((CELL_W, int(im.height * CELL_W / im.width)), Image.LANCZOS)
f = {k: fit(v) for k, v in ims.items()}
row1 = max(f['a'].height, f['b'].height)
row2 = max(f['c'].height, f['d'].height)
pad = 90
W = CELL_W * 2 + pad
H = row1 + row2 + pad
out = Image.new('RGB', (W, H), 'white')
out.paste(f['a'], (0, (row1 - f['a'].height) // 2))
out.paste(f['b'], (CELL_W + pad, (row1 - f['b'].height) // 2))
out.paste(f['c'], (0, row1 + pad + (row2 - f['c'].height) // 2))
out.paste(f['d'], (CELL_W + pad, row1 + pad + (row2 - f['d'].height) // 2))
d = ImageDraw.Draw(out)
try:
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 150)
except OSError:
    font = ImageFont.load_default()
# a and c carry their own letters from matplotlib; b and d need them drawn
d.text((CELL_W + pad + 40, 40), 'b', fill='black', font=font)
d.text((CELL_W + pad + 40, row1 + pad + 40), 'd', fill='black', font=font)
out.save(OUT / 'Figure_5_rebuilt_full.png', optimize=True)
print('composed', out.size)
