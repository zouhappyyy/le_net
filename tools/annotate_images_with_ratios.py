#!/usr/bin/env python3
"""Annotate existing per-band comparison images with numeric energy ratios from CSV.
Reads tools/batch_band_energies.csv and finds matching files in paper_figs_trained/,
writes annotated copies into paper_figs_annotated/ with ratio printed on the image.
"""
import os, csv, sys
from PIL import Image, ImageDraw, ImageFont

CSV = 'tools/batch_band_energies.csv'
INPUT_DIR = 'paper_figs_trained'
OUT_DIR = 'paper_figs_annotated'
FONT_PATH = None

os.makedirs(OUT_DIR, exist_ok=True)

# load CSV into dict keyed by case_layer_band
data = {}
with open(CSV, newline='') as cf:
    reader = csv.DictReader(cf)
    for r in reader:
        key = (r['case'], r['layer'], r['band'])
        data[key] = r

# attempt to find a font
try:
    from matplotlib import font_manager
    fp = font_manager.findfont('DejaVuSans')
    FONT_PATH = fp
except Exception:
    FONT_PATH = None

font = None
if FONT_PATH:
    try:
        font = ImageFont.truetype(FONT_PATH, size=24)
    except Exception:
        font = ImageFont.load_default()
else:
    font = ImageFont.load_default()

# list files in input dir
files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
for f in files:
    # expect: fdconv_trained_<case>_<layer>_band<idx>_comparison_*.png
    if not f.startswith('fdconv_trained_'):
        continue
    s = f[len('fdconv_trained_'):]
    # s -> <case>_<layer>_band<idx>_comparison_...
    parts = s.split('_')
    if len(parts) < 4:
        continue
    case = parts[0]
    layer = parts[1]
    band_part = parts[2]
    if not band_part.startswith('band'):
        continue
    band = band_part.replace('band','')
    key = (case, layer, band)
    if key not in data:
        # try removing potential duplicates in case name
        # some case names may include underscores; try merging until layer token found
        alt = s.split('_')
        # find index of 'f0' or 'f1'
        idx_layer = None
        for i,p in enumerate(alt):
            if p in ('f0','f1'):
                idx_layer = i
                break
        if idx_layer is None:
            continue
        case = '_'.join(alt[:idx_layer])
        layer = alt[idx_layer]
        band_part = alt[idx_layer+1]
        if not band_part.startswith('band'):
            continue
        band = band_part.replace('band','')
        key = (case, layer, band)
        if key not in data:
            continue
    ratio = float(data[key]['ratio'])
    raw_e = float(data[key]['raw_energy'])
    w_e = float(data[key]['weighted_energy'])
    # open image and annotate
    inp = os.path.join(INPUT_DIR, f)
    img = Image.open(inp).convert('RGBA')
    txt = Image.new('RGBA', img.size, (255,255,255,0))
    d = ImageDraw.Draw(txt)
    s = f'ratio={ratio:.2f}\nraw={raw_e:.1f}\nweighted={w_e:.1f}'
    # place at top-left with semi-transparent rectangle
    margin = 10
    lines = s.split('\n')
    w,h = img.size
    # compute text size
    tw = 0; th = 0
    for line in lines:
        sz = d.textsize(line, font=font)
        tw = max(tw, sz[0])
        th += sz[1] + 2
    rect_w = tw + 12
    rect_h = th + 12
    box = (margin, margin, margin+rect_w, margin+rect_h)
    # semi-transparent background
    d.rectangle(box, fill=(0,0,0,160))
    # draw text
    y = margin + 6
    for line in lines:
        d.text((margin+6, y), line, font=font, fill=(255,255,255,255))
        y += d.textsize(line, font=font)[1] + 2
    out = Image.alpha_composite(img, txt)
    out_path = os.path.join(OUT_DIR, f)
    out.convert('RGB').save(out_path)
    print('Annotated', f, '->', out_path)

print('Done')
