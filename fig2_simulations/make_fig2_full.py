"""Assemble the full paper Fig 2: panels A-C (_fig2_orig_assemble.py) above panels D-F (the
scaling grid from fig1_scaling/_make_scaling_figure.py, relabelled D-F). Vector PDF via pypdf.
Run the two figure scripts first. Output: <OUT>/fig2_simulations/fig2.pdf"""
import os, sys, re, subprocess, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shared')))
from paths import out as _out, outdir
from pypdf import PdfReader, PdfWriter, Transformation, PageObject

HERE = os.path.dirname(os.path.abspath(__file__))
top_pdf = os.path.join(outdir('fig2_simulations'), 'fig2_orig.pdf')
# D-F: rerun the scaling figure with panel letters D, E, F (the script labels them a, b, c)
src = open(os.path.join(HERE, '..', 'fig1_scaling', '_make_scaling_figure.py')).read()
for a, b in (("'a   Combined score'", "'D   Combined score'"), ("'b   Runtime scaling'", "'E   Runtime scaling'"),
             ("'c   Sample complexity'", "'F   Sample complexity'")):
    assert a in src, a
    src = src.replace(a, b)
src = src.replace("_out('fig1_scaling', 'scaling_grid_figure.", "_out('fig2_simulations', 'fig2_DEF.")
with tempfile.NamedTemporaryFile('w', suffix='.py', dir=os.path.join(HERE, '..', 'fig1_scaling'), delete=False) as f:
    f.write(src); tmp = f.name
try:
    subprocess.run([sys.executable, tmp], check=True, cwd=os.path.dirname(tmp))
finally:
    os.remove(tmp)
bot_pdf = os.path.join(outdir('fig2_simulations'), 'fig2_DEF.pdf')

top = PdfReader(top_pdf).pages[0]; bot = PdfReader(bot_pdf).pages[0]
tw, th = float(top.mediabox.width), float(top.mediabox.height)
bw, bh = float(bot.mediabox.width), float(bot.mediabox.height)
s = 0.82 * tw / bw; gap = 8                     # D-F scaled so fonts match A-C
page = PageObject.create_blank_page(width=tw, height=th + bh * s + gap)
page.merge_transformed_page(bot, Transformation().scale(s, s).translate(4, 0))
page.merge_transformed_page(top, Transformation().translate(0, bh * s + gap))
w = PdfWriter(); w.add_page(page); w.write(_out('fig2_simulations', 'fig2.pdf'))
print('saved', _out('fig2_simulations', 'fig2.pdf'))
