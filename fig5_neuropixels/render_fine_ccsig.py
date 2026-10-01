#!/usr/bin/env python3
# Finer brain-region strips (within-area gradient) + detailed legend (names + counts).
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.cm as cm, matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
TEAL=mcolors.LinearSegmentedColormap.from_list('teal',['#D6EFEA','#5CB8A8','#0E6B5B'])
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','Liberation Sans','Nimbus Sans','DejaVu Sans']  # Arial (Liberation Sans = Arial-metric substitute)
OUT='/home/rbiswas1/microns/CITS_manuscript/figures'
METHODS=['CORR','GC1','GC2','CITS']; TITLES=['Pearson Corr.','GC1','GC2','CITS']
STIMS=['natural_scenes','static_gratings','gabors']
SLAB={'natural_scenes':'Natural Scenes','static_gratings':'Static Gratings','gabors':'Gabors'}
N=68; PCTL=90; CIRCLE_S=15; STRIP=3.0; GAP=1.6
lo=['VISp','VISl','VISrl','VISal','VISpm','VISam','CA1','CA2','CA3','DG','SUB','POL','LGv','LP']
NAMES={'VISp':'Primary Visual Cortex','VISl':'Lateromedial','VISrl':'Rostrolateral','VISal':'Anterolateral','VISpm':'Posteromedial','VISam':'Anteromedial','CA1':'Cornu Ammonis 1','CA2':'Cornu Ammonis 2','CA3':'Cornu Ammonis 3','DG':'Dentate Gyrus','SUB':'Subiculum','POL':'Prosubiculum','LGv':'Lateral Geniculate Nuc.','LP':'Lateral Posterior Nuc.'}
# colorblind-safe: orange / blue / purple sequential families (distinct under deuteran/protan/tritan)
GROUPS=[('Visual Cortex',['VISp','VISl','VISrl','VISal','VISpm','VISam'],cm.Oranges,(0.28,0.90)),
        ('Hippocampal Formation',['CA1','CA2','CA3','DG','SUB','POL'],TEAL,(0.18,0.95)),
        ('Thalamus',['LGv','LP'],cm.Purples,(0.55,0.88))]
# fine color per region
RCOL={}
for gname,regs,cmap,(a,b) in GROUPS:
    for k,r in enumerate(regs):
        RCOL[r]=cmap(a+(b-a)*(k/max(1,len(regs)-1)))
ul=np.load(f'{OUT}/cits_v2_union_labels.npy',allow_pickle=True)
perm=[]
for l in lo: perm+=list(np.where(ul==l)[0])
perm=np.array(perm); labels_pos=ul[perm]
COUNT={r:int((ul==r).sum()) for r in lo}
# contiguous fine-region runs in frame order
runs=[]; s=0
for i in range(1,N+1):
    if i==N or labels_pos[i]!=labels_pos[s]:
        runs.append((s,i-1,labels_pos[s])); s=i
def scale(U):
    nz=np.abs(U[U!=0]); return (float(np.percentile(nz,PCTL)) or float(nz.max())) if nz.size else 1.0
def draw(ax,U):
    M=scale(U); r,c=np.nonzero(U); v=U[r,c]
    ax.scatter(c,r,c=v,cmap='RdBu_r',vmin=-M,vmax=M,s=CIRCLE_S,linewidths=0,alpha=0.9,zorder=2)
    off=0.5+GAP
    for a,b,reg in runs:
        col=RCOL[reg]
        ax.add_patch(Rectangle((a-0.5,-STRIP-off),b-a+1,STRIP,color=col,lw=0,clip_on=False,zorder=3))
        ax.add_patch(Rectangle((-STRIP-off,a-0.5),STRIP,b-a+1,color=col,lw=0,clip_on=False,zorder=3))
    ax.set_xlim(-STRIP-off,N-0.5); ax.set_ylim(N-0.5,-STRIP-off)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
GML={'Visual Cortex':'Visual\nCortex','Hippocampal Formation':'Hippocampal\nFormation','Thalamus':'Thalamus'}
def draw_legend(ax):
    ax.axis('off'); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.text(0.02,0.985,'Brain Regions\n(# active neurons)',ha='left',va='top',fontsize=22,fontweight='bold')
    y=0.86; dy=0.052; sw=0.11; xsw=0.30
    for gname,regs,cmap,rng in GROUPS:
        centers=[]
        for r in regs:
            ax.add_patch(Rectangle((xsw,y-0.04),sw,0.04,color=RCOL[r],lw=0))
            ax.text(xsw+sw+0.03,y-0.02,f'{NAMES[r]} ({COUNT[r]})',ha='left',va='center',fontsize=20)
            centers.append(y-0.02); y-=dy
        c0,c1=centers[0],centers[-1]
        ax.plot([xsw-0.035,xsw-0.035],[c0+0.024,c1-0.024],color='k',lw=1.4)
        ax.text(xsw-0.16,(c0+c1)/2,GML[gname],rotation=90,ha='center',va='center',
                fontsize=20,fontweight='bold',linespacing=0.85,clip_on=False)
        y-=0.02

# ---- montage with embedded legend (right column)
from matplotlib.gridspec import GridSpec
fig=plt.figure(figsize=(18,9))
gs=GridSpec(3,5,figure=fig,width_ratios=[1,1,1,1,1.7],wspace=0.12,hspace=0.06,
            left=0.12,right=0.90,top=0.95,bottom=0.07)
left_ax={}; corner={}
for ri,s in enumerate(STIMS):
    for ci,m in enumerate(METHODS):
        ax=fig.add_subplot(gs[ri,ci]); U=np.nan_to_num(np.load(f'{OUT}/cmp_ccsig_{s}_{m}_68.npy')); draw(ax,U)
        if ri==0: ax.set_title(TITLES[ci],fontsize=24,fontweight='bold',pad=10)
        if ci==0: left_ax[ri]=ax
        if ri==2 and ci==0: corner['bl']=ax
        if ri==2 and ci==3: corner['br']=ax
axL=fig.add_subplot(gs[:,4]); draw_legend(axL)
# y labels: "Neurons" = inner axis label (close to matrices); stimulus names = outer row titles
for ri,s in enumerate(STIMS):
    pos=left_ax[ri].get_position(); yc=(pos.y0+pos.y1)/2
    fig.text(pos.x0-0.020, yc, 'Neurons', rotation=90, ha='center', va='center', fontsize=18)          # inner (axis)
    fig.text(pos.x0-0.055, yc, SLAB[s], rotation=90, ha='center', va='center', fontsize=23, fontweight='bold')  # outer (row title)
# x label: "Neurons" centered under the 4 matrix columns
bl=corner['bl'].get_position(); br=corner['br'].get_position()
fig.text((bl.x0+br.x1)/2, bl.y0-0.045, 'Neurons', ha='center', va='top', fontsize=18)
fig.savefig(f'{OUT}/cmp_montage_4col_ccsig.png',dpi=170,bbox_inches='tight'); print('saved cmp_montage_4col_ccsig.png')
plt.close(fig)
# ---- detailed legend (like the original): grouped swatches + names + counts
figL=plt.figure(figsize=(5.2,7.8)); axL=figL.add_axes([0,0,1,1]); axL.axis('off')
axL.set_xlim(0,1); axL.set_ylim(0,1)
axL.text(0.5,0.985,'Brain Regions (# active neurons)',ha='center',va='top',fontsize=22,fontweight='bold')
y=0.93; dy=0.058; sw=0.11
for gname,regs,cmap,rng in GROUPS:
    ytop=y
    for r in regs:
        axL.add_patch(Rectangle((0.30,y-0.045),sw,0.045,color=RCOL[r],lw=0))
        axL.text(0.44,y-0.022,f'{NAMES[r]} ({COUNT[r]})',ha='left',va='center',fontsize=20)
        y-=dy
    # broad-area bracket label on the left
    axL.plot([0.26,0.26],[ytop-0.045,y+dy-0.045],color='k',lw=1.2)
    axL.text(0.22,(ytop+y+dy)/2-0.045,gname,rotation=90,ha='center',va='center',fontsize=22,fontweight='bold')
    y-=0.02
figL.savefig(f'{OUT}/region_legend_fine.png',dpi=200,bbox_inches='tight'); print('saved region_legend_fine.png')
