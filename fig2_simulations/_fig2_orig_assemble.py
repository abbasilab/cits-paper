"""Fig 2 in the original CITS/TPC design:
  Row 1  CS vs noise (one line per method)
  Row 2  grouped IFPR / TPR / CS bars per method at noise=1  (original 3-metric bars)
  Row 3  ground-truth graph, true edge weights overlaid on the arrows
  Row 4  CITS-estimated graph, weights overlaid; non-linear paradigms show
         detection + sign (magnitude undefined), log|.|-coupled edges = sign n/a
"""
import numpy as np, pandas as pd, json, os, matplotlib; matplotlib.use('Agg')
matplotlib.rcParams['font.family']='sans-serif'
matplotlib.rcParams['font.sans-serif']=['Arial','Liberation Sans','Nimbus Sans','DejaVu Sans']  # Arial (metric-identical fallback)
matplotlib.rcParams['pdf.fonttype']=42; matplotlib.rcParams['ps.fonttype']=42                    # embed editable TrueType (no Type 3)
matplotlib.rcParams['mathtext.fontset']='custom'                                                 # route math (eta, alpha, etc.) through the sans font
matplotlib.rcParams['mathtext.rm']='Liberation Sans'; matplotlib.rcParams['mathtext.it']='Liberation Sans:italic'; matplotlib.rcParams['mathtext.bf']='Liberation Sans:bold'
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Patch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
D=os.path.dirname(os.path.abspath(__file__))
frames=[]
for f in ['_fig2_baselines_noise50.csv','_fig2_cits_noise50.csv','_fig2_cits_noise50_nlng.csv','_fig2_kgc_noise50.csv','_fig2_pcmci_noise50.csv']:
    p=os.path.join(D,f)
    if os.path.exists(p): frames.append(pd.read_csv(p))
d=pd.concat(frames,ignore_index=True)
for c in ['TPR','FPR','CS']: d[c]=pd.to_numeric(d[c],errors='coerce')
d['IFPR']=1.0-d['FPR']
esign=json.load(open(os.path.join(D,'_fig2_edgesign.json')))    # unified signed-LSCM weight/sign (single source)
REG=[('lingauss1','Linear Gaussian 1'),('lingauss2','Linear Gaussian 2'),
     ('nonlinnongauss1','Non-linear Non-Gaussian 1'),('nonlinnongauss2','Non-linear Non-Gaussian 2')]
LINEAR={'lingauss1':True,'lingauss2':True,'nonlinnongauss1':False,'nonlinnongauss2':False}
GT_EVEN={'nonlinnongauss2':{(1,3),(2,3)}}   # true couplings with no monotone sign (log|.|)
METH=['CITS','TPC','GC1','GC2','KernelGC','PCMCIplus','LPCMCI','PC']
LAB={'CITS':'CITS','TPC':'TPC','GC1':'GC1','GC2':'GC2','KernelGC':'Kernel GC','PCMCIplus':'PCMCI+','LPCMCI':'LPCMCI','PC':'PC'}
col={'CITS':'#0072B2','TPC':'#009E73','PCMCIplus':'#D55E00','LPCMCI':'#E69F00','KernelGC':'#CC79A7','GC1':'#56B4E9','GC2':'#000000','PC':'#999999'}
mk={'CITS':'o','TPC':'^','GC1':'s','GC2':'D','KernelGC':'P','PCMCIplus':'X','LPCMCI':'*','PC':'.'}
# distinct line styles so same-hue pairs (CITS/GC1 blue, PCMCI+/LPCMCI orange) separate
LS={'CITS':'-','TPC':'-','GC2':'-','GC1':(0,(4,1.5)),'KernelGC':(0,(3,1,1,1)),
    'PCMCIplus':(0,(1,1.2)),'LPCMCI':(0,(4,1,1,1)),'PC':(0,(6,2))}
NOISE=[0.1,0.5,1,1.5,2,2.5,3,3.5]
# true edges (0-indexed) and the ground-truth coupling label per paradigm
TE={'lingauss1':[(0,2),(1,2),(2,3)],'lingauss2':[(0,1),(0,2),(1,3),(2,3)],
    'nonlinnongauss1':[(0,2),(1,2),(2,3)],'nonlinnongauss2':[(0,1),(0,2),(1,3),(2,3)]}
TRUELAB={'lingauss1':{(0,2):'+2',(1,2):'-1',(2,3):'+2'},
         'lingauss2':{(0,1):'+2',(0,2):'+2',(1,3):'+1',(2,3):'+1'},
         'nonlinnongauss1':{(0,2):'+4 sin',(1,2):'-3 sin',(2,3):'+3 sin'},
         'nonlinnongauss2':{(0,1):'+4',(0,2):'+3 sin',(1,3):'8 log|·|',(2,3):'9 log|·|'}}
# convergence paradigms: 1,2 -> 3 -> 4 ; diamond paradigms: 1 -> {2,3} -> 4
POS_CONV={0:(0,1),1:(0,0),2:(1,0.5),3:(2,0.5)}
POS_DIAM={0:(0,0.5),1:(1,1),2:(1,0),3:(2,0.5)}
POSREG={'lingauss1':POS_CONV,'nonlinnongauss1':POS_CONV,'lingauss2':POS_DIAM,'nonlinnongauss2':POS_DIAM}
def keyfor(e): return f"{e[0]+1}->{e[1]+1}"
def agg(m,reg,metric):
    s=d[(d.method==m)&(d.regime==reg)]; g=s.groupby('noise')[metric]
    return np.array([g.mean().get(n,np.nan) for n in NOISE]),np.array([g.sem().get(n,np.nan) for n in NOISE])
SIGNCOL={'+':'#0072B2','-':'#D55E00','0':'#9AA0A6'}      # sign -> edge color
# glyph shown on ground-truth non-linear edges (function form; sign is in color)
GTGLYPH={'nonlinnongauss1':{(0,2):'+ sin',(1,2):'− sin',(2,3):'+ sin'},
         'nonlinnongauss2':{(0,1):'+ lin',(0,2):'+ sin',(1,3):'log|·|',(2,3):'log|·|'}}
DIRWORD={'+':'+ve','-':'−ve','0':'sign n/a'}   # edge detected, sign not identifiable
def _truesign(reg,e):
    s=TRUELAB[reg][e][0]; return s if s in '+-' else '+'
def draw_nodes(ax,POS):                  # neutral nodes so sign-colored edges are unambiguous
    for n,(x,y) in POS.items():
        ax.add_patch(Circle((x,y),0.185,fc='#e9edf1',ec='#333',lw=1.3,zorder=6))
        ax.text(x,y,str(n+1),ha='center',va='center',fontsize=11,fontweight='bold',color='#222',zorder=7)
def draw_motif(ax,reg):                 # topology-only header (plain black arrows, no weights)
    POS=POSREG[reg]
    ax.set_xlim(-0.55,2.55); ax.set_ylim(-0.55,1.55); ax.axis('off'); ax.set_aspect('equal'); ax.set_anchor('W')
    for (i,k) in TE[reg]:
        x1,y1=POS[i]; x2,y2=POS[k]
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,color='#333',
                     lw=1.5,shrinkA=16,shrinkB=16,zorder=2,connectionstyle='arc3,rad=0.05'))
    draw_nodes(ax,POS)
def draw_graph(ax,reg,est=False):
    POS=POSREG[reg]
    ax.set_xlim(-0.55,2.55); ax.set_ylim(-0.55,1.55); ax.axis('off'); ax.set_aspect('equal'); ax.set_anchor('W')
    e=esign[reg]
    cx=np.mean([p[0] for p in POS.values()]); cy=np.mean([p[1] for p in POS.values()])   # centroid
    # width scaling from |true weight| for the two linear paradigms
    if LINEAR[reg]:
        mags=[abs(float(TRUELAB[reg][ed].replace('+',''))) for ed in TE[reg]]
        wmin,wmax=min(mags),max(mags)
    for (i,k) in TE[reg]:
        x1,y1=POS[i]; x2,y2=POS[k]; rec=e[keyfor((i,k))]
        if est:                                    # sign of the CITS (LSCM) edge weight
            sign=rec['sign']
        else:                                      # ground truth
            sign='0' if (i,k) in GT_EVEN.get(reg,set()) else _truesign(reg,(i,k))
        undef=(sign=='0'); ecol=SIGNCOL[sign]
        if LINEAR[reg]:
            m=abs(float(TRUELAB[reg][(i,k)].replace('+','')))
            lw=1.5+1.4*((m-wmin)/(wmax-wmin) if wmax>wmin else 0.5)   # slimmer, less cartoonish
            if not est:
                lab=TRUELAB[reg][(i,k)]
            else:                                  # estimate + 95% CI over simulations
                lab=f"{rec['med']:.2f}\n[{rec['lo']:.2f}, {rec['hi']:.2f}]"
        else:
            lw=2.0
            lab=(GTGLYPH[reg][(i,k)] if not est else DIRWORD[sign])
        ls='-' if not undef else (0,(4,2))
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
                     color=ecol,lw=lw,ls=ls,shrinkA=16,shrinkB=16,zorder=2,
                     connectionstyle='arc3,rad=0.05',capstyle='round'))
        if lab:
            dx,dy=x2-x1,y2-y1; L=(dx*dx+dy*dy)**0.5
            px,py=(-dy/L,dx/L) if L else (0,0)                 # unit perpendicular
            wide=('\n' in lab); d=0.46 if wide else 0.30       # push wide CI labels out farther
            nodes=list(POS.values()); best=None
            # search a few positions along the edge x both sides; maximize clearance from all nodes
            for t in (0.4,0.5,0.6):
                bx,by=x1+t*dx,y1+t*dy
                for s in (1,-1):
                    lx,ly=bx+s*px*d,by+s*py*d
                    sc=min((lx-nx)**2+(ly-ny)**2 for nx,ny in nodes)
                    if best is None or sc>best[0]: best=(sc,lx,ly)
            lx,ly=best[1],best[2]; fs=9.5 if not wide else 8.8
            ax.text(lx,ly,lab,ha='center',va='center',fontsize=fs,color=ecol,zorder=4,
                    fontweight='bold',path_effects=[pe.withStroke(linewidth=2.4,foreground='white')])
    draw_nodes(ax,POS)

# flat gridspec with explicit spacer rows so each inter-row gap is set independently
# rows: motif, [gap], bars, [gap], lines, [gap], ground-truth, [gap], CITS-inferred
fig=plt.figure(figsize=(15.5,12.2))
hr=[0.62, 0.10, 1.0, 0.42, 1.3, 0.38, 0.85, 0.05, 0.85]    # noise-curve row (B) height (trimmed to cut whitespace)
gs=fig.add_gridspec(9,4,height_ratios=hr,hspace=0.0,wspace=0.22,left=0.095,right=0.88,top=0.95,bottom=0.03)
rowmap=[0,2,4,6,8]                                          # motif,bars,lines,GT,CITS
axes=np.empty((5,4),dtype=object)
for i,gr in enumerate(rowmap):
    for c in range(4): axes[i,c]=fig.add_subplot(gs[gr,c])
for j,(reg,title) in enumerate(REG):
    # Row0 ground-truth motif (topology header)
    draw_motif(axes[0,j],reg); axes[0,j].set_title(title,fontsize=11)
    # Row1 IFPR/TPR/CS bars at noise=1
    ax=axes[1,j]; x=np.arange(len(METH)); mcol={'IFPR':'#BBBBBB','TPR':'#777777','CS':'#6A51A3'}; wdt=0.26
    for k,metric in enumerate(['IFPR','TPR','CS']):
        sub=[d[(d.method==m)&(d.regime==reg)&(d.noise==1)][metric] for m in METH]
        vals=[s.mean() for s in sub]; errs=[s.sem() for s in sub]
        ax.bar(x+(k-1)*wdt,vals,wdt,color=mcol[metric],yerr=errs,error_kw=dict(elinewidth=0.7,capsize=1.5,ecolor='#444'))
    ax.set_xticks(x); ax.set_xticklabels([LAB[m] for m in METH],fontsize=6.8,rotation=40,ha='right'); ax.set_ylim(0,1.12)
    ax.get_xticklabels()[0].set_fontweight('bold'); ax.get_xticklabels()[0].set_color('#0072B2')
    ax.spines[['top','right']].set_visible(False)
    if j==0: ax.set_ylabel(r'Metric at noise $\eta=1$, $\alpha=0.05$',fontsize=9)
    # Row2 CS vs noise
    ax=axes[2,j]
    for m in METH:
        mu,se=agg(m,reg,'CS')
        if np.all(np.isnan(mu)): continue
        lw=2.4 if m=='CITS' else (1.0 if m=='PC' else 1.4); z=5 if m=='CITS' else 2
        ax.plot(NOISE,mu,marker=mk[m],ls=LS[m],color=col[m],ms=4.6,lw=lw,zorder=z,mew=0.5,mec='white')
        ax.fill_between(NOISE,mu-se,mu+se,color=col[m],alpha=0.10,lw=0,zorder=z-1)
    ax.set_ylim(-0.32,1.05); ax.set_xlabel(r'Noise $\eta$',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    if j==0: ax.set_ylabel(r'Combined Score ($\alpha=0.05$)',fontsize=9)
    # Row3 ground-truth graph, Row4 estimated graph
    draw_graph(axes[3,j],reg,est=False)
    draw_graph(axes[4,j],reg,est=True)
# panel letters + short titles at fixed figure-x so A/B/C align vertically
fig.canvas.draw()
def panel(ax,letter,title,dy):
    y=ax.get_position().y1+dy
    fig.text(0.012,y,letter,fontsize=16,fontweight='bold',va='baseline',ha='left')
    fig.text(0.036,y,title,fontsize=11,fontweight='bold',va='baseline',ha='left')
panel(axes[0,0],'A','Simulation paradigms and recovery performance',0.028)  # above column titles
panel(axes[2,0],'B','Combined Score across noise levels',0.006)
panel(axes[3,0],'C','Edge weight and sign recovery',0.006)
# sub-row labels within panel D
axes[3,0].text(-0.06,0.5,'Ground-truth\ngraph',fontsize=9.5,va='center',ha='right',transform=axes[3,0].transAxes)
axes[4,0].text(-0.06,0.5,'CITS-inferred\ngraph',fontsize=9.5,va='center',ha='right',transform=axes[4,0].transAxes)
# legends: ordered top->bottom to match the rows they describe; titles in sentence case
tkw=dict(loc='center left',frameon=False,fontsize=8.5,title_fontproperties={'weight':'bold','size':9})
bh=[Patch(color='#BBBBBB',label='1 - FPR (IFPR)'),Patch(color='#777777',label='TPR'),Patch(color='#6A51A3',label='CS')]
fig.legend(handles=bh,bbox_to_anchor=(0.905,0.80),title='Metric',**tkw)
mh=[Line2D([0],[0],color=col[m],marker=mk[m],lw=(2.2 if m=='CITS' else 1.4),ls=LS[m],ms=5,label=(LAB[m]+' (ours)' if m=='CITS' else LAB[m])) for m in METH]
fig.legend(handles=mh,bbox_to_anchor=(0.905,0.55),title='Method',**tkw)
sh=[Line2D([0],[0],color='#0072B2',lw=3,label='Positive'),
    Line2D([0],[0],color='#D55E00',lw=3,label='Negative'),
    Line2D([0],[0],color='#9AA0A6',lw=3,ls=(0,(4,2)),label='Sign undefined')]
fig.legend(handles=sh,bbox_to_anchor=(0.905,0.27),title='Edge sign',**tkw)
for ext in ['png','pdf']: fig.savefig(f'/home/rbiswas1/microns/CITS_manuscript/figures/fig2_orig.{ext}',dpi=200,bbox_inches='tight')
print("saved fig2_orig")
