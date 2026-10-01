import os,sys,pickle,numpy as np,pandas as pd
SP="/tmp/claude-1004/-home-rbiswas1-microns/48b8216b-5c45-4c8f-923d-dc312e0dbb46/scratchpad"
sys.path.insert(0,SP); import stim_baseline_enrichment as SB
def present_lag(XB,L):
    if L==0:
        Z=SB.zscore_rows(XB); n=Z.shape[1]; C=(Z@Z.T)/n
    else:
        past=SB.zscore_rows(XB[:,:-L]); fut=SB.zscore_rows(XB[:,L:]); n=past.shape[1]; C=(past@fut.T)/n
    np.fill_diagonal(C,0.0); return SB._fdr_directed_present(C,n-3)
d=np.load(SB.EM_KEYS_NPZ,allow_pickle=True); em=[tuple(int(x) for x in k) for k in d['field_keys']]
mdf=pickle.load(open(SB.SAVES+'matched_df_v1718.pkl','rb')).dropna(subset=['pt_root_id_v1718']).copy()
mdf['pt_root_id_v1718']=mdf['pt_root_id_v1718'].astype(np.int64)
syn=pickle.load(open(SB.SAVES+'synapses_matcheddf_frompre_v1718.pkl','rb'))
scp=set()
for pre,df in syn.items():
    if hasattr(df,'columns') and 'post_pt_root_id' in df.columns:
        for post in df['post_pt_root_id'].astype(np.int64).values: scp.add((int(pre),int(post)))
rows=[]
for (s,sc,field) in em:
    avail=[st for st in SB.STIMS if os.path.exists(f'{SB.STAGE_FC}/s{s}sc{sc}f{field}_{st}.npz')]
    if not avail: continue
    XB0,nids=SB.build_XB(s,sc,field,avail[0])
    if XB0 is None: continue
    p=len(nids); Xs=[SB.build_XB(s,sc,field,st)[0] for st in avail]; Xs=[x for x in Xs if x is not None and x.shape[0]==p]
    if not Xs: continue
    id2i={u:i for i,u in enumerate(nids)}
    fu=mdf[(mdf.session==s)&(mdf.scan_idx==sc)&(mdf.field==field)]
    i2pt={id2i[int(u)]:int(pt) for u,pt in zip(fu.unit_id.astype(np.int64),fu.pt_root_id_v1718.astype(np.int64)) if int(u) in id2i}
    m=sorted(i2pt)
    if not m: continue
    pres=np.zeros((p,p),bool)
    for XB in Xs: pres|=present_lag(XB,0)|present_lag(XB,1)
    a=b=c=e=0
    for i in m:
        for j in m:
            if i==j: continue
            sij=(i2pt[i],i2pt[j]) in scp
            if pres[i,j]: a+=1; b+=int(sij)
            else: c+=1; e+=int(sij)
    rows.append({'field_key':str((s,sc,field)),'fc_plus':a,'fcplus_scplus':b,'fc_minus':c,'fcminus_scplus':e})
corr=pd.DataFrame(rows)
corr.to_csv(SP+'/stim_baseline_perfield_lagged01.csv',index=False)
print("saved CSV rows=",len(corr),flush=True)
cits=pd.read_csv(SP+'/panelA_perfield_counts.csv')
cits['fk']=cits['field_key'].apply(lambda x:tuple(int(v) for v in str(x).strip('()').split(',')))
corr['fk']=corr['field_key'].apply(lambda x:tuple(int(v) for v in str(x).strip('()').split(',')))
M=cits.merge(corr,on='fk',suffixes=('_cits','_corr'))
def fold(df,suf):
    fcp=df['fc_plus'+suf].sum(); fps=df['fcplus_scplus'+suf].sum(); fca=df['fc_minus'+suf].sum(); fas=df['fcminus_scplus'+suf].sum()
    return (fps/fcp)/(fas/fca)
cf=fold(M,'_corr'); ci=fold(M,'_cits'); dens=M['fc_plus_corr'].sum()/(M['fc_plus_corr'].sum()+M['fc_minus_corr'].sum())
rng=np.random.default_rng(42); N=5000; n=len(M); Mi=M.reset_index(drop=True); bc=[];bd=[]
for _ in range(N):
    idx=rng.integers(0,n,n); mb=Mi.iloc[idx]; c_=fold(mb,'_corr'); i_=fold(mb,'_cits'); bc.append(c_); bd.append(i_-c_)
bc=np.array(bc); bd=np.array(bd)
clo,chi=np.percentile(bc,[2.5,97.5]); dlo,dhi=np.percentile(bd,[2.5,97.5]); pval=max(2*min((bd<=0).mean(),(bd>=0).mean()),1/N)
print(f"n_fields={n}",flush=True)
print(f"CORR lag0Ulag1 fold={cf:.3f} 95%CI [{clo:.2f},{chi:.2f}] density={dens*100:.1f}%",flush=True)
print(f"CITS fold={ci:.3f}",flush=True)
print(f"paired Delta(CITS-corr)={ci-cf:.3f} 95%CI [{dlo:.2f},{dhi:.2f}] p={pval:.4f}",flush=True)
print("STATSDONE",flush=True)
