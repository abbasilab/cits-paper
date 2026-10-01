import numpy as np, sys, os, warnings, csv; warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from glm_spiking_sim import simulate_glm_spiking, directed_cs
from _glm_suite_baselines import pcmci_plus, lpcmci
w=csv.writer(open('_sd_tigra.csv','w',newline='')); w.writerow(['cell','seed','cs'])
SPK={'convergence':'conv','diamond':'cce','depression':'depr'}
for m,tag in SPK.items():
    for s in range(50):
        X,GT,_=simulate_glm_spiking(s,motif=m)
        try: pp=pcmci_plus(X,'parcorr'); w.writerow([f'PCMCIplus_spk_{tag}',s,directed_cs((np.asarray(pp)!=0).astype(int),GT)[0]])
        except Exception as e: w.writerow([f'PCMCIplus_spk_{tag}',s,''])
        try: lp=lpcmci(X); w.writerow([f'LPCMCI_spk_{tag}',s,directed_cs((np.asarray(lp)!=0).astype(int),GT)[0]])
        except Exception as e: w.writerow([f'LPCMCI_spk_{tag}',s,''])
print("tigra sd done",flush=True)
