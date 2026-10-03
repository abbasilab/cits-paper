#%%
def convert_fcmat2units(mat,stim_label,units,permute):
    import pickle as pkl
    import numpy as np
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'shared')))
    from paths import outdir as _outdir
    data_directory = os.path.join(_outdir('fig5_neuropixels/notebooks/data'), '')  # was 'data/' (data folder of script.ipynb) #'D:\\OneDrive - UW\\research\\projects_git\\mice-aibs\\data'#'D:\\OneDrive - UW\\research\\projects_git\\mice-aibs\\data'
    sess_id = 791319847
    class A_Config:
        def __init__(self, save_dir, sess_id, stimulus):
            
            #self.data_dir = data_dir
            self.save_dir = save_dir
            self.sess_id = sess_id
            self.stimulus = stimulus

            self.name = "{}ID{}_{}".format(self.save_dir, self.sess_id, self.stimulus)
    allencon = A_Config(data_directory,sess_id,stim_label)
    units_to_use_idx_ = np.where(pkl.load(open(allencon.name+'_units2use_stim_{}.p'.format(stim_label),'rb')))[0]
    n_units = len(units)
    matunits = np.zeros((n_units,n_units))
    for i in range(n_units):
        for j in range(n_units):
            if units[i] in units_to_use_idx_ and units[j] in units_to_use_idx_:
                idxi = np.where(units_to_use_idx_==units[i])
                idxj = np.where(units_to_use_idx_==units[j])
                matunits[i,j] = mat[idxi,idxj]
    matunits2 = matunits[np.ix_(permute,permute)]
    return matunits2
#%%
def get_graph_summary(mat):
    import numpy as np
    import networkx as nx
    nnzidx0 = np.nonzero(mat)
    nnz = list(zip(nnzidx0[0],nnzidx0[1]))#,mat[mat!=0]))
    G = nx.DiGraph()
    G.add_nodes_from(list(range(mat.shape[0])))
    G.add_edges_from(nnz)
    #G.add_weighted_edges_from(nnz)
    #G = nx.algorithms.dag.transitive_closure(G)
    metrics = [nx.algorithms.centrality.betweenness_centrality, nx.algorithms.cluster.transitivity, nx.algorithms.assortativity.degree_pearson_correlation_coefficient, nx.algorithms.cluster.average_clustering, nx.algorithms.efficiency_measures.global_efficiency,nx.algorithms.efficiency_measures.local_efficiency]# nx.algorithms.smallworld.sigma, nx.algorithms.cluster.average_clustering]
    out = []
    for i in range(4):
        out.append(metrics[i](G))
    for i in range(4,6):   
        out.append(metrics[i](G.to_undirected()))
    out[0]= np.mean(list(out[0].values()))
    return out
#%%
#%%
# stim_label_list = ['natural_scenes','static_gratings','gabors','flashes']
# units_to_use_idx_={}
# for stim_label in stim_label_list:
#     allencon = A_Config(data_directory,sess_id,stim_label)
#     units_to_use_idx_[stim_label] = np.where(pkl.load(open(allencon.name+'_units2use_stim_{}.p'.format(stim_label),'rb')))[0]

# setnow = [set(unit) for unit in units_to_use_idx_.values()]
# setnow = set.union(*setnow)
# setnow = sorted(list(setnow))
# units = setnow
# #%%
# method_name = 'neuropc'
# bin_size = 0.01
# niter = 50
# thresh_list = [0.75,0.70,0.65,0.60,0.55,0.50,0.45,0.40,0.35,0.30,0.25,0.20,0.15,0.10,0.05,0.01]
# save_directory = 'D:\\OneDrive - UW\\research\\projects_git\\mice-aibs\\save\\'+method_name
# ce_store={}
# #%%
# cefm={}
# for idx0 in range(4):
#     stim_label=stim_label_list[idx0]
#     cefm[stim_label]=[]
#     alpha = 0.2
#     for thresh in [0.01]:
#         if idx0==0:
#             idxlist=set(list(range(60)))-{5,7}#[45, 47, 49, 51, 52, 54, 55, 57, 59]
#         elif idx0==2:
#             idxlist=[0,2,3,4,5,6,7,8,9]
#         elif idx0==3:
#             idxlist=[0,1,2]
#         else:
#             idxlist=[10,11,12,13,14,15,16,17,18,19]
#         #for idx in set(list(range(60)))-{5,97}:
#         allencon_save = A_Config(save_directory,sess_id,stim_label)
#         c = 0
#         for file in os.listdir(save_directory):
#             filestart = 'ID{}_{}'.format(sess_id,stim_label) + '_bin{}_alpha{}_thresh{}_idx'.format(bin_size,alpha,thresh)
#             if file.startswith(filestart):
#                 result = pkl.load(open(save_directory+'\\'+file,'rb'))
#                 ce = result['ce']
#                 adj = result['adj']
#                 c=c+1
#                 #ce[adj==0]=0
#                 adj = adj - np.diag(np.diag(adj))
#                 #ce[ce<(np.max(ce)/10)]=0
#                 #ce[np.isnan(ce)]=0
#                 #ce = ce - np.diag(np.diag(ce))
#                 cefm[stim_label].append(convert_fcmat2units(adj,stim_label,units,permute))
#         print(c)
#     #plot_matrix(cefm,0.5,str(idx),allencon_save,'FM_'+'thresh'+str(thresh)+'_alpha'+str(alpha),method_name)

# # %%
# idx0=0
# idx=0
# stim_label=stim_label_list[idx0]
# allencon_save = A_Config(save_directory,sess_id,stim_label)
# plot_matrix(cefm[stim_label][0],0.5,str(idx),allencon_save,'temp_FM_'+'thresh'+str(thresh)+'_alpha'+str(alpha),method_name)
# # %%
# colnames=['Stimulus', 'Avg. Betweenness Centrality', 'Transitivity', 'Assortativity', 'Avg. Clustering Coefficient', 'Global Efficiency', 'Local Efficiency']
# graphssum=np.zeros((sum([len(cefm[stim_label]) for stim_label in stim_label_list]),7))
# graphssum = pd.DataFrame(data=graphssum,columns=colnames)
# idx = 0
# stim_label_list1 = ['Natural Scenes', 'Static Gratings', 'Gabors', 'Flashes']
# for i,stim_label in enumerate(stim_label_list):
#     for j in range(len(cefm[stim_label])):
#         graphssum.iloc[idx] = [stim_label_list1[i]]+get_graph_summary(cefm[stim_label][j])
#         idx = idx + 1
# # %%
# import seaborn as sns
# method = 'tpc'
# for i in range(1,7):
#     fig, ax = plt.subplots(1,1,figsize=(8,5))
#     ax = sns.boxplot(x=colnames[0],y=colnames[i],data=graphssum, linewidth= 2, palette = 'Set2')
#     ax.tick_params(axis='y', which = 'major', labelsize = 25)
#     ax.set_ylabel(colnames[i],fontsize=12)
#     fig.savefig(save_directory+ '\\ID'+str(sess_id) + '_'+method+'_'+colnames[i]+'.png',format='png',dpi=600)
# # %%
