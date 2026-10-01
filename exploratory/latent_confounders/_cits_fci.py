"""CITS-FCI (exploratory, NOT for paper): replace CITS's PC backbone with an FCI
backbone (relaxes causal sufficiency -> can emit bidirected <-> for latent common
causes). Temporal lag-embedding + FCI (causal-learn) + temporal background knowledge,
mapped back to a neuron-level rolled graph.

CI test: 'fisherz' (linear, fast, for validation/linear regime) or 'kci' (kernel
nonparametric, the CITS-family test type, for nonlinear).

Returns dict: adj (directed rolled), bidir (neuron-pair latent<-> flag), classify().
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint

def lag_embed(X, tau):
    """X (p,T) -> design (N, p*(tau+1)); col index c = v*(tau+1)+l is X_v(t-l)."""
    p, T = X.shape
    N = T - tau
    cols = []
    for v in range(p):
        for l in range(tau+1):
            cols.append(X[v, tau-l : T-l])   # length N; l=0 is present
    return np.column_stack(cols)             # (N, p*(tau+1))

def _cidx(v, l, tau): return v*(tau+1)+l

def run_cits_fci(X, tau=2, alpha=0.05, ci='fisherz', kci_kwargs=None):
    p, T = X.shape
    D = lag_embed(X, tau)
    m = p*(tau+1)
    nodes = [GraphNode(f"v{v}_l{l}") for v in range(p) for l in range(tau+1)]
    # temporal background knowledge: forbid future->past (l_from < l_to => can't cause)
    bk = BackgroundKnowledge()
    for v in range(p):
        for l in range(tau+1):
            for v2 in range(p):
                for l2 in range(tau+1):
                    if v==v2 and l==l2: continue
                    # relative real time of node = -l ; arrow node_from -> node_to needs time_from <= time_to
                    # forbid if source is LATER in real time than target: (-l_from) > (-l_to) => l_from < l_to
                    if l < l2:
                        bk.add_forbidden_by_node(nodes[_cidx(v,l,tau)], nodes[_cidx(v2,l2,tau)])
                    # lag-only (match CITS): forbid contemporaneous cross-neuron edges both ways
                    if l==0 and l2==0 and v!=v2:
                        bk.add_forbidden_by_node(nodes[_cidx(v,l,tau)], nodes[_cidx(v2,l2,tau)])
    kwargs = kci_kwargs or {}
    g, edges = fci(D, independence_test_method=ci, alpha=alpha, background_knowledge=bk,
                   show_progress=False, node_names=[n.get_name() for n in nodes], **kwargs)
    # map edges to neuron-pair relationships
    adj = np.zeros((p,p),int)          # directed a->b
    bidir = np.zeros((p,p),bool)       # latent <-> between neurons a,b
    circ = np.zeros((p,p),bool)        # ambiguous o-marks
    def parse(name):
        s=name[1:].split('_l'); return int(s[0]), int(s[1])
    for e in edges:
        n1=e.get_node1().get_name(); n2=e.get_node2().get_name()
        v1,l1=parse(n1); v2,l2=parse(n2)
        if v1==v2: continue
        ep1=e.get_endpoint1(); ep2=e.get_endpoint2()   # endpoints at n1, n2
        A=Endpoint.ARROW; Ta=Endpoint.TAIL; Ci=Endpoint.CIRCLE
        if ep1==A and ep2==A:
            bidir[v1,v2]=bidir[v2,v1]=True
        elif ep2==A and ep1==Ta:   # n1 --> n2  (v1 -> v2)
            adj[v1,v2]=1
        elif ep1==A and ep2==Ta:   # n2 --> n1
            adj[v2,v1]=1
        else:                       # any circle mark => ambiguous
            circ[v1,v2]=circ[v2,v1]=True
    return dict(adj=adj, bidir=bidir, circ=circ)

def classify_pair(res, a, b):
    if res['bidir'][a,b]: return 'bidirected'
    if res['adj'][a,b] or res['adj'][b,a]: return 'directed'
    if res['circ'][a,b]: return 'ambiguous'
    return 'none'

# ---------------- validation ----------------
if __name__=='__main__':
    import sys
    T=1500; p=3
    def sim_latent(seed, phi=0.7):
        # H (AR) -> X0, X1 (lag1, symmetric); NO direct X0-X1; X2 independent
        rng=np.random.default_rng(seed); H=np.zeros(T)
        for t in range(1,T): H[t]=phi*H[t-1]+rng.standard_normal()
        X=rng.standard_normal((p,T))*0.5
        for t in range(1,T):
            X[0,t]=H[t-1]+0.4*rng.standard_normal()
            X[1,t]=H[t-1]+0.4*rng.standard_normal()
        return X
    print("VALIDATION: latent H->X0,X1 (no direct edge). Expect X0-X1 = bidirected or ambiguous, NOT directed.",flush=True)
    from collections import Counter
    c=Counter()
    for s in range(10):
        X=sim_latent(s)
        res=run_cits_fci(X, tau=2, alpha=0.05, ci='fisherz')
        c[classify_pair(res,0,1)]+=1
    print("X0-X1 classification over 10 seeds:", dict(c), flush=True)
    print("VALIDATION DONE",flush=True)
