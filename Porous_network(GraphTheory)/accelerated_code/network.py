import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
from input import Config
from solver import PanelMethod

# ==============================================================================
# POROUS NETWORK SYSTEM
# ==============================================================================
class PorousNetwork:
    def __init__(self, aero: PanelMethod, cp_solid, config: Config):
        self.aero = aero
        self.cfg = config
        self.G = nx.Graph()
        self.active_pores = []
        
        self._build_network(cp_solid)

    def _build_network(self, cp_solid):
        xc, yc = self.aero.XC, self.aero.YC
        self.G = nx.Graph()
        
        spar_id = 99999
        spar_pos = np.array([0.25, 0.0]) 
        self.G.add_node(spar_id, pos=spar_pos, type='internal')
        
        inlet_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
        outlet_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.05 <= xc[i] <= 0.20]
        
        inlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in inlet_candidates]
        inlet_scores.sort(key=lambda x: x['cp'], reverse=True)
        selected_inlets = [x['id'] for x in inlet_scores[:self.cfg.N_INLETS]]
        
        outlet_scores = [{'id': i, 'cp': cp_solid[i]} for i in outlet_candidates]
        outlet_scores.sort(key=lambda x: x['cp'])
        
        num_outlets_to_use = min(self.cfg.N_OUTLETS, len(outlet_scores))
        selected_outlets = [x['id'] for x in outlet_scores[:num_outlets_to_use]]
        
        self.active_pores = selected_inlets + selected_outlets

        print(f"   -> Generating Passive Network: {len(selected_inlets)} Pores Top-TE -> {len(selected_outlets)} Pores Bottom-LE.")
          
        for pid in self.active_pores:
            p_pos = np.array([xc[pid], yc[pid]])
            dist = np.linalg.norm(p_pos - spar_pos)
            
            if pid not in self.G:
                self.G.add_node(pid, pos=(xc[pid], yc[pid]), type='boundary', panel_idx=pid)
            
            if pid in selected_inlets:
                r = self.cfg.PORE_RADIUS_INLET
                etype = 'plenum_in'
            else:
                r = self.cfg.PORE_RADIUS_OUTLET 
                etype = 'plenum_out'
                
            cond = (np.pi * r**4) / (8 * self.cfg.MU * dist)
            self.G.add_edge(pid, spar_id, length=dist, cond=cond, type=etype)
            
        self._prepare_network_solver()

    def _prepare_network_solver(self):
        """Assembles the sparse conductance matrix once and precomputes the factorization."""
        self.nodes = list(self.G.nodes())
        self.n_nodes = len(self.nodes)
        self.node_map = {node: i for i, node in enumerate(self.nodes)}
        self.boundary_nodes = [n for n in self.G.nodes() if self.G.nodes[n]['type'] == 'boundary']

        A = sp.lil_matrix((self.n_nodes, self.n_nodes))

        for node in self.nodes:
            idx = self.node_map[node]
            if node in self.boundary_nodes:
                A[idx, idx] = 1.0  
            else:
                sigma_cond = 0.0
                for nbr in self.G.neighbors(node):
                    c = self.G[node][nbr]['cond']
                    nbr_idx = self.node_map[nbr]
                    A[idx, nbr_idx] = -c
                    sigma_cond += c
                A[idx, idx] = sigma_cond if sigma_cond > 0 else 1.0 

        self.net_solver = scipy.sparse.linalg.factorized(A.tocsc())

    def solve_flow(self, P_boundary):
        """Solves the network using the pre-factorized matrix."""
        b = np.zeros(self.n_nodes)

        for node in self.boundary_nodes:
            idx = self.node_map[node]
            pid = self.G.nodes[node]['panel_idx']
            if pid in P_boundary:
                b[idx] = P_boundary[pid]
            else:
                b[idx] = 0.0

        try:
            P_nodes = self.net_solver(b)
        except Exception as e:
            print(f"   [!] Matrix solve failed: {e}")
            return {}, np.zeros(self.n_nodes)

        velocities = {}
        for node in self.boundary_nodes:
            pid = self.G.nodes[node]['panel_idx']
            idx = self.node_map[node]
            
            is_inlet = any(self.G[node][nbr]['type'] == 'plenum_in' for nbr in self.G.neighbors(node))
            radius = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * radius**2

            Q_net = 0.0
            for nbr in self.G.neighbors(node):
                c = self.G[node][nbr]['cond']
                nbr_idx = self.node_map[nbr]
                Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])
                
            if area > 0:
                velocities[pid] = -Q_net / area
            else:
                velocities[pid] = 0.0

        return velocities, P_nodes