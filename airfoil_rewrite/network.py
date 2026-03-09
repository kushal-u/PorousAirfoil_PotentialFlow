# network.py
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
from input import Config
from solver import PanelMethod


class PorousNetwork:
    def __init__(self, aero: PanelMethod, cp_solid, config: Config, auto_build=True):
        self.aero = aero
        self.cfg = config
        self.G = nx.Graph()
        self.active_pores = []

        if auto_build and cp_solid is not None:
            self._build_network(cp_solid)

    def _build_network(self, cp_solid):
        """
        Default one-chamber passive example network.
        """
        xc, yc = self.aero.XC, self.aero.YC
        self.G = nx.Graph()

        spar_id = 99999
        spar_pos = np.array([0.25, 0.0])
        self.G.add_node(spar_id, pos=spar_pos, type='internal')

        top_candidates = [i for i in range(len(xc)) if yc[i] > 0 and xc[i] >= 0.85]
        bot_candidates = [i for i in range(len(xc)) if yc[i] < 0 and 0.05 <= xc[i] <= 0.20]

        top_scores = [{'id': i, 'cp': cp_solid[i]} for i in top_candidates]
        top_scores.sort(key=lambda x: x['cp'], reverse=True)
        selected_top = [x['id'] for x in top_scores[:self.cfg.N_INLETS]]

        bot_scores = [{'id': i, 'cp': cp_solid[i]} for i in bot_candidates]
        bot_scores.sort(key=lambda x: x['cp'])
        selected_bot = [x['id'] for x in bot_scores[:self.cfg.N_OUTLETS]]

        self.active_pores = selected_top + selected_bot

        for pid in self.active_pores:
            p_pos = np.array([xc[pid], yc[pid]])
            dist = np.linalg.norm(p_pos - spar_pos)

            if pid not in self.G:
                self.G.add_node(pid, pos=(xc[pid], yc[pid]), type='boundary', panel_idx=pid)

            r = self.cfg.PORE_RADIUS_INLET if yc[pid] > 0 else self.cfg.PORE_RADIUS_OUTLET
            cond = (np.pi * r**4) / (8.0 * self.cfg.MU * dist)

            self.G.add_edge(
                pid, spar_id,
                length=dist,
                radius=r,
                cond=cond,
                type='pore'
            )

        self._prepare_network_solver()

    def build_from_pores(self, pore_specs, x_plenum):
        """
        One-chamber arbitrary passive pore layout.

        pore_specs = [(panel_id, radius), ...]
        """
        self.G.clear()
        self.active_pores = []

        spar_id = 99999
        spar_pos = np.array([x_plenum, 0.0])
        self.G.add_node(spar_id, pos=spar_pos, type='internal')

        for pid, r in pore_specs:
            if pid is None or pid < 0 or pid >= self.aero.N:
                raise RuntimeError(f"Invalid panel id: {pid}")
            if not np.isfinite(r) or r <= 0.0:
                raise RuntimeError(f"Invalid pore radius: {r}")

            p_pos = np.array([self.aero.XC[pid], self.aero.YC[pid]])
            dist = np.linalg.norm(p_pos - spar_pos)
            if not np.isfinite(dist) or dist <= 1e-12:
                raise RuntimeError(f"Invalid pore-plenum distance for panel {pid}")

            cond = (np.pi * r**4) / (8.0 * self.cfg.MU * dist)
            if not np.isfinite(cond) or cond <= 0.0:
                raise RuntimeError(f"Invalid conductance for panel {pid}")

            self.G.add_node(pid, pos=(self.aero.XC[pid], self.aero.YC[pid]), type='boundary', panel_idx=pid)
            self.G.add_edge(
                pid, spar_id,
                length=dist,
                radius=r,
                cond=cond,
                type='pore'
            )
            self.active_pores.append(pid)

        self._prepare_network_solver()

    def build_from_two_chambers(self, pore_specs, xA, xB, r_link):
        """
        Two-chamber passive closed-cavity layout.

        Parameters
        ----------
        pore_specs : list of tuples
            [(panel_id, pore_radius, chamber_id), ...]
            chamber_id must be 0 or 1
        xA : float
            x-location of chamber A
        xB : float
            x-location of chamber B
        r_link : float
            hydraulic radius of the passive link between chambers
        """
        self.G.clear()
        self.active_pores = []

        chamber_A = 90001
        chamber_B = 90002

        pos_A = np.array([xA, 0.0])
        pos_B = np.array([xB, 0.0])

        self.G.add_node(chamber_A, pos=pos_A, type='internal')
        self.G.add_node(chamber_B, pos=pos_B, type='internal')

        # Chamber-to-chamber passive internal link
        if not np.isfinite(r_link) or r_link <= 0.0:
            raise RuntimeError(f"Invalid inter-chamber link radius: {r_link}")

        dist_AB = np.linalg.norm(pos_A - pos_B)
        if not np.isfinite(dist_AB) or dist_AB <= 1e-12:
            raise RuntimeError("Invalid chamber spacing.")

        cond_AB = (np.pi * r_link**4) / (8.0 * self.cfg.MU * dist_AB)
        if not np.isfinite(cond_AB) or cond_AB <= 0.0:
            raise RuntimeError("Invalid chamber-to-chamber conductance.")

        self.G.add_edge(
            chamber_A, chamber_B,
            length=dist_AB,
            radius=r_link,
            cond=cond_AB,
            type='link'
        )

        # Pores connected to assigned chamber
        for pid, r_pore, chamber_id in pore_specs:
            if pid is None or pid < 0 or pid >= self.aero.N:
                raise RuntimeError(f"Invalid panel id: {pid}")
            if not np.isfinite(r_pore) or r_pore <= 0.0:
                raise RuntimeError(f"Invalid pore radius: {r_pore}")
            if chamber_id not in (0, 1):
                raise RuntimeError(f"Invalid chamber id {chamber_id}; must be 0 or 1.")

            chamber_node = chamber_A if chamber_id == 0 else chamber_B
            chamber_pos = pos_A if chamber_id == 0 else pos_B

            pore_pos = np.array([self.aero.XC[pid], self.aero.YC[pid]])
            dist = np.linalg.norm(pore_pos - chamber_pos)
            if not np.isfinite(dist) or dist <= 1e-12:
                raise RuntimeError(f"Invalid pore-chamber distance for panel {pid}")

            cond = (np.pi * r_pore**4) / (8.0 * self.cfg.MU * dist)
            if not np.isfinite(cond) or cond <= 0.0:
                raise RuntimeError(f"Invalid pore conductance for panel {pid}")

            self.G.add_node(
                pid,
                pos=(self.aero.XC[pid], self.aero.YC[pid]),
                type='boundary',
                panel_idx=pid
            )
            self.G.add_edge(
                pid, chamber_node,
                length=dist,
                radius=r_pore,
                cond=cond,
                type='pore'
            )
            self.active_pores.append(pid)

        self._prepare_network_solver()

    def _prepare_network_solver(self):
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
        """
        Solve passive internal redistribution.

        Returns
        -------
        velocities : dict
            Signed pore velocities keyed by panel id.
        P_nodes : ndarray
            Node pressures.
        fluxes : dict
            Signed pore volumetric fluxes keyed by panel id.
        net_flux : float
            Total flux balance across all pores; should be near zero.
        """
        b = np.zeros(self.n_nodes)

        for node in self.boundary_nodes:
            idx = self.node_map[node]
            pid = self.G.nodes[node]['panel_idx']
            b[idx] = P_boundary.get(pid, 0.0)

        try:
            P_nodes = self.net_solver(b)
        except Exception as e:
            raise RuntimeError(f"Network solve failed: {e}")

        if not np.all(np.isfinite(P_nodes)):
            raise RuntimeError("Network solve produced non-finite node pressures.")

        velocities = {}
        fluxes = {}

        for node in self.boundary_nodes:
            pid = self.G.nodes[node]['panel_idx']
            idx = self.node_map[node]

            Q_net = 0.0
            area = None

            for nbr in self.G.neighbors(node):
                edge = self.G[node][nbr]
                c = edge['cond']
                r = edge['radius']
                nbr_idx = self.node_map[nbr]

                Q_net += c * (P_nodes[idx] - P_nodes[nbr_idx])
                area = np.pi * r**2

            if area is None or area <= 0.0:
                raise RuntimeError(f"Invalid pore area at panel {pid}")

            fluxes[pid] = Q_net
            velocities[pid] = -Q_net / area

        net_flux = sum(fluxes.values())
        return velocities, P_nodes, fluxes, net_flux