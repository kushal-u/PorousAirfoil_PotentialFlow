import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from input import Config

# ==============================================================================
# VISUALIZATION
# ==============================================================================
class Visualizer:
    def __init__(self, config: Config):
        self.cfg = config
        
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir = os.getcwd()
        self.output_dir = os.path.join(base_dir, self.cfg.OUTPUT_DIR)
        os.makedirs(self.output_dir, exist_ok=True)

    def save_csv(self, aero, Cp, Cp_solid, V_leakage, CL, CL_solid, CD, CD_solid):
        path = os.path.join(self.output_dir, 'simulation_data.csv')
        with open(path, 'w') as f:
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            cl_chg = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100
            cd_chg = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_chg:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_chg:.2f}%\n\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n")

    def plot_all(self, aero_solid, aero_porous, porous_net, Cp, Cp_solid, P_nodes):
        print(f"-> Generating plots in {self.output_dir}...")
        self._plot_geometry_cp(aero_porous, porous_net, Cp, Cp_solid)
        self._plot_flow_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)
        self._plot_pressure_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)
        self._plot_internal_flow(aero_porous, porous_net, P_nodes)

    def _plot_geometry_cp(self, aero, net, Cp, Cp_solid):
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1)
        
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(aero.X, aero.Y, 'k-')
        ax1.fill(aero.X, aero.Y, 'whitesmoke')
        pos = nx.get_node_attributes(net.G, 'pos')
        b_nodes = [n for n in net.G.nodes if net.G.nodes[n]['type']=='boundary']
        nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax1, node_size=20, node_color='r')
        nx.draw_networkx_edges(net.G, pos, ax=ax1, edge_color='b', alpha=0.3)
        ax1.axis('equal')
        ax1.set_title("Network Topology")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(aero.XC, Cp_solid, 'k--', label='Solid')
        ax2.plot(aero.XC, Cp, 'b-', label='Porous')
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.set_title("Pressure Coefficient")
        
        fig.savefig(os.path.join(self.output_dir, '01_Geometry_Cp.png'), dpi=150)
        plt.close(fig)

    def _plot_internal_flow(self, aero, net, P_nodes):
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        ax.plot(aero.X, aero.Y, 'k-', lw=1.5, zorder=1)
        ax.fill(aero.X, aero.Y, 'whitesmoke', zorder=0)
        
        pos = nx.get_node_attributes(net.G, 'pos')
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}
        
        edge_data = []
        max_vel = 1e-9 
        
        for u, v, d in net.G.edges(data=True):
            idx_u, idx_v = node_map[u], node_map[v]
            P_u, P_v = P_nodes[idx_u], P_nodes[idx_v]
            
            c = d['cond']
            Q = c * (P_u - P_v)
            
            if Q >= 0:
                source, target = u, v
            else:
                source, target = v, u
                Q = abs(Q)
                
            is_inlet = d.get('type') == 'plenum_in'
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = Q / area
            max_vel = max(max_vel, vel)
            
            edge_data.append({'source': source, 'target': target, 'Q': Q, 'vel': vel})

        cmap_vel = cm.get_cmap('plasma')
        norm_vel = mcolors.Normalize(vmin=0, vmax=max_vel)

        for edge in edge_data:
            x_s, y_s = pos[edge['source']]
            x_t, y_t = pos[edge['target']]
            vel = edge['vel']
            color = cmap_vel(norm_vel(vel))
            
            ax.plot([x_s, x_t], [y_s, y_t], color=color, lw=2, alpha=0.7, zorder=2)
            
            mid_x, mid_y = x_s + 0.6 * (x_t - x_s), y_s + 0.6 * (y_t - y_s)
            dir_x, dir_y = x_t - x_s, y_t - y_s
            length = np.hypot(dir_x, dir_y)
            
            if length > 0 and vel > 1e-4:  
                dx, dy = (dir_x / length) * 0.03, (dir_y / length) * 0.03
                ax.annotate('', xy=(mid_x + dx, mid_y + dy), xytext=(mid_x - dx, mid_y - dy),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, 
                                            mutation_scale=15), zorder=4)

        p_values = [P_nodes[node_map[n]] for n in node_list]
        sc = ax.scatter([pos[n][0] for n in node_list],
                        [pos[n][1] for n in node_list],
                        c=p_values, cmap='viridis', s=60, zorder=5, edgecolors='black')
        
        cbar1 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Node Pressure [Pa]")
        
        sm = plt.cm.ScalarMappable(cmap=cmap_vel, norm=norm_vel)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("Pipe Velocity [m/s]")

        ax.axis('equal')
        ax.set_title("Internal Porous Network: Pressure & Flow Direction")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, '06_Internal_Flow.png'), dpi=150)
        plt.close(fig)

    def _compute_directed_network_edges(self, net, P_nodes):
        if P_nodes is None or len(net.G.nodes()) == 0:
            return []

        pos = nx.get_node_attributes(net.G, 'pos')
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        directed = []
        for u, v, d in net.G.edges(data=True):
            iu, iv = node_map[u], node_map[v]
            Pu, Pv = P_nodes[iu], P_nodes[iv]
            c = d.get('cond', 0.0)
            Q = c * (Pu - Pv) 

            if Q >= 0:
                src, dst = u, v
                Qabs = Q
            else:
                src, dst = v, u
                Qabs = -Q

            is_inlet = d.get('type') == 'plenum_in'
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = (Qabs / area) if area > 0 else 0.0

            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            directed.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, vel=vel, Q=Qabs))

        return directed

    def _plot_flow_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5), sharex=False, sharey=False, constrained_layout=True
        )

        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 1000), np.linspace(-0.6, 0.6, 1000))

        cases = [
            ("SOLID: |V| + Streamlines", aero_solid, False),
            ("POROUS: |V| + Streamlines + Network Dir", aero_porous, True),
        ]

        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u**2 + v**2)

            cf = ax.contourf(xg, yg, mag, 40, cmap="viridis")
            ax.streamplot(xg, yg, u, v, density=1.6, linewidth=0.7, arrowsize=0.9)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax,
                                    node_size=14, node_color="red", edgecolors="white")

            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]], lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12: continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate("", xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                                xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                                arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=14),
                                zorder=5)

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Velocity Magnitude |V| [m/s]")

        fig.savefig(os.path.join(self.output_dir, "05_Compare_FlowField_Velocity.png"), dpi=150)
        plt.close(fig)

    def _plot_pressure_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(
            1, 2, figsize=(14, 5), sharex=False, sharey=False, constrained_layout=True
        )

        xg, yg = np.meshgrid(np.linspace(-0.5, 1.5, 1000), np.linspace(-0.6, 0.6, 1000))

        cases = [
            ("SOLID: Pressure + Streamlines", aero_solid, False),
            ("POROUS: Pressure + Streamlines + Network Dir", aero_porous, True),
        ]

        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u**2 + v**2)

            Cp_field = 1.0 - (mag / (self.cfg.V_INF + 1e-12))**2
            P_field = self.cfg.P_INF + 0.5 * self.cfg.RHO * (self.cfg.V_INF**2) * Cp_field

            cf = ax.contourf(xg, yg, P_field, 40, cmap="viridis")
            ax.streamplot(xg, yg, u, v, density=1.6, linewidth=0.7, arrowsize=0.9)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax,
                                    node_size=14, node_color="red", edgecolors="white")

            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]], lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12: continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate("", xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                                xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                                arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=14),
                                zorder=5)

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect('equal', adjustable='box')

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Pressure [Pa]")

        fig.savefig(os.path.join(self.output_dir, "05b_Compare_FlowField_Pressure.png"), dpi=150)
        plt.close(fig)