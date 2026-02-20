# plotter.py
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from input import Config


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
        path = os.path.join(self.output_dir, "simulation_data.csv")
        with open(path, "w") as f:
            f.write("Metric,Solid_Baseline,Porous_Result,Change_Percent\n")
            cl_chg = ((CL - CL_solid) / (abs(CL_solid) + 1e-9)) * 100
            cd_chg = ((CD - CD_solid) / (abs(CD_solid) + 1e-9)) * 100
            f.write(f"CL,{CL_solid:.6f},{CL:.6f},{cl_chg:.2f}%\n")
            f.write(f"CD,{CD_solid:.6f},{CD:.6f},{cd_chg:.2f}%\n\n")
            f.write("Panel_ID,XC,YC,Cp_Solid,Cp_Porous,V_leakage\n")
            for i in range(aero.N):
                f.write(
                    f"{i},{aero.XC[i]:.6f},{aero.YC[i]:.6f},"
                    f"{Cp_solid[i]:.6f},{Cp[i]:.6f},{V_leakage[i]:.6f}\n"
                )

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
        ax1.plot(aero.X, aero.Y, "k-")
        ax1.fill(aero.X, aero.Y, "whitesmoke")
        pos = nx.get_node_attributes(net.G, "pos")
        b_nodes = [n for n in net.G.nodes if net.G.nodes[n]["type"] == "boundary"]
        nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax1, node_size=20, node_color="r")
        nx.draw_networkx_edges(net.G, pos, ax=ax1, edge_color="b", alpha=0.3)
        ax1.axis("equal")
        ax1.set_title("Network Topology")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(aero.XC, Cp_solid, "k--", label="Solid")
        ax2.plot(aero.XC, Cp, "b-", label="Porous")
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.set_title("Pressure Coefficient")

        fig.savefig(os.path.join(self.output_dir, "01_Geometry_Cp.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def _compute_directed_network_edges(self, net, P_nodes):
        if P_nodes is None or len(net.G.nodes()) == 0:
            return []
        pos = nx.get_node_attributes(net.G, "pos")
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        directed = []
        for u, v, d in net.G.edges(data=True):
            iu, iv = node_map[u], node_map[v]
            Pu, Pv = P_nodes[iu], P_nodes[iv]
            c = d.get("cond", 0.0)
            Q = c * (Pu - Pv)  # positive means u->v

            if Q >= 0:
                src, dst = u, v
                Qabs = Q
            else:
                src, dst = v, u
                Qabs = -Q

            is_inlet = d.get("type") == "plenum_in"
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = (Qabs / area) if area > 0 else 0.0

            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            directed.append(dict(x0=x0, y0=y0, x1=x1, y1=y1, vel=vel, Q=Qabs))
        return directed

    def _plot_flow_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False, constrained_layout=True)

        xg, yg = np.meshgrid(
            np.linspace(-0.5, 1.5, self.cfg.FLOW_NX),
            np.linspace(-0.6, 0.6, self.cfg.FLOW_NY),
        )

        cases = [
            ("SOLID: |V| + Streamlines", aero_solid, False),
            ("POROUS: |V| + Streamlines + Network Dir", aero_porous, True),
        ]
        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u**2 + v**2)

            cf = ax.contourf(xg, yg, mag, self.cfg.CONTOUR_LEVELS, cmap="viridis")
            ax.streamplot(xg, yg, u, v, density=self.cfg.STREAM_DENSITY, linewidth=0.6, arrowsize=0.8)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(net.G, pos, nodelist=b_nodes, ax=ax,
                                       node_size=12, node_color="red", edgecolors="white")

            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]], lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12:
                        continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate(
                        "",
                        xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                        xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, mutation_scale=13),
                        zorder=5,
                    )

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Velocity Magnitude |V| [m/s]")
        fig.savefig(os.path.join(self.output_dir, "05_Compare_FlowField_Velocity.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def _plot_pressure_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False, sharey=False, constrained_layout=True)

        xg, yg = np.meshgrid(
            np.linspace(-0.5, 1.5, self.cfg.FLOW_NX),
            np.linspace(-0.6, 0.6, self.cfg.FLOW_NY),
        )

        cases = [
            ("SOLID: Pressure", aero_solid, False),
            ("POROUS: Pressure + Network Dir", aero_porous, True),
        ]
        directed_edges = self._compute_directed_network_edges(net, P_nodes)

        cf = None
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            # Correct: compute_pressure_field returns (P_field, Cp_field)
            P_field, _Cp_field = aero_case.compute_pressure_field(xg, yg)

            # Pressure contours only (NO streamlines)
            cf = ax.contourf(xg, yg, P_field, self.cfg.CONTOUR_LEVELS, cmap="viridis")

            # Airfoil body
            ax.fill(aero_case.X, aero_case.Y, "k", zorder=3)

            # Pore nodes
            pos = nx.get_node_attributes(net.G, "pos")
            b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
            if pos and len(b_nodes) > 0:
                nx.draw_networkx_nodes(
                    net.G, pos, nodelist=b_nodes, ax=ax,
                    node_size=12, node_color="red", edgecolors="white"
                )

            # Directed network overlay (porous only)
            if overlay_net and directed_edges:
                for e in directed_edges:
                    ax.plot([e["x0"], e["x1"]], [e["y0"], e["y1"]], lw=2.0, alpha=0.9, zorder=4)

                for e in directed_edges:
                    dx = e["x1"] - e["x0"]
                    dy = e["y1"] - e["y0"]
                    L = np.hypot(dx, dy)
                    if L < 1e-12:
                        continue
                    mx = e["x0"] + 0.6 * dx
                    my = e["y0"] + 0.6 * dy
                    ax.annotate(
                        "",
                        xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                        xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                        arrowprops=dict(arrowstyle="-|>", lw=1.6, mutation_scale=13),
                        zorder=5,
                    )

            ax.set_title(title)
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.6, 0.6)
            ax.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(cf, ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cbar.set_label("Pressure [Pa]")
        fig.savefig(
            os.path.join(self.output_dir, "05b_Compare_FlowField_Pressure.png"),
            dpi=self.cfg.FIG_DPI,
            bbox_inches="tight"
        )
        plt.close(fig)

    def _plot_internal_flow(self, aero, net, P_nodes):
        if P_nodes is None:
            print("   (Skipping internal flow plot: P_nodes is None)")
            return

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        ax.plot(aero.X, aero.Y, "k-", lw=1.5, zorder=1)
        ax.fill(aero.X, aero.Y, "whitesmoke", zorder=0)

        pos = nx.get_node_attributes(net.G, "pos")
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        edge_data = []
        max_vel = 1e-9

        for u, v, d in net.G.edges(data=True):
            idx_u, idx_v = node_map[u], node_map[v]
            P_u, P_v = P_nodes[idx_u], P_nodes[idx_v]
            c = d["cond"]
            Q = c * (P_u - P_v)

            if Q >= 0:
                source, target = u, v
            else:
                source, target = v, u
                Q = abs(Q)

            is_inlet = d.get("type") == "plenum_in"
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad**2
            vel = Q / area
            max_vel = max(max_vel, vel)

            edge_data.append({"source": source, "target": target, "Q": Q, "vel": vel})

        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        cmap_vel = cm.get_cmap("plasma")
        norm_vel = mcolors.Normalize(vmin=0, vmax=max_vel)

        for edge in edge_data:
            x_s, y_s = pos[edge["source"]]
            x_t, y_t = pos[edge["target"]]
            vel = edge["vel"]
            color = cmap_vel(norm_vel(vel))

            ax.plot([x_s, x_t], [y_s, y_t], color=color, lw=2, alpha=0.7, zorder=2)

            mid_x, mid_y = x_s + 0.6 * (x_t - x_s), y_s + 0.6 * (y_t - y_s)
            dir_x, dir_y = x_t - x_s, y_t - y_s
            length = np.hypot(dir_x, dir_y)

            if length > 0 and vel > 1e-4:
                dx, dy = (dir_x / length) * 0.03, (dir_y / length) * 0.03
                ax.annotate(
                    "",
                    xy=(mid_x + dx, mid_y + dy),
                    xytext=(mid_x - dx, mid_y - dy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=15),
                    zorder=4,
                )

        p_values = [P_nodes[node_map[n]] for n in node_list]
        sc = ax.scatter(
            [pos[n][0] for n in node_list],
            [pos[n][1] for n in node_list],
            c=p_values,
            cmap="viridis",
            s=60,
            zorder=5,
            edgecolors="black",
        )

        cbar1 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Node Pressure [Pa]")

        sm = plt.cm.ScalarMappable(cmap=cmap_vel, norm=norm_vel)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("Pipe Velocity [m/s]")

        ax.axis("equal")
        ax.set_title("Internal Porous Network: Pressure & Flow Direction")
        fig.savefig(os.path.join(self.output_dir, "06_Internal_Flow.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def save_sweep_summary(self, cases, filename="polar_summary.csv"):
        """
        cases: list[SweepResult]
        Saves one CSV with solid baseline + each porous case.
        Assumes all cases share the same angle list (same sweep).
        """
        import numpy as np

        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        # Use first case as reference
        a = np.asarray(cases[0].angles, float)

        with open(path, "w") as f:
            f.write("--- POLAR SUMMARY ---\n")

            header = "Alpha_deg,CL_Solid,CD_Solid"
            for c in cases:
                header += f",{c.name}_CL,{c.name}_CD,{c.name}_DeltaCL,{c.name}_PctChangeCL,{c.name}_PctChangeCD"
            header += "\n"
            f.write(header)

            for i in range(len(a)):
                # solid baseline is the same for each case (taken from case 0)
                cl_s = cases[0].cl_solid[i]
                cd_s = cases[0].cd_solid[i]

                line = f"{a[i]:.2f},{cl_s:.6f},{cd_s:.6f}"
                for c in cases:
                    clp = c.cl_porous[i]
                    cdp = c.cd_porous[i]
                    dcl = clp - cl_s
                    pcl = 100.0 * dcl / (abs(cl_s) + 1e-12)
                    pcd = 100.0 * (cdp - cd_s) / (abs(cd_s) + 1e-12)
                    line += f",{clp:.6f},{cdp:.6f},{dcl:.6f},{pcl:.2f},{pcd:.2f}"
                line += "\n"
                f.write(line)

    def plot_polars(self, cases, filename_prefix="01"):
        """
        cases: list[SweepResult]
        Produces:
        1) CL vs AoA
        2) CD vs AoA
        3) Drag polar (CL vs CD)
        4) Efficiency (L/D) vs AoA
        5) Percentage changes (lift & drag) vs AoA
        """
        import numpy as np

        # styles
        style_solid = dict(color="gray", linestyle="--", linewidth=1.8, label="Solid Baseline")
        markers = ["o", "s", "D", "^", "v", "x"]

        # assume same alpha list
        aoa = np.asarray(cases[0].angles, float)
        cl_solid = np.asarray(cases[0].cl_solid, float)
        cd_solid = np.asarray(cases[0].cd_solid, float)

        # ---------------- FIGURE 1: 2x2 summary ----------------
        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle("Aerodynamic Polars Comparison", fontsize=16)

        ax1, ax2, ax3, ax4 = axes[0,0], axes[0,1], axes[1,0], axes[1,1]

        # CL vs AoA
        ax1.plot(aoa, cl_solid, **style_solid)
        for k, c in enumerate(cases):
            ax1.plot(aoa, c.cl_porous, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)
        ax1.set_title("Lift coefficient vs AoA")
        ax1.set_xlabel("AoA (deg)")
        ax1.set_ylabel("CL")
        ax1.grid(True, alpha=0.4)
        ax1.legend()

        # CD vs AoA
        ax2.plot(aoa, cd_solid, **style_solid)
        for k, c in enumerate(cases):
            ax2.plot(aoa, c.cd_porous, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)
        ax2.set_title("Drag coefficient vs AoA")
        ax2.set_xlabel("AoA (deg)")
        ax2.set_ylabel("CD")
        ax2.grid(True, alpha=0.4)

        # Drag polar: CL vs CD
        ax3.plot(cd_solid, cl_solid, **style_solid)
        for k, c in enumerate(cases):
            ax3.plot(c.cd_porous, c.cl_porous, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)
        ax3.set_title("Drag polar")
        ax3.set_xlabel("CD")
        ax3.set_ylabel("CL")
        ax3.grid(True, alpha=0.4)

        # Efficiency L/D vs AoA
        ld_s = cl_solid / (cd_solid + 1e-12)
        ax4.plot(aoa, ld_s, **style_solid)
        for k, c in enumerate(cases):
            clp = np.asarray(c.cl_porous, float)
            cdp = np.asarray(c.cd_porous, float)
            ldp = clp / (cdp + 1e-12)
            ax4.plot(aoa, ldp, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)
        ax4.set_title("Efficiency (L/D) vs AoA")
        ax4.set_xlabel("AoA (deg)")
        ax4.set_ylabel("CL/CD")
        ax4.grid(True, alpha=0.4)

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(os.path.join(self.output_dir, f"{filename_prefix}_Polars.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig1)

        # ---------------- FIGURE 2: Percentage changes ----------------
        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Relative Performance vs Solid Baseline", fontsize=14)

        for k, c in enumerate(cases):
            dcl_pct = 100.0 * np.asarray(c.delta_r_cl, float)
            dcd_pct = 100.0 * np.asarray(c.delta_r_cd, float)
            ax5.plot(aoa, dcl_pct, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)
            ax6.plot(aoa, dcd_pct, linestyle="-", marker=markers[k % len(markers)],
                    markersize=4, label=c.name)

        ax5.set_title("Percentage change in CL")
        ax5.set_xlabel("AoA (deg)")
        ax5.set_ylabel("ΔCL (%)")
        ax5.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax5.grid(True, alpha=0.4)
        ax5.legend()

        ax6.set_title("Percentage change in CD")
        ax6.set_xlabel("AoA (deg)")
        ax6.set_ylabel("ΔCD (%)")
        ax6.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax6.grid(True, alpha=0.4)
        ax6.legend()

        fig2.tight_layout()
        fig2.savefig(os.path.join(self.output_dir, f"{filename_prefix}_Percentage_Changes.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig2)

    def stack_case_images(self, sweep_result, out_name="Stacked_Cp_Summary.png"):
        """
        Takes SweepResult.capture_image_paths and stacks existing PNGs vertically.
        """
        from PIL import Image
        import os

        paths = getattr(sweep_result, "capture_image_paths", [])
        if not paths:
            return

        images = []
        for p in paths:
            if os.path.exists(p):
                images.append(Image.open(p))

        if not images:
            return

        widths, heights = zip(*(im.size for im in images))
        total_w = max(widths)
        total_h = sum(heights)

        canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
        y = 0
        for im in images:
            canvas.paste(im, (0, y))
            y += im.size[1]

        out_path = os.path.join(self.output_dir, out_name)
        canvas.save(out_path)
        print(f"-> Stacked Cp summary created: {out_path}")

