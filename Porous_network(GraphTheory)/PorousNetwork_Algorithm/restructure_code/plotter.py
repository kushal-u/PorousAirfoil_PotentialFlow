# plotter.py
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
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

    # -------------------------
    # I/O
    # -------------------------
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

    # -------------------------
    # Public plotting entrypoint
    # -------------------------
    def plot_all(self, aero_solid, aero_porous, porous_net, Cp, Cp_solid, P_nodes):
        print(f"-> Generating plots in {self.output_dir}...")
        self._plot_geometry_cp(aero_porous, porous_net, Cp, Cp_solid)
        self._plot_flow_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)
        self._plot_pressure_field_comparison(aero_solid, aero_porous, porous_net, P_nodes)
        self._plot_internal_flow(aero_porous, porous_net, P_nodes)

    # =========================================================
    # FAST FIELD PLOTTING (core optimization)
    # =========================================================
    def plot_field_on_grid(
        self,
        ax,
        xg,
        yg,
        field,
        *,
        title="",
        cmap="viridis",
        robust=True,
        vmin=None,
        vmax=None,
        cbar=None,
        cbar_label="",
        interpolation="nearest",
    ):
        """
        Fast raster plotting for large regular grids (e.g., 1000x1000).

        Uses imshow (fast) instead of contourf (slow).
        """
        # field should be shape (Ny, Nx) matching xg/yg meshgrid
        field = np.asarray(field)

        # Robust color scaling (better structure visibility)
        if robust and (vmin is None or vmax is None):
            finite = np.isfinite(field)
            if finite.any():
                lo, hi = np.percentile(field[finite], [2, 98])
                if vmin is None:
                    vmin = float(lo)
                if vmax is None:
                    vmax = float(hi)

        extent = [float(xg.min()), float(xg.max()), float(yg.min()), float(yg.max())]
        im = ax.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation=interpolation,
            aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        if cbar is not None:
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            cb.set_label(cbar_label)

        return im

    def _make_grid(self, xmin, xmax, ymin, ymax, nx, ny, dtype=np.float32):
        xs = np.linspace(xmin, xmax, int(nx), dtype=dtype)
        ys = np.linspace(ymin, ymax, int(ny), dtype=dtype)
        xg, yg = np.meshgrid(xs, ys)
        return xg, yg

    def _make_stream_grid(self, xmin, xmax, ymin, ymax, nx, ny):
        """
        Use a COARSER grid for streamplot so we don't streamplot 1e6 points.
        """
        # Cap streamline grid density
        nx_s = int(min(max(120, nx // 5), 250))
        ny_s = int(min(max(120, ny // 5), 250))
        return self._make_grid(xmin, xmax, ymin, ymax, nx_s, ny_s, dtype=np.float64)

    # =========================================================
    # NETWORK OVERLAYS (fast Matplotlib primitives)
    # =========================================================
    def _get_boundary_positions(self, net):
        pos = nx.get_node_attributes(net.G, "pos")
        b_nodes = [n for n in net.G.nodes if net.G.nodes[n].get("type") == "boundary"]
        if not pos or not b_nodes:
            return None, None
        bx = [pos[n][0] for n in b_nodes]
        by = [pos[n][1] for n in b_nodes]
        return np.asarray(bx, float), np.asarray(by, float)

    def _compute_directed_edges(self, net, P_nodes):
        """
        Return list of segments and speeds (for coloring/thickness)
        """
        if P_nodes is None or len(net.G.nodes()) == 0:
            return [], []

        pos = nx.get_node_attributes(net.G, "pos")
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        segs = []
        speeds = []

        for u, v, d in net.G.edges(data=True):
            iu, iv = node_map[u], node_map[v]
            Pu, Pv = P_nodes[iu], P_nodes[iv]
            c = float(d.get("cond", 0.0))
            Q = c * (Pu - Pv)  # + means u->v

            if Q >= 0:
                src, dst, Qabs = u, v, Q
            else:
                src, dst, Qabs = v, u, -Q

            # Use pore radius selection for a speed proxy:
            is_inlet = d.get("type") == "plenum_in"
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad * rad
            vel = (Qabs / area) if area > 0 else 0.0

            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            segs.append([(x0, y0), (x1, y1)])
            speeds.append(vel)

        return segs, np.asarray(speeds, float)

    def _draw_network_overlay(self, ax, net, P_nodes, *, draw_edges=True, draw_arrows=True):
        """
        Fast overlay:
        - boundary nodes as scatter
        - directed edges as LineCollection + small arrow marks
        """
        bx, by = self._get_boundary_positions(net)
        if bx is not None:
            ax.scatter(bx, by, s=18, c="red", edgecolors="white", linewidths=0.6, zorder=6)

        if not draw_edges or P_nodes is None:
            return

        segs, speeds = self._compute_directed_edges(net, P_nodes)
        if len(segs) == 0:
            return

        # Color edges by speed (robust)
        vmin, vmax = None, None
        if speeds.size:
            finite = np.isfinite(speeds)
            if finite.any():
                lo, hi = np.percentile(speeds[finite], [5, 95])
                vmin, vmax = float(lo), float(hi)
        norm = Normalize(vmin=vmin, vmax=vmax)

        lc = LineCollection(segs, cmap="plasma", norm=norm, linewidths=2.0, alpha=0.85, zorder=5)
        lc.set_array(speeds)
        ax.add_collection(lc)

        if draw_arrows:
            # Draw small arrows on each segment (cheap)
            for seg in segs:
                (x0, y0), (x1, y1) = seg
                dx, dy = x1 - x0, y1 - y0
                L = np.hypot(dx, dy)
                if L < 1e-12:
                    continue
                mx, my = x0 + 0.6 * dx, y0 + 0.6 * dy
                ax.annotate(
                    "",
                    xy=(mx + 0.03 * dx / L, my + 0.03 * dy / L),
                    xytext=(mx - 0.03 * dx / L, my - 0.03 * dy / L),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, mutation_scale=12),
                    zorder=7,
                )

    # =========================================================
    # PLOTS
    # =========================================================
    def _plot_geometry_cp(self, aero, net, Cp, Cp_solid):
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 1)

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(aero.X, aero.Y, "k-", lw=1.3)
        ax1.fill(aero.X, aero.Y, "whitesmoke")
        ax1.set_aspect("equal", adjustable="box")
        ax1.set_title("Network Topology")

        # Network overlay fast
        self._draw_network_overlay(ax1, net, P_nodes=None, draw_edges=True, draw_arrows=False)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(aero.XC, Cp_solid, "k--", label="Solid")
        ax2.plot(aero.XC, Cp, "b-", label="Porous")
        ax2.invert_yaxis()
        ax2.grid(alpha=0.3)
        ax2.legend()
        ax2.set_title("Pressure Coefficient")
        ax2.set_xlabel("x/c")
        ax2.set_ylabel("Cp")

        fig.savefig(os.path.join(self.output_dir, "01_Geometry_Cp.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def _plot_flow_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        # Field grid (high res allowed)
        xmin, xmax = -0.5, 1.5
        ymin, ymax = -0.6, 0.6
        xg, yg = self._make_grid(xmin, xmax, ymin, ymax, self.cfg.FLOW_NX, self.cfg.FLOW_NY)

        # Streamline grid (coarser, always)
        xs, ys = self._make_stream_grid(xmin, xmax, ymin, ymax, self.cfg.FLOW_NX, self.cfg.FLOW_NY)

        cases = [
            ("SOLID: |V| + Streamlines", aero_solid, False),
            ("POROUS: |V| + Streamlines + Network Dir", aero_porous, True),
        ]

        ims = []
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            # Compute field at high-res grid
            u, v = aero_case.compute_velocity_field(xg, yg)
            mag = np.sqrt(u * u + v * v)

            # Fast raster plot
            im = self.plot_field_on_grid(
                ax, xg, yg, mag,
                title=title, cmap="viridis",
                cbar=None, robust=True,
                interpolation="nearest",
            )
            ims.append(im)

            # Streamplot computed on coarser grid
            u_s, v_s = aero_case.compute_velocity_field(xs, ys)
            ax.streamplot(xs, ys, u_s, v_s, density=self.cfg.STREAM_DENSITY, linewidth=0.6, arrowsize=0.8)

            # Airfoil body
            ax.fill(aero_case.X, aero_case.Y, "k", zorder=8)

            # Pores and directed network on porous panel only
            if overlay_net:
                self._draw_network_overlay(ax, net, P_nodes, draw_edges=True, draw_arrows=True)
            else:
                self._draw_network_overlay(ax, net, None, draw_edges=False)

        # Single shared colorbar
        cb = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cb.set_label("Velocity Magnitude |V| [m/s]")

        fig.savefig(os.path.join(self.output_dir, "05_Compare_FlowField_Velocity.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def _plot_pressure_field_comparison(self, aero_solid, aero_porous, net, P_nodes):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

        xmin, xmax = -0.5, 1.5
        ymin, ymax = -0.6, 0.6
        xg, yg = self._make_grid(xmin, xmax, ymin, ymax, self.cfg.FLOW_NX, self.cfg.FLOW_NY)

        cases = [
            ("SOLID: Pressure", aero_solid, False),
            ("POROUS: Pressure + Network Dir", aero_porous, True),
        ]

        ims = []
        for ax, (title, aero_case, overlay_net) in zip(axes, cases):
            P_field, _ = aero_case.compute_pressure_field(xg, yg)

            im = self.plot_field_on_grid(
                ax, xg, yg, P_field,
                title=title, cmap="viridis",
                robust=True,
                interpolation="nearest",
            )
            ims.append(im)

            ax.fill(aero_case.X, aero_case.Y, "k", zorder=8)

            if overlay_net:
                self._draw_network_overlay(ax, net, P_nodes, draw_edges=True, draw_arrows=True)
            else:
                self._draw_network_overlay(ax, net, None, draw_edges=False)

        cb = fig.colorbar(ims[-1], ax=axes.ravel().tolist(), fraction=0.035, pad=0.02)
        cb.set_label("Pressure [Pa]")

        fig.savefig(os.path.join(self.output_dir, "05b_Compare_FlowField_Pressure.png"),
                    dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    def _plot_internal_flow(self, aero, net, P_nodes):
        if P_nodes is None:
            print("   (Skipping internal flow plot: P_nodes is None)")
            return
        if len(net.G.nodes()) == 0:
            print("   (Skipping internal flow plot: empty network)")
            return

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)

        ax.plot(aero.X, aero.Y, "k-", lw=1.4, zorder=1)
        ax.fill(aero.X, aero.Y, "whitesmoke", zorder=0)

        pos = nx.get_node_attributes(net.G, "pos")
        node_list = list(net.G.nodes())
        node_map = {n: i for i, n in enumerate(node_list)}

        # Compute edge velocities
        segs = []
        vels = []
        for u, v, d in net.G.edges(data=True):
            iu, iv = node_map[u], node_map[v]
            Pu, Pv = P_nodes[iu], P_nodes[iv]
            c = float(d.get("cond", 0.0))
            Q = c * (Pu - Pv)

            if Q >= 0:
                src, dst, Qabs = u, v, Q
            else:
                src, dst, Qabs = v, u, -Q

            is_inlet = d.get("type") == "plenum_in"
            rad = self.cfg.PORE_RADIUS_INLET if is_inlet else self.cfg.PORE_RADIUS_OUTLET
            area = np.pi * rad * rad
            vel = (Qabs / area) if area > 0 else 0.0

            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            segs.append([(x0, y0), (x1, y1)])
            vels.append(vel)

        vels = np.asarray(vels, float)
        if vels.size and np.isfinite(vels).any():
            lo, hi = np.percentile(vels[np.isfinite(vels)], [5, 95])
            norm = Normalize(vmin=float(lo), vmax=float(hi))
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)

        lc = LineCollection(segs, cmap="plasma", norm=norm, linewidths=2.0, alpha=0.8, zorder=2)
        lc.set_array(vels)
        ax.add_collection(lc)

        # Node pressure scatter
        p_values = np.array([P_nodes[node_map[n]] for n in node_list], dtype=float)
        xy = np.array([pos[n] for n in node_list], dtype=float)

        sc = ax.scatter(
            xy[:, 0], xy[:, 1],
            c=p_values,
            cmap="viridis",
            s=55,
            zorder=5,
            edgecolors="black",
            linewidths=0.4,
        )

        cbar1 = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label("Node Pressure [Pa]")

        sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm)
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label("Pipe Velocity Proxy [m/s]")

        ax.set_aspect("equal", adjustable="box")
        ax.set_title("Internal Porous Network: Pressure & Flow")
        fig.savefig(os.path.join(self.output_dir, "06_Internal_Flow.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig)

    # =========================================================
    # Sweep utilities (kept)
    # =========================================================
    def save_sweep_summary(self, cases, filename="polar_summary.csv"):
        import numpy as np
        path = os.path.join(self.output_dir, filename)
        os.makedirs(self.output_dir, exist_ok=True)

        a = np.asarray(cases[0].angles, float)
        with open(path, "w") as f:
            f.write("--- POLAR SUMMARY ---\n")
            header = "Alpha_deg,CL_Solid,CD_Solid"
            for c in cases:
                header += f",{c.name}_CL,{c.name}_CD,{c.name}_DeltaCL,{c.name}_PctChangeCL,{c.name}_PctChangeCD"
            header += "\n"
            f.write(header)

            for i in range(len(a)):
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
        import numpy as np
        style_solid = dict(color="gray", linestyle="--", linewidth=1.8, label="Solid Baseline")
        markers = ["o", "s", "D", "^", "v", "x"]

        aoa = np.asarray(cases[0].angles, float)
        cl_solid = np.asarray(cases[0].cl_solid, float)
        cd_solid = np.asarray(cases[0].cd_solid, float)

        fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig1.suptitle("Aerodynamic Polars Comparison", fontsize=16)
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

        ax1.plot(aoa, cl_solid, **style_solid)
        for k, c in enumerate(cases):
            ax1.plot(aoa, c.cl_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
        ax1.set_title("Lift coefficient vs AoA")
        ax1.set_xlabel("AoA (deg)")
        ax1.set_ylabel("CL")
        ax1.grid(True, alpha=0.4)
        ax1.legend()

        ax2.plot(aoa, cd_solid, **style_solid)
        for k, c in enumerate(cases):
            ax2.plot(aoa, c.cd_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
        ax2.set_title("Drag coefficient vs AoA")
        ax2.set_xlabel("AoA (deg)")
        ax2.set_ylabel("CD")
        ax2.grid(True, alpha=0.4)

        ax3.plot(cd_solid, cl_solid, **style_solid)
        for k, c in enumerate(cases):
            ax3.plot(c.cd_porous, c.cl_porous, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
        ax3.set_title("Drag polar")
        ax3.set_xlabel("CD")
        ax3.set_ylabel("CL")
        ax3.grid(True, alpha=0.4)

        ld_s = cl_solid / (cd_solid + 1e-12)
        ax4.plot(aoa, ld_s, **style_solid)
        for k, c in enumerate(cases):
            clp = np.asarray(c.cl_porous, float)
            cdp = np.asarray(c.cd_porous, float)
            ldp = clp / (cdp + 1e-12)
            ax4.plot(aoa, ldp, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
        ax4.set_title("Efficiency (L/D) vs AoA")
        ax4.set_xlabel("AoA (deg)")
        ax4.set_ylabel("CL/CD")
        ax4.grid(True, alpha=0.4)

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(os.path.join(self.output_dir, f"{filename_prefix}_Polars.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig1)

        fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
        fig2.suptitle("Relative Performance vs Solid Baseline", fontsize=14)

        for k, c in enumerate(cases):
            dcl_pct = 100.0 * np.asarray(c.delta_r_cl, float)
            dcd_pct = 100.0 * np.asarray(c.delta_r_cd, float)
            ax5.plot(aoa, dcl_pct, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)
            ax6.plot(aoa, dcd_pct, linestyle="-", marker=markers[k % len(markers)], markersize=4, label=c.name)

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
        fig2.savefig(os.path.join(self.output_dir, f"{filename_prefix}_Percentage_Changes.png"), dpi=self.cfg.FIG_DPI, bbox_inches="tight")
        plt.close(fig2)

    def stack_case_images(self, sweep_result, out_name="Stacked_Cp_Summary.png"):
        from PIL import Image

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