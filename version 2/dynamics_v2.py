# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

from mesh_v2 import Mesh, Material, timer
from static_v2 import Calculate

def compute_rayleigh_alpha(zeta_target, omega_target):
    """
    根據目標模態阻尼比與模態頻率，回傳 Rayleigh damping 的 alpha 值
    假設只考慮單一模態（beta = 0）
    
    Parameters:
        zeta_target : float   # 目標阻尼比（如 0.03）
        omega_target : float  # 目標模態頻率（單位：rad/s）

    Returns:
        alpha : float
    """
    alpha = 2.0 * zeta_target * omega_target
    return alpha

class DynamicsSolver:
    def __init__(self, msh, material, t_end, dt, lumped_mass=False, alpha=0.0, beta=0.0):
        self.msh = msh
        self.material = material
        self.tns = np.arange(0, t_end + dt, dt)
        self.dt = dt
        self.len_tns = len(self.tns)

        self.total_dof = msh.tot_node_num * 2

        self.cal = Calculate(msh, material)
        self.K_global, self.M_global, self.F_global = self.cal.assemble_global_matrices()

        if lumped_mass:
            self.M_global = np.diag(np.sum(self.M_global, axis=1))

        self.dof_map = self._build_dof_map()
        self.free_dofs = self._get_free_dofs()
        self.num_free = len(self.free_dofs)

        self.U = np.zeros((self.len_tns, self.num_free))
        self.V = np.zeros((self.len_tns, self.num_free))
        self.A = np.zeros((self.len_tns, self.num_free))

        self.M_ff = self.M_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.K_ff = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.C_ff = alpha * self.M_ff + beta * self.K_ff  # Rayleigh damping

        self.M_inv = np.linalg.inv(self.M_ff)

    def _build_dof_map(self):
        map_list = []
        for node in self.msh.nodes:
            for i in range(2):
                if not node.is_constrained[i]:
                    map_list.append(node.gid[i])
        return np.array(map_list)

    def _get_free_dofs(self):
        return self._build_dof_map()

    def apply_modal_initial_velocity(self, mode_shape_func, direction=1, v0=1.0):
        for node in self.msh.nodes:
            x = node.position[0]
            shape_val = mode_shape_func(x)
            if node.is_constrained[direction]:
                continue
            gid = node.gid[direction]
            if gid in self.free_dofs:
                idx = self.free_dofs.tolist().index(gid)
                self.V[0, idx] = v0 * shape_val

    def apply_force(self, node_id, force_vector):
        F = np.zeros(self.total_dof)
        self.msh.nodes[node_id].f_ext = np.array(force_vector)
        for node in self.msh.nodes:
            F[node.gid[0]] += node.f_ext[0]
            F[node.gid[1]] += node.f_ext[1]
        return F[self.free_dofs]

    def step_newmark(self, i, F_f, beta=1/4, gamma=1/2):
        dt = self.dt
        M = self.M_ff
        K = self.K_ff
        C = self.C_ff

        a1 = 1 / (beta * dt**2)
        a2 = 1 / (beta * dt)
        a3 = (1 / (2 * beta)) - 1

        A_eff = M * a1 + C * (gamma / (beta * dt)) + K
        rhs = F_f \
              + M @ (a1 * self.U[i] + a2 * self.V[i] + a3 * self.A[i]) \
              + C @ ( (gamma / (beta * dt)) * self.U[i] + ((gamma / beta) - 1) * self.V[i] + dt * ((gamma / (2 * beta)) - 1) * self.A[i])

        self.U[i + 1] = np.linalg.solve(A_eff, rhs)
        self.A[i + 1] = a1 * (self.U[i + 1] - self.U[i]) - a2 * self.V[i] - a3 * self.A[i]
        self.V[i + 1] = self.V[i] + dt * ((1 - gamma) * self.A[i] + gamma * self.A[i + 1])

    @timer
    def run(self, force_node_id, force_vec, ti, result_tracker=None, method="central"):
        for i in range(self.len_tns - 1):
            if abs(self.tns[i] - ti) < 1e-10:
                F_f = self.apply_force(force_node_id, force_vec)
            else:
                F_f = np.zeros(self.num_free)

            if method == "newmark":
                self.step_newmark(i, F_f)
            else:
                self.step_central_difference(i, F_f)

            self.update_mesh_disp(i + 1)

            if result_tracker:
                result_tracker.record(i)

        if result_tracker:
            result_tracker.save_txt(self.tns)

    def step_central_difference(self, i, F_f):
        if i == 0:
            self.A[0] = self.M_inv @ (F_f - self.K_ff @ self.U[0])
            self.U[1] = self.U[0] + self.dt * self.V[0] + 0.5 * self.dt**2 * self.A[0]
        else:
            self.U[i + 1] = (self.dt**2 * self.M_inv @ (F_f - self.K_ff @ self.U[i]) +
                             2 * self.U[i] - self.U[i - 1])

    def update_mesh_disp(self, step_idx):
        full_disp = np.zeros(self.total_dof)
        full_disp[self.free_dofs] = self.U[step_idx]
        for node in self.msh.nodes:
            node.displacement[0] = full_disp[node.gid[0]]
            node.displacement[1] = full_disp[node.gid[1]]

    def get_displacement(self, node_id):
        node = self.msh.nodes[node_id]
        return node.displacement

    def modal_participation(self, shape_func, direction=1):
        phi_full = np.zeros(self.total_dof)
        for node in self.msh.nodes:
            if not node.is_constrained[direction]:
                gid = node.gid[direction]
                phi_full[gid] = shape_func(node.position[0])
        phi_dofs = phi_full[self.free_dofs]
        V0 = self.V[0]
        num = np.dot(V0, phi_dofs)
        den = np.dot(phi_dofs, phi_dofs)
        alpha = num / den if den != 0 else 0.0
        return alpha


class DynamicResults:
    def __init__(self, msh, solver, pid=0):
        self.msh = msh
        self.solver = solver
        self.pid = pid
        self.U_history = []

    def record(self, i):
        disp = np.zeros((self.msh.tot_node_num, 2))
        for node in self.msh.nodes:
            disp[node.nid, :] = node.displacement
        self.U_history.append(disp)

    def save_txt(self, tns, filename="FEM_DYNAMICS_CENTER.txt"):
        with open(filename, 'w') as f:
            for i, disp in enumerate(self.U_history):
                u = disp[self.pid, 0]
                v = disp[self.pid, 1]
                f.write(f"{i:7d} {self.pid:5d} {u:14.6e} {v:14.6e}\n")
        print(f"[✓] 中心節點位移歷史儲存至: {filename}")

    def output_fig(self, i, tns, k=0, folder="fig_store", scale=5.0):

        if not os.path.exists(folder):
            os.makedirs(folder)
    
        fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
        
        undeformed_segments = []
        deformed_segments = []
        
        for elem in self.msh.elements:
            nodes = elem.nodes
            orig_coords = [n.position for n in nodes]
            def_coords = [n.position + n.displacement * scale for n in nodes]
    
            undeformed_segments.extend([
                (orig_coords[0], orig_coords[1]),
                (orig_coords[1], orig_coords[2]),
                (orig_coords[2], orig_coords[3]),
                (orig_coords[3], orig_coords[0])
            ])
            deformed_segments.extend([
                (def_coords[0], def_coords[1]),
                (def_coords[1], def_coords[2]),
                (def_coords[2], def_coords[3]),
                (def_coords[3], def_coords[0])
            ])
    
        ax.add_collection(LineCollection(undeformed_segments, colors='gray', linewidths=0.8, label='Undeformed'))
        ax.add_collection(LineCollection(deformed_segments, colors='tab:red', linewidths=1.5, linestyles='--', label='Deformed'))
    
        ax.set_xlabel("Length (m)", fontsize=20)
        ax.set_ylabel("Height (m)", fontsize=20)
        ax.set_title(f"t = {tns[i]:.4f} s", fontsize=24)
        ax.set_aspect('equal', 'box')
        ax.legend(fontsize=14)
        plt.tick_params(labelsize=16)
        ax.set_xlim(-1.0, self.msh.L + 1.0)
        ax.set_ylim(-1.0, self.msh.h + 1.0)
    
        plt.tight_layout()
        fig_path = os.path.join(folder, f"beam_deformed_{i + k:05d}.png")
        plt.savefig(fig_path)
        plt.ioff()                   # ✅ 關閉互動模式，防止圖窗跳出
        plt.close()                 # ✅ 關閉圖窗釋放記憶體


    
    def output_fig_all(self, tns, skip=200, scale=5.0, folder="fig_store"):
        print(f"[*] Export images to folder '{folder}' (one per {skip} steps, scale = {scale})...")
        for i in range(0, len(self.U_history), skip):
            # 更新 mesh 位移
            for node in self.msh.nodes:
                node.displacement[:] = self.U_history[i][node.nid, :]
            self.output_fig(i, tns, k=0, folder=folder, scale=scale)


if __name__ == "__main__":
    
    L = 10.0
    h = 2.0
    w = 0.5
    Nx = 10
    Ny = 2
    dt = 1e-4
    t_end = 2.0
    force_time = 0.0                # 外力施加的時間 (秒)
    force_vec = [0.0, -10.0]        # 在 y 方向施加瞬時集中力 -10 N


    msh = Mesh(L, h, Nx, Ny)
    mat = Material(2300.0, 432e6, 0.3)
    mat.cross_section(h, w)

    for node in msh.nodes:
        if np.isclose(node.position[0], 0.0) and np.isclose(node.position[1], 0.0):
            node.is_constrained = [True, True]
        elif np.isclose(node.position[0], L) and np.isclose(node.position[1], 0.0):
            node.is_constrained[1] = True

    omega_1 = 10.0         # 第一模態頻率 (rad/s)
    zeta = 0.03            # 3% 阻尼比
    
    alpha = compute_rayleigh_alpha(zeta, omega_1)
    solver = DynamicsSolver(msh, mat, t_end, dt, lumped_mass=True, alpha=alpha, beta=0.0)

    beam_center = np.array([L / 2, h / 2])
    center_node = min(msh.nodes, key=lambda n: np.linalg.norm(n.position - beam_center))
    center_node_id = center_node.nid
    gid_y = center_node.gid[1]
    idx_y = solver.free_dofs.tolist().index(gid_y)

    print(f"[\u2713] Auto-identified center node: ID = {center_node_id}, Pos = {center_node.position}")

    solver.apply_modal_initial_velocity(lambda x: np.sin(np.pi * x / L), direction=1, v0=-0.01)

    results = DynamicResults(msh, solver, pid=center_node_id)
    solver.run(center_node_id, force_vec, force_time, result_tracker=results, method="newmark")
    
    alpha = solver.modal_participation(lambda x: np.sin(np.pi * x / L))
    print(f"Modal participation for mode 1: α = {alpha:.4f}")
    

    from postprocess_auto import auto_postprocess
    auto_postprocess(msh, results, solver.tns, skip=200, scale=50.0, dt=dt)
