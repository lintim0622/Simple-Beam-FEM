# -*- coding: utf-8 -*-
"""
Created on Sat May 24 23:48:30 2025

@author: lintim0622
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve # 導入稀疏矩陣求解器
from scipy.sparse import csr_matrix # <--- 新增: 導入 csr_matrix

# From your provided structure, Mesh, Material, and Calculate should be imported from your mesh_module.py and static_analysis_solver.py.
# Assuming mesh_module.py is renamed to mesh_v2.py and static_analysis_solver.py functions are in a Calculate class in static_v2.py for the original user context.
from mesh_v2 import Mesh, Material
from static_v2 import Calculate # This should be your static_analysis_solver.py's Calculate class

def compute_rayleigh_damping(alpha, beta, M, K):
    """
    Computes the Rayleigh damping matrix C = alpha * M + beta * K.
    M and K can be dense or sparse matrices.
    """
    # If M and K are dense, the result will be dense.
    # We will convert it to sparse later.
    return alpha * M + beta * K

def compute_theoretical_FRF(freq_range, L, E, I, rho, A, x_in, x_out, zeta=0.03, num_modes=20):
    """
    Computes the theoretical Frequency Response Function (FRF) for a simply-supported beam.
    The FRF is defined as displacement / force.

    Args:
        freq_range (np.array): Array of frequencies (Hz) for which to compute FRF.
        L (float): Length of the beam.
        E (float): Young's Modulus.
        I (float): Area moment of inertia.
        rho (float): Material density.
        A (float): Cross-sectional area.
        x_in (float): X-coordinate of the force input point.
        x_out (float): X-coordinate of the displacement output point.
        zeta (float): Damping ratio.
        num_modes (int): Number of modes to include in the summation.

    Returns:
        np.array: Complex array of FRF values H(omega).
    """
    omega_range = 2 * np.pi * freq_range  # Convert frequencies to angular frequencies (rad/s)
    H = np.zeros(len(freq_range), dtype=complex)

    # For a simply-supported beam, the mode shapes are sin(r*pi*x/L)
    # The modal mass m_r for the r-th mode is (rho * A * L) / 2
    # The natural frequency omega_r for the r-th mode is (r*pi/L)^2 * sqrt(E*I / (rho*A))

    for r in range(1, num_modes + 1):
        beta_r = r * np.pi / L
        omega_r = beta_r**2 * np.sqrt(E * I / (rho * A))

        # Mode shape at input and output locations
        phi_in = np.sin(beta_r * x_in)
        phi_out = np.sin(beta_r * x_out)

        # Modal mass (consistent with the simply-supported beam's modal properties)
        m_r = rho * A * L / 2

        # Denominator of the modal contribution for a single mode
        # D_r = m_r * (omega_r^2 - omega^2 + i * 2 * zeta * omega_r * omega)
        denominator = m_r * (omega_r**2 - omega_range**2 + 1j * 2 * zeta * omega_r * omega_range)

        # Numerator of the modal contribution
        numerator = phi_out * phi_in

        # Add the contribution of the r-th mode to the total FRF
        # H = sum(phi_out * phi_in / D_r)
        # Handle cases where denominator might be very close to zero (e.g., undamped system at resonance)
        # to avoid division by zero or inf values.
        H += numerator / np.where(np.abs(denominator) < 1e-18, 1e-18, denominator) # Add a small epsilon to avoid true zero division

    return H

class FRFSolver:
    def __init__(self, msh, material, alpha=0.0, beta=0.0):
        self.msh = msh
        self.material = material
        self.cal = Calculate(msh, material)
        
        # Assemble global matrices. These will be dense numpy arrays from static_v2.py
        K_full_dense, M_full_dense, _ = self.cal.assemble_global_matrices()
        
        # Convert dense matrices to sparse CSR format right after assembly
        # <--- 修改點 1: 將 numpy.ndarray 轉換為 csr_matrix
        K_full_sparse = csr_matrix(K_full_dense)
        M_full_sparse = csr_matrix(M_full_dense)
        
        self.total_dof = msh.tot_node_num * 2

        # Identify constrained DOFs and free DOFs
        all_dofs = np.arange(self.total_dof)
        constrained_dofs_set = set()
        for node in msh.nodes:
            if node.is_constrained[0]:
                constrained_dofs_set.add(node.gid[0])
            if node.is_constrained[1]:
                constrained_dofs_set.add(node.gid[1])
        
        # This self.free_dofs calculation is correct for the `FRFSolver`
        self.free_dofs = np.array([dof for dof in all_dofs if dof not in constrained_dofs_set], dtype=int)

        # Condense the global matrices to free degrees of freedom
        # Slicing sparse matrices with np.ix_ returns a sparse matrix (often COO),
        # then convert to CSR for spsolve efficiency.
        # <--- 修改點 2: 確保切片後也轉換為 CSR
        self.K_ff = K_full_sparse[np.ix_(self.free_dofs, self.free_dofs)].tocsr()
        self.M_ff = M_full_sparse[np.ix_(self.free_dofs, self.free_dofs)].tocsr()
        
        # Compute sparse damping matrix based on condensed M_ff and K_ff
        # The result of `compute_rayleigh_damping` will be a dense array,
        # so convert it to CSR.
        # <--- 修改點 3: 將 compute_rayleigh_damping 的結果轉換為 CSR
        self.C_ff = csr_matrix(compute_rayleigh_damping(alpha, beta, self.M_ff.toarray(), self.K_ff.toarray())) # toarray() to perform dense arithmetic then convert back
        # Note: If M_ff and K_ff are sparse, alpha*M_ff + beta*K_ff will already be sparse.
        # But if compute_rayleigh_damping takes dense, we need to convert back.
        # For this specific case, as M_ff and K_ff are already CSR, their linear combination is also CSR.
        # So, the above line can be simplified to:
        self.C_ff = (alpha * self.M_ff + beta * self.K_ff).tocsr()


    def assemble_force_vector(self, free_dof_id, amplitude=1.0):
        """
        Assembles a force vector in the reduced (free) DoF space.
        Returns a dense numpy array for the force vector.
        """
        F_f = np.zeros(len(self.free_dofs), dtype=complex)
        F_f[free_dof_id] = amplitude
        return F_f

    def solve_frf(self, input_dof_global, output_dof_global, freq_range, amplitude=1.0):
        """
        Solves for the Frequency Response Function (FRF) given input and output DOFs.
        H(omega) = U_output(omega) / F_input(omega)
        """
        # Check if input/output DOFs are actually free
        if input_dof_global not in self.free_dofs:
            print(f"Warning: Input DOF {input_dof_global} is constrained. Please ensure it's unconstrained for proper force application.")
            # Or raise an error: raise ValueError(f"❌ Input DOF {input_dof_global} is constrained. Please select an unconstrained DOF for input.")
        if output_dof_global not in self.free_dofs:
            print(f"Warning: Output DOF {output_dof_global} is constrained. Please ensure it's unconstrained for proper displacement reading.")
            # Or raise an error: raise ValueError(f"❌ Output DOF {output_dof_global} is constrained. Please select an unconstrained DOF for output.")


        omegas = 2 * np.pi * freq_range  # rad/s
        H = np.zeros(len(freq_range), dtype=complex)

        # Map global DOFs to their indices within the free_dofs array
        # Using np.where to find index, handle cases where dof might not be in free_dofs (though we check above)
        input_idx_array = np.where(self.free_dofs == input_dof_global)[0]
        output_idx_array = np.where(self.free_dofs == output_dof_global)[0]

        if len(input_idx_array) == 0:
            print(f"Error: Input DOF {input_dof_global} not found in free_dofs. Cannot proceed.")
            return freq_range, np.full_like(freq_range, np.nan, dtype=complex)
        if len(output_idx_array) == 0:
            print(f"Error: Output DOF {output_dof_global} not found in free_dofs. Cannot proceed.")
            return freq_range, np.full_like(freq_range, np.nan, dtype=complex)

        input_idx = input_idx_array[0]
        output_idx = output_idx_array[0]

        F_f = self.assemble_force_vector(input_idx, amplitude)
        print(f"[Debug] Applied force at free_dofs[{input_idx}] (global DOF {input_dof_global}) with amplitude {amplitude}")

        for i, omega in enumerate(omegas):
            # Dynamic stiffness matrix: Z = K - omega^2 * M + i * omega * C
            # Z will also be a sparse matrix. Ensure it's CSR just before spsolve.
            # Operations like -omega**2 * self.M_ff + 1j * omega * self.C_ff + self.K_ff
            # on CSR matrices usually result in a CSR matrix, but .tocsr() ensures it.
            Z = (-omega**2 * self.M_ff + 1j * omega * self.C_ff + self.K_ff).tocsr() # <--- 確保 Z 是 CSR

            try:
                # Solve U_f = Z^-1 * F_f using sparse solver
                U_f = spsolve(Z, F_f) # F_f should be a dense numpy array
                H[i] = U_f[output_idx] / amplitude # FRF = Output Displacement / Input Force
            except RuntimeError as e: # spsolve might raise RuntimeError for singular matrix
                print(f"Warning: Sparse solver error at omega={omega:.2f} rad/s ({freq_range[i]:.2f} Hz). Details: {e}. Setting FRF to NaN.")
                H[i] = np.nan
            except Exception as e: # Catch other potential errors during solving
                print(f"Warning: Unexpected error at omega={omega:.2f} rad/s ({freq_range[i]:.2f} Hz). Details: {e}. Setting FRF to NaN.")
                H[i] = np.nan

        return freq_range, H

    def plot_frf(self, freq, H, H_theory=None, ylabel='|H(jw)| (m/N)', title='Frequency Response Function'):
        """
        Plots the magnitude and phase of the FRF.
        """
        mag_db = 20 * np.log10(np.abs(H))
        phase = np.angle(H, deg=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # Increased figsize for better readability
        
        # Magnitude Plot
        ax1.semilogx(freq, mag_db, label="FEM", linewidth=2)
        if H_theory is not None:
            mag_theory_db = 20 * np.log10(np.abs(H_theory))
            ax1.semilogx(freq, mag_theory_db, '--', label="Theory", linewidth=2)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12) # Increased fontsize
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax1.legend(fontsize=10) # Increased fontsize
        ax1.set_title(title, fontsize=14) # Title for the overall plot


        # Phase Plot
        ax2.semilogx(freq, phase, label="FEM", linewidth=2) # Use semilogx for consistency
        if H_theory is not None:
            ax2.semilogx(freq, np.angle(H_theory, deg=True), '--', label="Theory", linewidth=2)
        ax2.set_ylabel('Phase (deg)', fontsize=12) # Increased fontsize
        ax2.set_xlabel('Frequency (Hz)', fontsize=12) # Increased fontsize
        ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax2.legend(fontsize=10) # Increased fontsize
        ax2.set_yticks([-180, -90, 0, 90, 180]) # Common phase angles

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    # Geometry and mesh (using consistent units: m for length, kg for mass, N for force)
    L = 10.0  # meters
    h = 0.2   # meters (height of beam, was 2.0, reduced for more beam-like proportions)
    w = 0.5   # meters (out-of-plane thickness)
    Nx = 100  # Number of elements in X (can increase to 200 or 400 for better accuracy, but will slow down)
    Ny = 2    # Number of elements in Y

    msh = Mesh(L, h, Nx, Ny)

    # Material
    rho = 2300.0  # kg/m^3 (approx. for Aluminum, common in vibration examples)
    E = 70e9      # Pa (Young's Modulus for Aluminum)
    nu = 0.3      # Poisson's Ratio
    mat = Material(rho, E, nu)
    mat.cross_section(h, w) # Use h for height, w for width/thickness
    
    # Extract A and I directly from the material object after setting cross-section
    A_theory = mat.A
    I_theory = mat.I
    print(f"[✓] Material Properties for Theory: A={A_theory:.4e} m^2, I={I_theory:.4e} m^4")


    # Apply boundary conditions for a simply supported beam (Pinned-Roller)
    # Node at (0,0) is pinned (fixed in X and Y)
    for node in msh.nodes:
        if np.isclose(node.position[0], 0.0) and np.isclose(node.position[1], 0.0):
            node.is_constrained = [True, True]
            print(f"Pinned support at Node {node.nid} ({node.position[0]:.2f}, {node.position[1]:.2f})")
        # Node at (L,0) is roller (fixed in Y)
        elif np.isclose(node.position[0], L) and np.isclose(node.position[1], 0.0):
            node.is_constrained[1] = True
            print(f"Roller support at Node {node.nid} ({node.position[0]:.2f}, {node.position[1]:.2f})")

    # Auto-identify input and output DOFs
    # For a simply supported beam, an input force at mid-span in Y-direction
    # and output displacement at mid-span in Y-direction is a common FRF setup.
    # Find the node closest to (L/2, h/2) (mid-span, mid-height, i.e., neutral axis)
    mid_span_node_pos = np.array([L / 2, h / 2])
    input_node = min(msh.nodes, key=lambda n: np.linalg.norm(n.position - mid_span_node_pos))
    input_dof = input_node.gid[1] # Y-direction DOF (vertical force/displacement)

    # Find the output node (can be the same as input, or different)
    # Let's use the same node for input and output (colocated FRF) for a clearer comparison.
    output_node = input_node
    output_dof = output_node.gid[1] # Y-direction DOF

    # Get the actual x-coordinates for theoretical calculation
    x_in_theory = input_node.position[0]
    x_out_theory = output_node.position[0]

    print(f"[✓] Input DOF: {input_dof} (Node {input_node.nid} at {input_node.position[0]:.2f}, {input_node.position[1]:.2f})")
    print(f"[✓] Output DOF: {output_dof} (Node {output_node.nid} at {output_node.position[0]:.2f}, {output_node.position[1]:.2f})")
    print(f"[✓] Theoretical x_in: {x_in_theory:.2f}, Theoretical x_out: {x_out_theory:.2f}")

    # Frequency sweep setup
    # Adjust frequency range based on expected natural frequencies.
    # For a simply supported beam, first natural frequency (Hz): f1 = (pi/L)^2 * sqrt(EI/(rho*A)) / (2*pi)
    # With L=10, h=0.2, w=0.5, E=70e9, rho=2300:
    # A = 0.2 * 0.5 = 0.1 m^2
    # I = 0.5 * 0.2^3 / 12 = 0.0003333 m^4
    # f1 = (pi/10)^2 * sqrt(70e9 * 0.0003333 / (2300 * 0.1)) / (2*pi) = 4.41 Hz (approx)
    # So, up to 100 Hz or 200 Hz should capture multiple modes.
    freq_range = np.linspace(0.1, 200, 1000) # Increased frequency range and points

    # Rayleigh damping parameters
    zeta = 0.03  # Damping ratio
    
    # Let's calculate alpha and beta for 3% damping at the first two natural frequencies
    # For a simply supported beam, omega_r = (r*pi/L)^2 * sqrt(E*I / (rho*A))
    omega1_calc = (1 * np.pi / L)**2 * np.sqrt(E * I_theory / (rho * A_theory))
    omega2_calc = (2 * np.pi / L)**2 * np.sqrt(E * I_theory / (rho * A_theory))

    # Revert to a simpler Rayleigh damping choice, aiming for zeta at a single reference frequency.
    # Only mass-proportional damping for simplicity, common for low frequencies.
    omega_ref_for_damping = omega1_calc # Use the first natural frequency as reference
    alpha = 2 * zeta * omega_ref_for_damping
    beta = 0.0 

    print(f"[✓] Rayleigh Damping: alpha={alpha:.4e}, beta={beta:.4e} (targeted zeta={zeta} at first natural frequency)")

    solver = FRFSolver(msh, mat, alpha=alpha, beta=beta)
    freq_fem, H_fem = solver.solve_frf(input_dof, output_dof, freq_range)

    # Theoretical FRF calculation
    # Pass the material properties and coordinates explicitly
    H_theory = compute_theoretical_FRF(freq_range, L, E, I_theory, rho, A_theory, x_in_theory, x_out_theory, zeta=zeta, num_modes=20) # Increased num_modes for better resolution

    # Plot comparison
    solver.plot_frf(freq_fem, H_fem, H_theory=H_theory, ylabel='|U_y/F_y| (m/N)', title='FRF: FEM vs Theory for Simply-Supported Beam (Y-dir)')