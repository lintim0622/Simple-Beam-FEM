# static_analysis_solver.py
import numpy as np
import time
from sys import exit

# Import classes and timer from mesh_module
from mesh_v2 import Material, Mesh, timer

# --- Core FEM Calculation Helper Functions ---

def d_shape_functions_d_xi_eta(xi, eta):
    """
    Derivatives of bilinear quadrilateral shape functions with respect to local coordinates (xi, eta).
    Returns two (4,) numpy arrays for dN/dxi and dN/deta respectively.
    """
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return dN_dxi, dN_deta

def get_jacobian_and_det(nodes_coords, dN_dxi, dN_deta):
    """
    Calculates the Jacobian matrix J and its determinant for an element.
    nodes_coords: (4, 2) array of nodal coordinates [[x1, y1], ..., [x4, y4]]
    dN_dxi: (4,) array of dN/dxi
    dN_deta: (4,) array of dN/deta
    Returns J (2x2 matrix) and detJ (scalar).
    """
    J = np.array([
        [np.dot(dN_dxi, nodes_coords[:, 0]), np.dot(dN_dxi, nodes_coords[:, 1])],
        [np.dot(dN_deta, nodes_coords[:, 0]), np.dot(dN_deta, nodes_coords[:, 1])]
    ])
    detJ = np.linalg.det(J)
    return J, detJ

def get_B_matrix(nodes_coords, xi, eta):
    """
    Calculates the strain-displacement matrix B for a quadrilateral element.
    B is a (3, 8) matrix for plane stress/strain.
    Returns B matrix and the Jacobian determinant (detJ).
    """
    dN_dxi, dN_deta = d_shape_functions_d_xi_eta(xi, eta)
    J, detJ = get_jacobian_and_det(nodes_coords, dN_dxi, dN_deta)
    
    if detJ <= 1e-9: # Check for near-zero or negative determinant (e.g., inverted element)
        print(f"Warning: Jacobian determinant is zero or negative ({detJ}). Element might be inverted. Returning zero B matrix.")
        return np.zeros((3, 8)), detJ # Return zero B matrix for problematic elements
    
    invJ = np.linalg.inv(J)
    
    # Derivatives of shape functions with respect to global coordinates (x, y)
    dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
    dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
    
    B = np.zeros((3, 8))
    for i in range(4): # For each node (N1, N2, N3, N4)
        B[0, 2*i] = dN_dx[i]       
        B[1, 2*i + 1] = dN_dy[i]   
        B[2, 2*i] = dN_dy[i]       
        B[2, 2*i + 1] = dN_dx[i]   
    return B, detJ

def get_shape_functions(xi, eta):
    """
    Bilinear quadrilateral shape functions.
    Returns a (4,) numpy array of N_i values.
    """
    N = 0.25 * np.array([
        (1 - xi) * (1 - eta),
        (1 + xi) * (1 - eta),
        (1 + xi) * (1 + eta),
        (1 - xi) * (1 + eta)
    ])
    return N

def get_N_matrix(xi, eta):
    """
    Calculates the shape function matrix N for a quadrilateral element.
    N is a (2, 8) matrix, used for mapping nodal displacements to displacement at (xi, eta).
    """
    N_vals = get_shape_functions(xi, eta)
    
    N = np.zeros((2, 8))
    for i in range(4): # For each node
        N[0, 2*i] = N_vals[i]     
        N[1, 2*i + 1] = N_vals[i] 
    return N

# --- Element-level Matrix Calculation (Helper for Calculate Class) ---

def calculate_element_matrices(element, material):
    """
    Calculates the element stiffness (ke) and consistent mass (me) matrices
    for a given element using 2x2 Gaussian integration.
    """
    # Access nodes from the 'nodes' list attribute of the Element object
    nodes_coords = np.array([node.position for node in element.nodes])
    ke = np.zeros((8, 8))
    me = np.zeros((8, 8))

    # 2x2 Gaussian integration points and weights
    # (xi, eta, weight)
    gauss_points = [
        (-1/np.sqrt(3), -1/np.sqrt(3), 1.0),
        ( 1/np.sqrt(3), -1/np.sqrt(3), 1.0),
        ( 1/np.sqrt(3),  1/np.sqrt(3), 1.0),
        (-1/np.sqrt(3),  1/np.sqrt(3), 1.0)
    ]
    
    for xi, eta, weight in gauss_points:
        B, detJ = get_B_matrix(nodes_coords, xi, eta)
        
        if detJ <= 1e-9: 
            continue 
            
        ke += (B.T @ material.D @ B) * detJ * weight * material.w 

        N = get_N_matrix(xi, eta)
        me += material.rho * (N.T @ N) * detJ * weight * material.w
            
    return ke, me

# --- Main FEM Calculation Class ---

class Calculate(object):  
    """
    Handles the assembly of global matrices, application of boundary conditions,
    and solving the static equilibrium equations.
    """
    @timer
    def __init__(self, msh, material):
        self.msh = msh
        self.material = material
        self.num_dof = self.msh.tot_node_num * 2 # 2 degrees of freedom per node (ux, uy)
        
        self.K_global = np.zeros((self.num_dof, self.num_dof))
        self.M_global = np.zeros((self.num_dof, self.num_dof)) 
        self.F_global = np.zeros(self.num_dof)
        
        # Pre-calculate element matrices for all elements during initialization
        self.precalculate_element_matrices() 
        
    @timer
    def precalculate_element_matrices(self):
        """
        Calculates and stores element stiffness (ke) and mass (me) matrices
        for all elements in the mesh. This is done once at setup.
        """
        print("Pre-calculating element matrices...")
        for elem in self.msh.elements:
            elem.ke, elem.me = calculate_element_matrices(elem, self.material)
        print("Element matrices pre-calculated.")

    @timer
    def assemble_global_matrices(self):
        """
        Assembles the global stiffness (K_global) and mass (M_global) matrices,
        and the global force vector (F_global) from individual element contributions.
        """
        print("Assembling global matrices...")
        self.K_global.fill(0) # Reset to zero before assembly
        self.M_global.fill(0) # Reset to zero before assembly
        self.F_global.fill(0) # Reset to zero before assembly

        for elem in self.msh.elements:
            # Get global DoF indices for the current element's nodes
            global_dofs = []
            for node in elem.nodes:
                global_dofs.append(node.gid[0]) # ux DoF
                global_dofs.append(node.gid[1]) # uy DoF
            
            # Assemble element stiffness and mass matrices into global matrices
            # Using NumPy's advanced indexing for efficiency
            self.K_global[np.ix_(global_dofs, global_dofs)] += elem.ke
            self.M_global[np.ix_(global_dofs, global_dofs)] += elem.me
        
        # Assemble nodal forces from Node objects into F_global vector
        for node in self.msh.nodes:
            self.F_global[node.gid[0]] += node.f_ext[0] # Fx contribution
            self.F_global[node.gid[1]] += node.f_ext[1] # Fy contribution
        
        print("Global matrices assembled.")
        return self.K_global, self.M_global, self.F_global

    @timer
    def apply_boundary_conditions(self, K_global_in, F_global_in):
        """
        Applies Dirichlet boundary conditions (constrained displacements)
        to the global stiffness matrix and force vector using the direct stiffness method.
        Returns modified K and F, and a list of constrained DoFs.
        """
        K_modified = K_global_in.copy()
        F_modified = F_global_in.copy()

        constrained_dofs = []
        
        for node in self.msh.nodes: # Iterate through all nodes in the mesh
            if node.is_constrained[0]: # Constraint in x-direction
                dof = node.gid[0]
                value = node.constrained_value[0]
                
                # Apply constraint: K[dof, :] = 0, K[:, dof] = 0, K[dof, dof] = 1, F[dof] = value
                K_modified[dof, :] = 0
                K_modified[:, dof] = 0
                K_modified[dof, dof] = 1.0 # Set diagonal to 1 for identity row/column
                F_modified[dof] = value
                constrained_dofs.append(dof)

            if node.is_constrained[1]: # Constraint in y-direction
                dof = node.gid[1]
                value = node.constrained_value[1]
                
                # Apply constraint
                K_modified[dof, :] = 0
                K_modified[:, dof] = 0
                K_modified[dof, dof] = 1.0
                F_modified[dof] = value
                constrained_dofs.append(dof)

        return K_modified, F_modified, constrained_dofs

    @timer
    def solve_static_system(self, K_global_modified, F_global_modified):
        """
        Solves the linear system of equations K * U = F for U (global displacements).
        Uses numpy.linalg.solve for robust solution.
        """
        print(f"Solving for {len(F_global_modified)} degrees of freedom...")
        try:
            U_global = np.linalg.solve(K_global_modified, F_global_modified)
            print("System solved successfully.")
            return U_global
        except np.linalg.LinAlgError as e:
            print(f"Error: Linear system could not be solved. Check boundary conditions for stability. Details: {e}")
            return np.zeros_like(F_global_modified) # Return zeros or handle error appropriately

    # Methods to retrieve global matrices/vector (after assembly)
    def get_mass_matrix(self):
        return self.M_global
    
    def get_stiffness_matrix(self):
        return self.K_global

    def get_force_vector(self):
        return self.F_global

# --- Main Simulation Execution ---
if __name__ == "__main__":
    
    t0 = time.time()
    
    # 1. Define Geometry and Mesh Parameters
    beam_length = 1000.0 # L
    beam_height = 20.0  # h
    num_elements_x = 200 # Nx
    num_elements_y = 4  # Ny

    print("--- Starting Static Analysis ---")
    
    # 2. Generate Mesh
    my_mesh = Mesh(beam_length, beam_height, num_elements_x, num_elements_y)
    print(f"Mesh generated with {my_mesh.tot_node_num} nodes and {my_mesh.tot_elem_num} elements.")

    # 3. Define Material Properties
    rho_steel = 7.85e-9 # t/mm^3
    E_steel = 210000.0  # MPa
    v_steel = 0.3
    
    steel = Material(rho=rho_steel, E=E_steel, poisson=v_steel)
    # Set cross-section (thickness) for plane stress/strain
    beam_thickness = 10.0 # Example: 10 mm out-of-plane thickness
    steel.cross_section(H=beam_height, W=beam_thickness) 
    print(f"Material: Young's Modulus={steel.E/1e3:.1f} GPa, Poisson's Ratio={steel.v}, Thickness={steel.w} mm")

    # 4. Initialize Calculate object (this will pre-calculate element matrices)
    fem_solver = Calculate(my_mesh, steel)

    # 5. Apply Boundary Conditions
    # Find and apply pinned support at bottom-left corner (fixed in X and Y)
    pinned_node = None
    for p_node in my_mesh.nodes:
        if np.isclose(p_node.position[0], 0.0) and np.isclose(p_node.position[1], 0.0):
            p_node.is_constrained[0] = True # Fix ux
            p_node.is_constrained[1] = True # Fix uy
            pinned_node = p_node
            print(f"Pinned support at Node {p_node.nid} ({p_node.position[0]:.2f}, {p_node.position[1]:.2f})")
            break

    # Find and apply roller support at bottom-right corner (fixed in Y)
    roller_node = None
    for p_node in my_mesh.nodes:
        if np.isclose(p_node.position[0], beam_length) and np.isclose(p_node.position[1], 0.0):
            p_node.is_constrained[1] = True # Fix uy
            roller_node = p_node
            print(f"Roller support at Node {p_node.nid} ({p_node.position[0]:.2f}, {p_node.position[1]:.2f})")
            break
    
    if not pinned_node or not roller_node:
        print("Warning: Boundary condition nodes not found! Check mesh dimensions and node positions.")
        exit() # Exit if critical BCs are missing

    # 6. Apply External Force (e.g., a concentrated force at a specific node)
    # This part replaces your `cal.apply_info(pid, P)` and `cal.node_f_ext(ti=0.0)`
    # 由於樑長度變為2000，中點節點ID需要重新確認
    # (Nx + 1) * (Ny + 1) 個節點，節點ID是 column-major
    # 中點的節點大約在 (Nx/2) * (Ny+1) + Ny/2
    # 對於 (100, 2) 網格，x方向100個單元，y方向2個單元
    # 節點數 (100+1)*(2+1) = 101 * 3 = 303
    # 樑的中點 x 座標為 beam_length / 2 = 1000.0
    # 在 2000.0 長度的樑中，中點的 x 座標為 1000.0
    # 尋找 x=1000.0, y=10.0 的節點 (中性軸)
    force_node = None
    for p_node in my_mesh.nodes:
        if np.isclose(p_node.position[0], beam_length / 2) and np.isclose(p_node.position[1], beam_height / 2):
            force_node = p_node
            break

    if force_node:
        force_node.f_ext = np.array([0.0, -1000.0]) # Example force vector
        print(f"Applied force {force_node.f_ext} to Node {force_node.nid} at ({force_node.position[0]:.2f}, {force_node.position[1]:.2f})")
    else:
        print(f"Error: Force node at mid-span ({beam_length/2}, {beam_height/2}) not found!")
        exit()

    # 7. Assemble Global Matrices
    K_global, M_global, F_global = fem_solver.assemble_global_matrices()

    # 8. Apply Boundary Conditions to Global System
    K_modified, F_modified, constrained_dofs = fem_solver.apply_boundary_conditions(K_global, F_global)

    # 9. Solve for Global Displacements
    U_global = fem_solver.solve_static_system(K_modified, F_modified)

    # 10. Update Node Displacements in the Mesh
    for node in my_mesh.nodes:
        node.displacement[0] = U_global[node.gid[0]]
        node.displacement[1] = U_global[node.gid[1]]

    print("\n--- Selected Displacement Results ---")
    print(f"Pinned Node ({pinned_node.nid}): u_x={pinned_node.displacement[0]:.6e}, u_y={pinned_node.displacement[1]:.6e}")
    print(f"Roller Node ({roller_node.nid}): u_x={roller_node.displacement[0]:.6e}, u_y={roller_node.displacement[1]:.6e}")
    print(f"Force Node ({force_node.nid}): u_x={force_node.displacement[0]:.6e}, u_y={force_node.displacement[1]:.6e}")

    # Find the node at the middle of the bottom edge to see displacement
    mid_bottom_node = None
    for p_node in my_mesh.nodes:
        if np.isclose(p_node.position[0], beam_length / 2) and np.isclose(p_node.position[1], 0.0):
            mid_bottom_node = p_node
            break
    if mid_bottom_node:
        print(f"Mid-Bottom Node ({mid_bottom_node.nid}): u_x={mid_bottom_node.displacement[0]:.6e}, u_y={mid_bottom_node.displacement[1]:.6e}")

    # 11. Post-processing: Calculate Stresses and Strains (at element center)
    print("\n--- Element Stresses and Strains (at element center) ---")
    
    element_stress_results = []
    for elem in my_mesh.elements:
        nodes_coords = np.array([node.position for node in elem.nodes])
        
        # Get B matrix at element center (xi=0, eta=0) for average strain/stress
        B_center, detJ_center = get_B_matrix(nodes_coords, 0.0, 0.0)
        
        if detJ_center <= 1e-9:
            strain_e = np.array([np.nan, np.nan, np.nan])
            stress_e = np.array([np.nan, np.nan, np.nan])
        else:
            element_dofs = []
            for node in elem.nodes:
                element_dofs.extend(node.gid) 
            
            u_element_local = U_global[element_dofs] 
            
            strain_e = B_center @ u_element_local 
            stress_e = steel.D @ strain_e       
        
        element_stress_results.append({
            'eid': elem.eid,
            'center_x': np.mean(nodes_coords[:,0]),
            'center_y': np.mean(nodes_coords[:,1]),
            'strain_xx': strain_e[0], 'strain_yy': strain_e[1], 'strain_xy': strain_e[2],
            'stress_xx': stress_e[0], 'stress_yy': stress_e[1], 'stress_xy': stress_e[2]
        })

    valid_stress_xx = [res['stress_xx'] for res in element_stress_results if not np.isnan(res['stress_xx'])]
    if valid_stress_xx:
        print(f"Max Stress_xx: {np.max(valid_stress_xx):.2f} MPa")
        print(f"Min Stress_xx: {np.min(valid_stress_xx):.2f} MPa")
        print(f"Average Stress_xx: {np.mean(valid_stress_xx):.2f} MPa")
    
    valid_stress_yy = [res['stress_yy'] for res in element_stress_results if not np.isnan(res['stress_yy'])]
    if valid_stress_yy:
        print(f"Max Stress_yy: {np.max(valid_stress_yy):.2f} MPa")
        print(f"Min Stress_yy: {np.min(valid_stress_yy):.2f} MPa")
        print(f"Average Stress_yy: {np.mean(valid_stress_yy):.2f} MPa")
    
    valid_stress_xy = [res['stress_xy'] for res in element_stress_results if not np.isnan(res['stress_xy'])]
    if valid_stress_xy:
        print(f"Max Stress_xy: {np.max(valid_stress_xy):.2f} MPa")
        print(f"Min Stress_xy: {np.min(valid_stress_xy):.2f} MPa")
        print(f"Average Stress_xy: {np.mean(valid_stress_xy):.2f} MPa")
        
    print(f"\nTotal analysis time: {time.time() - t0:.4f} seconds")

    # 12. Visualization of Mesh and Deformed Shape
    # The plot_mesh method is now part of the Mesh class, imported from mesh_module
    my_mesh.plot_mesh(is_plot=True, scale_factor=1) # Scale factor can be adjusted for visualization