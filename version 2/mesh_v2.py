# mesh_module.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import time

def timer(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

class Material(object):
    def __init__(self, rho, E, poisson):
        self.rho = rho
        self.E   = E
        self.v   = poisson
        
        # Cross-section properties, set later by cross_section method
        self.h = 0.0
        self.w = 0.0
        self.A = 0.0
        self.I = 0.0
        
        # Plane stress constitutive matrix D
        E1 = self.E / (1 - self.v**2)
        E2 = self.v * self.E / (1 - self.v**2)
        self.G = self.E / (2 * (1 + self.v)) # Shear Modulus
        self.D = np.array([[E1,    E2,    0.0],
                           [E2,    E1,    0.0],
                           [0.0 , 0.0, self.G]])
    
    def cross_section(self, H, W):
        """
        Sets the cross-sectional dimensions and calculates area and moment of inertia.
        H: Height of the cross-section
        W: Width (out-of-plane thickness) of the cross-section
        """
        self.h = float(H)
        self.w = float(W)
        self.A = self.w * self.h
        self.I = self.w * self.h**3 / 12.0

class Node(object):
    def __init__(self, nid=0, position=np.zeros(2)):
        self.nid = nid
        # Global degrees of freedom IDs (0-indexed for NumPy)
        self.gid = np.array([nid * 2, nid * 2 + 1], dtype=int) 
        self.position = position # Initial (x, y) coordinates
        
        # Analysis-specific properties (initialized to zero vectors)
        self.displacement = np.zeros(2) # [u_x, u_y]
        self.velocity = np.zeros(2)     # [v_x, v_y] (for dynamic analysis, not used in static)
        self.f_ext = np.zeros(2)        # External force [F_x, F_y] applied to this node
        
        # Boundary condition flags and values
        self.is_constrained = [False, False] # [is_constrained_x, is_constrained_y]
        self.constrained_value = [0.0, 0.0]  # Constrained displacement value [val_x, val_y]

class Element(object):
    def __init__(self, eid=0, n1=None, n2=None, n3=None, n4=None):
        self.eid = eid
        # Store Node objects directly for easier access
        self.nodes = [n1, n2, n3, n4] # Nodes are stored in a list
        self.me = np.zeros((8, 8)) # Element mass matrix (8x8 for 2D quad)
        self.ke = np.zeros((8, 8)) # Element stiffness matrix (8x8 for 2D quad)
        
class Mesh(object):
    @timer
    def __init__(self, L, h, Nx, Ny):
        self.L = L
        self.h = h
        self.Nx = Nx
        self.Ny = Ny
        self.tot_elem_num = Nx * Ny
        self.tot_node_num = (Nx + 1) * (Ny + 1)
        self.elements = [None] * self.tot_elem_num # Pre-allocate list with None
        self.nodes = [None] * self.tot_node_num    # Pre-allocate list with None
        
        # Generate node positions using numpy.linspace and meshgrid for efficiency
        xn = np.linspace(0, L, Nx + 1)
        yn = np.linspace(0, h, Ny + 1)
        xx, yy = np.meshgrid(xn, yn, indexing='ij') # 'ij' indexing: x varies along rows, y along columns
        
        # Flatten and combine x and y coordinates into a (tot_node_num, 2) array
        node_positions_flat = np.column_stack((xx.ravel(), yy.ravel()))

        # Create Node objects
        for i in range(self.tot_node_num):
            self.nodes[i] = Node(nid=i, position=node_positions_flat[i])
        
        # Create Element objects, referencing Node objects
        # Nodes are numbered in a column-major order (similar to your original code's logic)
        eid = 0
        nodes_per_x_column = Ny + 1 # Number of nodes in one vertical column

        for i_x in range(Nx): # Loop through element columns (0 to Nx-1)
            for i_y in range(Ny): # Loop through element rows (0 to Ny-1)
                # Calculate the Node IDs for the 4 nodes of the current element
                # The nodes are ordered counter-clockwise starting from bottom-left (n1)
                n1_idx = i_x * nodes_per_x_column + i_y
                n2_idx = (i_x + 1) * nodes_per_x_column + i_y
                n3_idx = (i_x + 1) * nodes_per_x_column + (i_y + 1)
                n4_idx = i_x * nodes_per_x_column + (i_y + 1)

                self.elements[eid] = Element(
                    eid=eid,
                    n1=self.nodes[n1_idx],
                    n2=self.nodes[n2_idx],
                    n3=self.nodes[n3_idx],
                    n4=self.nodes[n4_idx]
                )
                eid += 1
    
    def plot_mesh(self, is_plot=False, scale_factor=1.0):
        """
        Plots the undeformed and deformed mesh using LineCollection for efficiency.
        scale_factor: Multiplier for displacement to visualize deformation.
        """
        if is_plot:
            plt.rcParams["font.family"] = "Times New Roman"
            
            # 調整圖形大小和邊距以提供更多空間
            fig, ax = plt.subplots(figsize=(18, 10), dpi=150) # 增大 figsize
            fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1) # 調整邊距
            
            undeformed_segments = []
            deformed_segments = []

            for ie in self.elements:
                # Corrected access to nodes using the 'nodes' list
                n1_pos = ie.nodes[0].position
                n2_pos = ie.nodes[1].position
                n3_pos = ie.nodes[2].position
                n4_pos = ie.nodes[3].position

                # Undeformed segments
                undeformed_segments.extend([
                    (n1_pos, n2_pos),
                    (n2_pos, n3_pos),
                    (n3_pos, n4_pos),
                    (n4_pos, n1_pos)
                ])

                # Deformed positions
                # Corrected access to nodes using the 'nodes' list
                n1_def = ie.nodes[0].position + ie.nodes[0].displacement * scale_factor
                n2_def = ie.nodes[1].position + ie.nodes[1].displacement * scale_factor
                n3_def = ie.nodes[2].position + ie.nodes[2].displacement * scale_factor
                n4_def = ie.nodes[3].position + ie.nodes[3].displacement * scale_factor

                # Deformed segments
                deformed_segments.extend([
                    (n1_def, n2_def),
                    (n2_def, n3_def),
                    (n3_def, n4_def),
                    (n4_def, n1_def)
                ])
            
            # Create LineCollections
            undeformed_lc = LineCollection(undeformed_segments, linewidths=0.8, colors='gray', label='Undeformed')
            deformed_lc = LineCollection(deformed_segments, linewidths=1.5, linestyles='--', colors='tab:red', label='Deformed')
            
            ax.add_collection(undeformed_lc)
            ax.add_collection(deformed_lc)

            # 增大標籤字體大小
            ax.set_xlabel("Length (m)", fontsize=30) # 增大字體
            ax.set_ylabel("Height (m)", fontsize=30) # 增大字體
            ax.set_aspect('equal', 'box')
            plt.tick_params(labelsize=25) # 保持刻度標籤字體大小

            # Manually set limits to ensure both undeformed and deformed fit nicely
            all_x_coords_orig = [node.position[0] for node in self.nodes]
            all_y_coords_orig = [node.position[1] for node in self.nodes]
            all_x_coords_def = [node.position[0] + node.displacement[0] * scale_factor for node in self.nodes]
            all_y_coords_def = [node.position[1] + node.displacement[1] * scale_factor for node in self.nodes]

            min_x = min(min(all_x_coords_orig), min(all_x_coords_def))
            max_x = max(max(all_x_coords_orig), max(all_x_coords_def))
            min_y = min(min(all_y_coords_orig), min(all_y_coords_def))
            max_y = max(max(all_y_coords_orig), max(all_y_coords_def))

            # Add more padding to the limits
            x_range = max_x - min_x
            y_range = max_y - min_y
            ax.set_xlim(min_x - 0.2 * x_range, max_x + 0.2 * x_range) # 增加邊距
            ax.set_ylim(min_y - 0.2 * y_range, max_y + 0.2 * y_range) # 增加邊距
            
            # Create a custom legend handle for LineCollection
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='gray', lw=0.8, label='Undeformed'),
                Line2D([0], [0], color='tab:red', lw=1.5, ls='--', label='Deformed')
            ]
            # ax.legend(handles=legend_elements, loc="upper left", fontsize=20, framealpha=1) # 增大圖例字體
            
            # 增大標題字體大小
            plt.title(f"Mesh Deformation (Scale Factor: {scale_factor})", fontsize=28) # 增大字體
            plt.show()

# Example usage for mesh_module.py (for self-testing the mesh generation part)
if __name__ == "__main__":
    
    L_test = 10.0
    h_test = 2.0
    Nx_test = 10
    Ny_test = 2

    print(f"--- Testing Mesh Generation ({Nx_test}x{Ny_test} elements) ---")
    test_mesh = Mesh(L_test, h_test, Nx_test, Ny_test)
    print(f"Total nodes: {test_mesh.tot_node_num}")
    print(f"Total elements: {test_mesh.tot_elem_num}")

    # Set some dummy displacements for visualization
    for i, node in enumerate(test_mesh.nodes):
        node.displacement[0] = 0.001 * np.sin(node.position[0] / L_test * np.pi) * node.position[1] / h_test
        node.displacement[1] = -0.005 * np.sin(node.position[0] / L_test * np.pi) * (1 - (node.position[1] / h_test)**2)

    test_mesh.plot_mesh(is_plot=True, scale_factor=1)