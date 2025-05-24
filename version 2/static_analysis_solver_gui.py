import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox

# Import core FEM classes and functions from external modules
# Make sure mesh_v2.py and static_v2.py are in the same directory or accessible via PYTHONPATH
# Note: Renamed mesh.py to mesh_v2.py and static_analysis.py to static_v2.py for consistency
from mesh_v2 import Material, Node, Element, Mesh
import static_v2 # Import the entire static_v2 module

class FEMGUI:
    def __init__(self, master):
        self.master = master
        master.title("FEM Beam Analysis")

        # Default parameters
        self.beam_length = 1000.0
        self.beam_height = 20.0
        self.num_elements_x = 100
        self.num_elements_y = 2
        self.rho_steel = 7.85e-9
        self.E_steel = 210000.0
        self.v_steel = 0.3 
        self.beam_thickness = 10.0

        self.my_mesh = None
        self.steel_material = None
        self.fem_solver = None
        self.element_stress_results = [] # To store stress results for later use (e.g., contour)

        self.create_widgets()
        self.generate_mesh_and_plot() # Initial mesh generation on startup

    def create_widgets(self):
        # Frame for parameters
        param_frame = ttk.LabelFrame(self.master, text="Parameters")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Geometry Inputs
        ttk.Label(param_frame, text="Beam Length (mm):").grid(row=0, column=0, sticky="w", pady=2)
        self.length_entry = ttk.Entry(param_frame)
        self.length_entry.insert(0, str(self.beam_length))
        self.length_entry.grid(row=0, column=1, pady=2)

        ttk.Label(param_frame, text="Beam Height (mm):").grid(row=1, column=0, sticky="w", pady=2)
        self.height_entry = ttk.Entry(param_frame)
        self.height_entry.insert(0, str(self.beam_height))
        self.height_entry.grid(row=1, column=1, pady=2)
        
        ttk.Label(param_frame, text="Beam Thickness (mm):").grid(row=2, column=0, sticky="w", pady=2)
        self.thickness_entry = ttk.Entry(param_frame)
        self.thickness_entry.insert(0, str(self.beam_thickness))
        self.thickness_entry.grid(row=2, column=1, pady=2)

        ttk.Label(param_frame, text="Elements X:").grid(row=3, column=0, sticky="w", pady=2)
        self.nx_entry = ttk.Entry(param_frame)
        self.nx_entry.insert(0, str(self.num_elements_x))
        self.nx_entry.grid(row=3, column=1, pady=2)

        ttk.Label(param_frame, text="Elements Y:").grid(row=4, column=0, sticky="w", pady=2)
        self.ny_entry = ttk.Entry(param_frame)
        self.ny_entry.insert(0, str(self.num_elements_y))
        self.ny_entry.grid(row=4, column=1, pady=2)

        # Material Inputs
        ttk.Label(param_frame, text="Young's Modulus (MPa):").grid(row=5, column=0, sticky="w", pady=2)
        self.e_entry = ttk.Entry(param_frame)
        self.e_entry.insert(0, str(self.E_steel))
        self.e_entry.grid(row=5, column=1, pady=2)

        ttk.Label(param_frame, text="Poisson's Ratio:").grid(row=6, column=0, sticky="w", pady=2)
        self.v_entry = ttk.Entry(param_frame)
        self.v_entry.insert(0, str(self.v_steel))
        self.v_entry.grid(row=6, column=1, pady=2)

        ttk.Button(param_frame, text="Generate Mesh", command=self.generate_mesh_and_plot).grid(row=7, column=0, columnspan=2, pady=10)

        # Node selection by ID (New Feature)
        node_select_frame = ttk.LabelFrame(param_frame, text="Select Node by ID")
        node_select_frame.grid(row=8, column=0, columnspan=2, padx=5, pady=10, sticky="ew")

        ttk.Label(node_select_frame, text="Node ID:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.node_id_entry = ttk.Entry(node_select_frame, width=10)
        self.node_id_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Button(node_select_frame, text="Select by ID", command=self.select_node_by_id).grid(row=1, column=0, columnspan=2, pady=5)


        # --- Plotting Area ---
        # 調整 figsize 讓繪圖區域更大
        self.fig, self.ax = plt.subplots(figsize=(14, 9), dpi=100) # Adjusted for larger plot
        self.ax.set_aspect('equal', 'box')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.master)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Bind click event for node selection
        self.canvas.mpl_connect("button_press_event", self.on_mesh_click)

        # --- Controls and Results Frame ---
        control_frame = ttk.LabelFrame(self.master, text="Controls & Results")
        control_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Action Buttons
        ttk.Button(control_frame, text="Reset BCs & Forces", command=self.reset_model).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis).pack(side=tk.LEFT, padx=5, pady=5)
        # New button for showing max displacement
        ttk.Button(control_frame, text="Show Max Displacement", command=self.show_max_displacement).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.scale_factor_label = ttk.Label(control_frame, text="Scale Factor:")
        self.scale_factor_label.pack(side=tk.LEFT, padx=5, pady=5)
        self.scale_factor_entry = ttk.Entry(control_frame, width=8)
        self.scale_factor_entry.insert(0, "100") # Default scale factor for deformed plot
        self.scale_factor_entry.pack(side=tk.LEFT, padx=5, pady=5)

        # Results Display
        self.results_text = tk.Text(control_frame, height=8, width=70)
        self.results_text.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.results_text.insert(tk.END, "FEM Results will appear here...\n")

    def generate_mesh_and_plot(self):
        try:
            self.beam_length = float(self.length_entry.get())
            self.beam_height = float(self.height_entry.get())
            self.beam_thickness = float(self.thickness_entry.get())
            self.num_elements_x = int(self.nx_entry.get())
            self.num_elements_y = int(self.ny_entry.get())
            self.E_steel = float(self.e_entry.get())
            self.v_steel = float(self.v_entry.get())

            if self.num_elements_x <= 0 or self.num_elements_y <= 0:
                raise ValueError("Number of elements must be positive.")
            if self.beam_length <= 0 or self.beam_height <= 0 or self.beam_thickness <= 0:
                raise ValueError("Dimensions must be positive.")
            if self.E_steel <= 0:
                raise ValueError("Young's Modulus must be positive.")
            if not (0 <= self.v_steel < 0.5):
                raise ValueError("Poisson's Ratio must be between 0 and 0.5.")

        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        self.my_mesh = Mesh(self.beam_length, self.beam_height, self.num_elements_x, self.num_elements_y)
        self.steel_material = Material(rho=self.rho_steel, E=self.E_steel, poisson=self.v_steel)
        self.steel_material.cross_section(H=self.beam_height, W=self.beam_thickness)

        # Reset all nodes to default (no constraints, no forces) when mesh is regenerated
        for node in self.my_mesh.nodes:
            node.is_constrained = [False, False]
            node.constrained_value = [0.0, 0.0]
            node.f_ext = np.zeros(2)
            node.displacement = np.zeros(2) # Clear previous displacements

        self.plot_mesh() # Plot initial undeformed mesh
        self.results_text.insert(tk.END, f"Mesh generated with {self.my_mesh.tot_node_num} nodes and {self.my_mesh.tot_elem_num} elements.\n")
        self.results_text.insert(tk.END, f"Material: E={self.steel_material.E/1e3:.1f} GPa, v={self.steel_material.v}, t={self.steel_material.w} mm\n")
        self.results_text.see(tk.END)


    def plot_mesh(self, deformed=False, scale_factor=1.0):
        self.ax.clear()
        
        # Plot nodes
        for node in self.my_mesh.nodes:
            # Determine the position to plot based on 'deformed' flag
            pos_to_plot = node.position
            if deformed:
                # Add displacement with scale factor for deformed plot
                pos_to_plot = node.position + node.displacement * scale_factor

            self.ax.plot(pos_to_plot[0], pos_to_plot[1], 'o', markersize=3, color='blue')

            # Mark boundary conditions and forces
            # These markers should always be based on the original position,
            # as BCs and forces are applied to the undeformed structure.
            pos_bc_force = node.position

            if node.is_constrained[0] and node.is_constrained[1]:
                self.ax.plot(pos_bc_force[0], pos_bc_force[1], 's', markersize=8, color='red', label='Pinned' if not self.ax.get_legend_handles_labels()[1].count('Pinned') else "")
            elif node.is_constrained[1]:
                self.ax.plot(pos_bc_force[0], pos_bc_force[1], '^', markersize=8, color='green', label='Roller' if not self.ax.get_legend_handles_labels()[1].count('Roller') else "")
            
            if np.any(node.f_ext != 0):
                # Scale arrow for visualization, might need adjustment based on max force and beam size
                arrow_scale = self.beam_length * 0.05 / max(1, np.linalg.norm(node.f_ext)) 
                self.ax.arrow(pos_bc_force[0], pos_bc_force[1], 
                              node.f_ext[0] * arrow_scale, node.f_ext[1] * arrow_scale,
                              head_width=self.beam_length*0.01, head_length=self.beam_length*0.015, fc='purple', ec='purple', 
                              label='Force' if not self.ax.get_legend_handles_labels()[1].count('Force') else "")

        # Adjust plot limits to focus on the beam
        padding = self.beam_height * 0.2  # Add some padding around the beam
        self.ax.set_xlim(-padding, self.beam_length + padding)
        self.ax.set_ylim(-padding, self.beam_height + padding)

        self.ax.set_xlabel("Length (mm)", fontsize=12) # Adjusted font size
        self.ax.set_ylabel("Height (mm)", fontsize=12) # Adjusted font size
        self.ax.set_title("Beam and Boundary Conditions", fontsize=14) # Adjusted title font size
        self.ax.set_aspect('equal', 'box')
        self.ax.legend(loc="upper left", fontsize=10) # Adjusted legend font size
        self.canvas.draw()

    def on_mesh_click(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        
        # Find the closest node
        closest_node = None
        min_dist = float('inf')
        
        # Adjust picking tolerance dynamically or based on average element size
        avg_elem_dim = max(self.beam_length/self.num_elements_x, self.beam_height/self.num_elements_y)
        picking_tolerance = avg_elem_dim * 0.25 # Smaller tolerance for dense meshes

        for node in self.my_mesh.nodes:
            dist = np.sqrt((node.position[0] - x)**2 + (node.position[1] - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_node = node
        
        if closest_node and min_dist < picking_tolerance: # Use adjusted threshold
            self.show_node_options(closest_node)
        else:
            self.results_text.insert(tk.END, "No node selected within tolerance.\n")
            self.results_text.see(tk.END)

    def select_node_by_id(self):
        if self.my_mesh is None:
            messagebox.showwarning("Warning", "Please generate mesh first.")
            return

        try:
            node_id_to_select = int(self.node_id_entry.get())
            if 0 <= node_id_to_select < self.my_mesh.tot_node_num:
                selected_node = self.my_mesh.nodes[node_id_to_select]
                self.show_node_options(selected_node)
            else:
                messagebox.showerror("Error", f"Node ID {node_id_to_select} is out of bounds (0 to {self.my_mesh.tot_node_num - 1}).")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid integer for Node ID.")

    def show_node_options(self, node):
        dialog = tk.Toplevel(self.master)
        dialog.title(f"Node {node.nid} Options")
        dialog.transient(self.master)
        dialog.grab_set() 

        ttk.Label(dialog, text=f"Node ID: {node.nid} (Pos: {node.position[0]:.2f}, {node.position[1]:.2f})").pack(pady=5)

        current_bc_text = "Current BC: "
        if node.is_constrained[0] and node.is_constrained[1]: current_bc_text += "Pinned"
        elif node.is_constrained[1]: current_bc_text += "Roller (Y-fixed)"
        else: current_bc_text += "Free"
        ttk.Label(dialog, text=current_bc_text).pack(pady=2)

        current_force_text = f"Current Force: [{node.f_ext[0]:.2f}, {node.f_ext[1]:.2f}] N"
        ttk.Label(dialog, text=current_force_text).pack(pady=2)

        # BC options
        ttk.Label(dialog, text="Set Boundary Condition:").pack(pady=5)
        ttk.Button(dialog, text="No Constraint (Free)", command=lambda: self.set_bc(node, "free", dialog)).pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(dialog, text="Pinned (Fixed X, Y)", command=lambda: self.set_bc(node, "pinned", dialog)).pack(fill=tk.X, padx=10, pady=2)
        ttk.Button(dialog, text="Roller (Fixed Y)", command=lambda: self.set_bc(node, "roller", dialog)).pack(fill=tk.X, padx=10, pady=2)

        # Force options
        ttk.Label(dialog, text="Apply External Force (N):").pack(pady=5)
        force_x_var = tk.DoubleVar(value=node.f_ext[0])
        force_y_var = tk.DoubleVar(value=node.f_ext[1])

        force_input_frame = ttk.Frame(dialog)
        force_input_frame.pack()
        ttk.Label(force_input_frame, text="Force X:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(force_input_frame, textvariable=force_x_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(force_input_frame, text="Force Y:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(force_input_frame, textvariable=force_y_var, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(dialog, text="Apply Force", 
                   command=lambda: self.apply_force(node, force_x_var.get(), force_y_var.get(), dialog)).pack(pady=5)

        # Close button
        ttk.Button(dialog, text="Done", command=dialog.destroy).pack(pady=10)

    def set_bc(self, node, bc_type, dialog):
        node.is_constrained = [False, False]
        node.constrained_value = [0.0, 0.0]

        if bc_type == "pinned":
            node.is_constrained[0] = True
            node.is_constrained[1] = True
            self.results_text.insert(tk.END, f"Node {node.nid}: Set as Pinned.\n")
        elif bc_type == "roller":
            node.is_constrained[1] = True
            self.results_text.insert(tk.END, f"Node {node.nid}: Set as Roller.\n")
        elif bc_type == "free":
            self.results_text.insert(tk.END, f"Node {node.nid}: Set as Free.\n")
        
        self.results_text.see(tk.END)
        self.plot_mesh() # Redraw to show BCs
        dialog.destroy()

    def apply_force(self, node, fx, fy, dialog):
        try:
            node.f_ext = np.array([float(fx), float(fy)])
            self.results_text.insert(tk.END, f"Node {node.nid}: Applied Force [{fx:.2f}, {fy:.2f}] N.\n")
            self.results_text.see(tk.END)
            self.plot_mesh() # Redraw to show forces
            dialog.destroy()
        except ValueError:
            messagebox.showerror("Input Error", "Force values must be numbers.")

    def reset_model(self):
        if self.my_mesh: # Ensure mesh exists before resetting
            for node in self.my_mesh.nodes:
                node.is_constrained = [False, False]
                node.constrained_value = [0.0, 0.0]
                node.f_ext = np.zeros(2)
                node.displacement = np.zeros(2) # Clear previous displacements
            self.plot_mesh()
        self.results_text.delete('1.0', tk.END)
        self.results_text.insert(tk.END, "Model reset. All BCs and Forces cleared.\n")
        self.results_text.see(tk.END)

    def run_analysis(self):
        if self.my_mesh is None or self.steel_material is None:
            messagebox.showerror("Error", "Please generate mesh first.")
            return

        self.results_text.insert(tk.END, "\n--- Running Static Analysis ---\n")
        self.results_text.see(tk.END)

        try:
            # Use the Calculate class from static_v2
            self.fem_solver = static_v2.Calculate(self.my_mesh, self.steel_material)
            K_global, M_global, F_global = self.fem_solver.assemble_global_matrices()
            K_modified, F_modified, constrained_dofs = self.fem_solver.apply_boundary_conditions(K_global, F_global)
            U_global = self.fem_solver.solve_static_system(K_modified, F_modified)

            if U_global is None or (U_global.size > 0 and np.all(U_global == 0)): 
                self.results_text.insert(tk.END, "Analysis failed or returned zero displacements. Check boundary conditions and forces.\n")
                self.results_text.see(tk.END)
                return

            # Update Node Displacements in the Mesh
            for node in self.my_mesh.nodes:
                node.displacement[0] = U_global[node.gid[0]]
                node.displacement[1] = U_global[node.gid[1]]

            self.results_text.insert(tk.END, "\n--- Selected Displacement Results ---\n")
            
            # Display displacements for some key nodes, e.g., pinned, roller, and force applied
            nodes_to_report = []
            for node in self.my_mesh.nodes:
                if (node.is_constrained[0] or node.is_constrained[1]) or np.any(node.f_ext != 0):
                    nodes_to_report.append(node)
            
            # Also add a mid-bottom node for deflection check
            mid_bottom_node = None
            for node in self.my_mesh.nodes:
                if np.isclose(node.position[0], self.beam_length / 2) and np.isclose(node.position[1], 0.0):
                    mid_bottom_node = node
                    break
            if mid_bottom_node and mid_bottom_node not in nodes_to_report:
                nodes_to_report.append(mid_bottom_node)

            # Sort by node ID for consistent output
            nodes_to_report.sort(key=lambda n: n.nid)

            for node in nodes_to_report:
                self.results_text.insert(tk.END, f"Node {node.nid} ({node.position[0]:.2f}, {node.position[1]:.2f}): u_x={node.displacement[0]:.6e}, u_y={node.displacement[1]:.6e}\n")
            self.results_text.see(tk.END)
            
            # Post-processing: Calculate Stresses and Strains (at element center)
            self.results_text.insert(tk.END, "\n--- Element Stresses and Strains (at element center) ---\n")
            self.element_stress_results = [] # Store for potential contour plotting
            for elem in self.my_mesh.elements:
                nodes_coords = np.array([node.position for node in elem.nodes])
                # Use functions imported from static_v2
                B_center, detJ_center = static_v2.get_B_matrix(nodes_coords, 0.0, 0.0)
                
                if detJ_center <= 1e-9:
                    # Handle cases where Jacobian determinant is zero or very small
                    # This might indicate a collapsed or inverted element, which can happen with highly distorted meshes
                    strain_e = np.array([np.nan, np.nan, np.nan])
                    stress_e = np.array([np.nan, np.nan, np.nan])
                    self.results_text.insert(tk.END, f"Warning: Element {elem.eid} has zero/negative Jacobian determinant. Stress/strain calculation skipped.\n")
                else:
                    element_dofs = []
                    for node in elem.nodes:
                        element_dofs.extend(node.gid) 
                    u_element_local = U_global[element_dofs] 
                    strain_e = B_center @ u_element_local # Corrected line
                    stress_e = self.steel_material.D @ strain_e       
                self.element_stress_results.append({
                    'eid': elem.eid,
                    'strain_xx': strain_e[0], 'strain_yy': strain_e[1], 'strain_xy': strain_e[2],
                    'stress_xx': stress_e[0], 'stress_yy': stress_e[1], 'stress_xy': stress_e[2]
                })

            valid_stress_xx = [res['stress_xx'] for res in self.element_stress_results if not np.isnan(res['stress_xx'])]
            if valid_stress_xx:
                self.results_text.insert(tk.END, f"Max Stress_xx: {np.max(valid_stress_xx):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Min Stress_xx: {np.min(valid_stress_xx):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Average Stress_xx: {np.mean(valid_stress_xx):.2f} MPa\n")
            
            valid_stress_yy = [res['stress_yy'] for res in self.element_stress_results if not np.isnan(res['stress_yy'])]
            if valid_stress_yy:
                self.results_text.insert(tk.END, f"Max Stress_yy: {np.max(valid_stress_yy):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Min Stress_yy: {np.min(valid_stress_yy):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Average Stress_yy: {np.mean(valid_stress_yy):.2f} MPa\n")
            
            valid_stress_xy = [res['stress_xy'] for res in self.element_stress_results if not np.isnan(res['stress_xy'])]
            if valid_stress_xy:
                self.results_text.insert(tk.END, f"Max Stress_xy: {np.max(valid_stress_xy):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Min Stress_xy: {np.min(valid_stress_xy):.2f} MPa\n")
                self.results_text.insert(tk.END, f"Average Stress_xy: {np.mean(valid_stress_xy):.2f} MPa\n")
            self.results_text.see(tk.END)
            
            try:
                scale_factor = float(self.scale_factor_entry.get())
            except ValueError:
                scale_factor = 1.0
                messagebox.showwarning("Input Error", "Invalid scale factor, using default 1.0")

            self.plot_mesh(deformed=True, scale_factor=scale_factor) # Plot deformed mesh
        
        except Exception as e:
            messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")
            self.results_text.insert(tk.END, f"Analysis Error: {e}\n")
            self.results_text.see(tk.END)


    def show_max_displacement(self):
        if self.my_mesh is None or not any(node.displacement.any() for node in self.my_mesh.nodes):
            messagebox.showwarning("Results Error", "Please run the analysis first to get displacement results.")
            return

        max_displacement_magnitude = 0.0
        node_with_max_disp = None

        for node in self.my_mesh.nodes:
            # Calculate the magnitude (Euclidean norm) of the displacement vector
            disp_magnitude = np.linalg.norm(node.displacement)
            if disp_magnitude > max_displacement_magnitude:
                max_displacement_magnitude = disp_magnitude
                node_with_max_disp = node

        if node_with_max_disp:
            self.results_text.insert(tk.END, "\n--- Max Displacement Result ---\n")
            self.results_text.insert(tk.END, f"Max Total Displacement: {max_displacement_magnitude:.6e} mm\n")
            self.results_text.insert(tk.END, f"  at Node ID: {node_with_max_disp.nid}\n")
            self.results_text.insert(tk.END, f"  Original Position: ({node_with_max_disp.position[0]:.2f}, {node_with_max_disp.position[1]:.2f}) mm\n")
            self.results_text.insert(tk.END, f"  Displacement (ux, uy): ({node_with_max_disp.displacement[0]:.6e}, {node_with_max_disp.displacement[1]:.6e}) mm\n")
            self.results_text.see(tk.END)
        else:
            self.results_text.insert(tk.END, "\nNo displacement found or analysis not run.\n")
            self.results_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = FEMGUI(root)
    root.mainloop()