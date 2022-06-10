# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 22:10:02 2022

@author: user
"""

import numpy as np

class Dynamics_Solver(object):
    def __init__(self, tns, Static_Sol):
        self.tns = tns
        self.dt  = tns[1]-tns[0]
        self.dof_id             = StaSol.get_dof_id()
        self.free_dof_num       = Static_Sol.get_free_node_num()
        self.constraint_dof_num = Static_Sol.get_constraint_dof_num()
        self.tot_dof_num        = StaSol.get_tot_dof_num()
        
        self.len_tns = len(self.tns)
        self.u = [0]*self.len_tns
        self.v = [0]*self.len_tns
        self.a = [0]*self.len_tns
        
        self.M_inv = 0
        self.K_inv = 0
        self.M_mK  = 0
        self.M_mF  = 0

    def set_Minv_K(self, M_ff, K_ff):
        self.M_inv = np.linalg.inv(M_ff)
        self.M_mK = np.dot(self.M_inv, K_ff)
        
    def set_K_inv(self, K_ff):
        self.K_inv = np.linalg.inv(K_ff)

    def set_Minv_F(self, F_f):
        self.M_mF = np.dot(self.M_inv, F_f)
    
    @staticmethod
    def node_f_ext(t, msh, cal, ti):
        ''' apply force in node '''
        for node in msh.nodes:
            node.f_ext = np.array([0.0, 0.0])
            if ((node.nid == cal.pid) and (t == ti)):
                node.f_ext[0] = cal.P[0]
                node.f_ext[1] = cal.P[1]
    
    def cal_Force_vector(self, t, ti, msh):
        global_F = np.zeros(self.tot_dof_num)
        if (t == ti):
            ''' calculate K matrix and F vector '''
            for ie in msh.elements:
                local_f = [ie.n1.f_ext[0], ie.n1.f_ext[1], ie.n2.f_ext[0], ie.n2.f_ext[1], ie.n3.f_ext[0], ie.n3.f_ext[1], ie.n4.f_ext[0], ie.n4.f_ext[1]]
                gid = [ie.n1.gid[0], ie.n1.gid[1], ie.n2.gid[0], ie.n2.gid[1], ie.n3.gid[0], ie.n3.gid[1], ie.n4.gid[0], ie.n4.gid[1]]
                
                for i in range(8):
                    global_F[gid[i]] = local_f[i]
        return global_F
    
    def cal_Force_free(self, t, ti, global_F):
        F_f = np.zeros(self.free_dof_num)
        if (t == ti):
            for i in range(self.free_dof_num):
                F_f[i] = global_F[self.dof_id[i+self.constraint_dof_num]]
        return F_f

    def get_K_inv(self):
        return self.K_inv

    def get_Minv_K(self):
        return self.M_mK
    
    def get_Minv_F(self, F_f):
        self.set_Minv_F(F_f)
        return self.M_mF
    
    def data_base(self):
        for i in range(self.len_tns):
            self.u[i] = np.zeros(self.free_dof_num)
            self.v[i] = np.zeros(self.free_dof_num)
            self.a[i] = np.zeros(self.free_dof_num)
            
    def modal_analysis(self):
        A = self.get_Minv_K()
        value, vector = np.linalg.eig(A)
        eigval = np.sort(value)
        eigIndex = np.argsort(value)
        eigvec = vector[:,eigIndex]
        wn = np.zeros(len(eigval))
        for i in range(len(eigval)):
            wn[i] = eigval[i].real**0.5
        
        T1 = 2.0*np.pi/wn[0]
        print("T1 =", T1)
        
    def integral(self, i, F_f):
        M_mF = self.get_Minv_F(F_f)
        M_Ku = np.dot(self.get_Minv_K(), self.u[i])
        for j in range(self.free_dof_num):
            if (i == 0):
                self.u[i][j] = u_f[j]
                self.v[i][j] = 0.0
                
            self.a[i][j] = M_mF[j]-M_Ku[j]
            self.v[i+1][j] = self.v[i][j] + self.dt*self.a[i][j]
            self.u[i+1][j] = self.u[i][j] + self.dt*self.v[i+1][j]

    def update_msh_disp(self, i, msh, F_f):
        global_U = np.zeros(self.tot_dof_num)
        for j in range(self.free_dof_num):
            global_U[self.dof_id[j+self.constraint_dof_num]] = self.u[i][j]
        
        i = 0
        for node in msh.nodes:
            node.displacement[0] += global_U[i]
            node.displacement[1] += global_U[i+1]
            i += 2

def output(i, msh, tns, k):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    plt.ion()    
    n3x = msh.elements[0].n3.position
    n4x = msh.elements[0].n4.position
    ax.plot(n3x[0], n3x[1], "-", lw=1, color="gray", label="undeformed")
    ax.plot(n3x[0], n3x[1], "--", lw=1, color="tab:red", label="deformed")
                
    for ie in msh.elements:
        n1x = ie.n1.position
        n2x = ie.n2.position
        n3x = ie.n3.position
        n4x = ie.n4.position
        ax.plot([n1x[0], n2x[0]], [n1x[1], n2x[1]], "-", lw=1, color="gray")
        ax.plot([n3x[0], n4x[0]], [n3x[1], n4x[1]], "-", lw=1, color="gray")
        ax.plot([n1x[0], n4x[0]], [n1x[1], n4x[1]], "-", lw=1, color="gray")
        ax.plot([n2x[0], n3x[0]], [n2x[1], n3x[1]], "-", lw=1, color="gray")
        
        ax.plot([n1x[0]+ie.n1.displacement[0], n2x[0]+ie.n2.displacement[0]], [n1x[1]+ie.n1.displacement[1], n2x[1]+ie.n2.displacement[1]], "--", lw=1, color="tab:red")
        ax.plot([n3x[0]+ie.n3.displacement[0], n4x[0]+ie.n4.displacement[0]], [n3x[1]+ie.n3.displacement[1], n4x[1]+ie.n4.displacement[1]], "--", lw=1, color="tab:red")
        ax.plot([n1x[0]+ie.n1.displacement[0], n4x[0]+ie.n4.displacement[0]], [n1x[1]+ie.n1.displacement[1], n4x[1]+ie.n4.displacement[1]], "--", lw=1, color="tab:red")
        ax.plot([n2x[0]+ie.n2.displacement[0], n3x[0]+ie.n3.displacement[0]], [n2x[1]+ie.n2.displacement[1], n3x[1]+ie.n3.displacement[1]], "--", lw=1, color="tab:red")
        
    ax.set_xlabel("Length (m)", fontsize=25)
    ax.set_ylabel("Height (m)", fontsize=25)
    ax.set_xlim(-1.0, 11.0)
    ax.set_ylim(-1.1, 3.1)
    
    ax.set_title("t = %.2f (s)"%tns[i], fontsize=25)
    ax.legend(loc="upper left", fontsize=15, framealpha=1)
    ax.set_aspect('equal', 'box')
    plt.tick_params(labelsize=25)
    plt.ioff()
    plt.savefig(r"D:\有限元素法\Final Report\fig_store\%07d_deformed.png"%(i+1+k))
    plt.close("all")

def is_out_fig(is_plot):
    if (is_plot):
        if (i % 10 == 0):
            output(i, msh, tns, k=0)
                    
        if (i == (DynSol.len_tns-1-1)):
            for k in range(5):
                output(i, msh, tns, k)      

if __name__ == "__main__":
    
    from sys import exit
    from time import time
    from mesh import Mesh, Material
    import matplotlib.pyplot as plt
    from static_analysis import Calculate, Static_Solver
    
    t0 = time()
    
    # set model -> plane stress
    L = 10.0 # beam length
    h = 2.0  # beam width
    w = 0.5  # beam thickness
    Nx = 10
    Ny = 2

    # set material
    rho = 2300.0
    E = 432e+6
    v = 0.3

    #set run time
    EndTime = 3.0
    dt = 1e-3

    # set external force with [apply time] and [node-ID] and [vector]
    ti = 0.0
    pid = 16 
    P = [0.0, -1e+5] 

    # Boundary ID
    bid_list=[0, 30]

    elastic = Material(rho, E, v)
    elastic.cross_section(h, w)
    msh = Mesh(L, h, Nx, Ny)

    cal = Calculate(msh)
    cal.apply_info(pid, P)
    cal.single_elem_matrix(elastic)
    cal.node_f_ext(ti)
    cal.set_M_K_F()
    M = cal.get_mass_matrix()
    K = cal.get_stiffness_matrix()
    F = cal.get_force_vector()

    StaSol = Static_Solver()
    StaSol.apply_BC(msh, bid_list) # simply supported beam
    StaSol.rearrange(msh, M, K, F)
    StaSol.update_msh_disp(msh)
    U = StaSol.get_disp_vector(msh)

    M_ff = StaSol.get_mass_free()
    K_ff = StaSol.get_stiffness_free()
    F_f = StaSol.get_force_free()
    u_f = StaSol.get_disp_free()

    tns = np.arange(0, EndTime+dt/10.0, dt)
    DynSol = Dynamics_Solver(tns, StaSol)
    DynSol.set_Minv_K(M_ff, K_ff)
    DynSol.set_K_inv(K_ff)
    DynSol.data_base()
    DynSol.modal_analysis()
    if (ti == 0.0):
        print("numerical solution =", round(msh.nodes[pid].displacement[1], 8), '\n')
    
    # exit()
    is_plot = True
    plt.rcParams["font.family"] = "Times New Roman"
    for i in range(DynSol.len_tns-1): # 
        t = tns[i]
        if (i != 0):
            DynSol.node_f_ext(t, msh, cal, ti)
            F = DynSol.cal_Force_vector(t, ti, msh)
            F_f = DynSol.cal_Force_free(t, ti, F)
        DynSol.integral(i, F_f)
        DynSol.update_msh_disp(i, msh, F_f)
        
        is_out_fig(is_plot)

    print(["time-used=", time()-t0])
    
    