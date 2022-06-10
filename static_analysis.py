# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:23:59 2022

@author: user
"""

import numpy as np

class Strain_Displacement(object):
    def __init__(self):
        self.dN1xi  = 0
        self.dN1eta = 0
        self.dN2xi  = 0
        self.dN2eta = 0
        self.dN3xi  = 0
        self.dN3eta = 0 
        self.dN4xi  = 0
        self.dN4eta = 0
        
        self.J11    = 0
        self.J12    = 0
        self.J21    = 0
        self.J22    = 0
        self.det_J  = 0
        
        self.B = 0
        
    def local_dN(self, xi, eta):
        self.dN1xi  = 0.25 * (-1.0)   * (1.0-eta)
        self.dN1eta = 0.25 * (1.0-xi) * (-1.0)
        
        self.dN2xi  = 0.25 * (1.0)    * (1.0-eta) 
        self.dN2eta = 0.25 * (1.0+xi) * (-1.0)
        
        self.dN3xi  = 0.25 * (1.0)    * (1.0+eta)  
        self.dN3eta = 0.25 * (1.0+xi) * (1.0)  
        
        self.dN4xi  = 0.25 * (-1.0)   * (1.0+eta)
        self.dN4eta = 0.25 * (1.0-xi) * (1.0)
        
    def Jacobian(self, x, y):
        x1, x2, x3, x4 = x
        y1, y2, y3, y4 = y
        self.J11 = x1*self.dN1xi  + x2*self.dN2xi  + x3*self.dN3xi  + x4*self.dN4xi
        self.J12 = y1*self.dN1xi  + y2*self.dN2xi  + y3*self.dN3xi  + y4*self.dN4xi
        self.J21 = x1*self.dN1eta + x2*self.dN2eta + x3*self.dN3eta + x4*self.dN4eta
        self.J22 = y1*self.dN1eta + y2*self.dN2eta + y3*self.dN3eta + y4*self.dN4eta
        self.det_J = self.J11*self.J22-self.J12*self.J21
        
    def matrix(self):
        det_J_inv = 1.0/self.det_J
        dN1x = det_J_inv * ( self.J22*self.dN1xi - self.J12*self.dN1eta )
        dN1y = det_J_inv * (-self.J21*self.dN1xi + self.J11*self.dN1eta )
        dN2x = det_J_inv * ( self.J22*self.dN2xi - self.J12*self.dN2eta )
        dN2y = det_J_inv * (-self.J21*self.dN2xi + self.J11*self.dN2eta )
        dN3x = det_J_inv * ( self.J22*self.dN3xi - self.J12*self.dN3eta )
        dN3y = det_J_inv * (-self.J21*self.dN3xi + self.J11*self.dN3eta )
        dN4x = det_J_inv * ( self.J22*self.dN4xi - self.J12*self.dN4eta )
        dN4y = det_J_inv * (-self.J21*self.dN4xi + self.J11*self.dN4eta )
        
        self.B = np.array([[dN1x,  0.0, dN2x,  0.0, dN3x,  0.0, dN4x,  0.0],
                           [0.0,  dN1y,  0.0, dN2y,  0.0, dN3y,  0.0, dN4y],
                           [dN1y, dN1x, dN2y, dN2x, dN3y, dN3x, dN4y, dN4x]])


class Shape_Function(Strain_Displacement):
    def __init__(self):
        super().__init__()
        self.N1 = 0
        self.N2 = 0
        self.N3 = 0
        self.N4 = 0
        self.N  = 0
    
    def local_N(self, xi, eta):
        self.N1 = 0.25 * (1-xi) * (1-eta)
        self.N2 = 0.25 * (1+xi) * (1-eta)
        self.N3 = 0.25 * (1+xi) * (1+eta)
        self.N4 = 0.25 * (1-xi) * (1+eta)
        
    def matrix(self):
        self.N = np.array([[self.N1, 0.0, self.N2, 0.0, self.N3, 0.0, self.N4, 0.0],
                           [0.0, self.N1, 0.0, self.N2, 0.0, self.N3, 0.0, self.N4]])


class Calculate(object):  
    def __init__(self, msh):
        double_tot_node_num = int(msh.tot_node_num*2)
        self.msh = msh
        self.M = np.zeros((double_tot_node_num, double_tot_node_num))
        self.K = np.zeros((double_tot_node_num, double_tot_node_num))
        self.F = np.zeros(double_tot_node_num)
        
        self.pid = None
        self.P = None
        
    def apply_info(self, pid, P):
        self.pid = pid
        self.P = P
            
    def single_elem_matrix(self, elastic):
        
        ''' strain-displacement matrix '''
        rho = elastic.rho
        D = elastic.D
        w = elastic.w
        A = elastic.A
        
        ieo = self.msh.elements[0]
        x1, y1 = ieo.n1.position[0], ieo.n1.position[1]
        x2, y2 = ieo.n2.position[0], ieo.n2.position[1]
        x3, y3 = ieo.n3.position[0], ieo.n3.position[1]
        x4, y4 = ieo.n4.position[0], ieo.n4.position[1]

        # w1 = w2 = 1.0 use 2 Gauss points
        eu = Strain_Displacement()
        sh = Shape_Function()
        eta_arr = [-1/np.sqrt(3), 1/np.sqrt(3)]
        xi_arr = [-1/np.sqrt(3), 1/np.sqrt(3)]
        for eta in eta_arr:
            for xi in xi_arr:
                eu.local_dN(xi, eta)
                eu.Jacobian([x1, x2, x3, x4], [y1, y2, y3, y4])
                eu.matrix()
                det_J = eu.det_J
                B = eu.B

                sh.local_N(xi, eta)
                sh.matrix()
                N = sh.N

                ieo.ke += np.dot(np.dot(B.T, D), B)*det_J*w
                ieo.me += rho*A*(N.T).dot(N)*det_J
                
    def node_f_ext(self, ti):
        ''' apply force in node '''
        if (self.pid > self.msh.tot_node_num):
            from sys import exit
            print("reset external force id")
            exit()
        
        for node in self.msh.nodes:
            node.f_ext = np.array([0.0, 0.0])
            if ((node.nid == self.pid) and (ti == 0.0)):
                node.f_ext[0] = self.P[0]
                node.f_ext[1] = self.P[1]
         
    def set_M_K_F(self):
        ''' calculate K matrix and F vector '''
        ieo = self.msh.elements[0]
        for ie in self.msh.elements:
            local_f = [ie.n1.f_ext[0], ie.n1.f_ext[1], ie.n2.f_ext[0], ie.n2.f_ext[1], ie.n3.f_ext[0], ie.n3.f_ext[1], ie.n4.f_ext[0], ie.n4.f_ext[1]]
            gid = [ie.n1.gid[0], ie.n1.gid[1], ie.n2.gid[0], ie.n2.gid[1], ie.n3.gid[0], ie.n3.gid[1], ie.n4.gid[0], ie.n4.gid[1]]
            
            for i in range(8):
                self.F[gid[i]] = local_f[i]
                
                for j in range(8):
                    self.M[gid[i], gid[j]] += ieo.me[i, j]
                    self.K[gid[i], gid[j]] += ieo.ke[i, j]
                    
    def get_mass_matrix(self):
        return self.M 
    
    def get_stiffness_matrix(self):
        return self.K

    def get_force_vector(self):
        return self.F


class Static_Solver(object):

    def __init__(self):
        # c -> constrain dof
        # f -> free dof
        self.M_ff = 0
        self.K_ff = 0
        self.F_f = 0
        
        self.bid_list = None # boundary id for nid
        self.dof_id = None   # global id for x and y direction
        self.len_bid = 0
        self.free_dof_num = 0
        self.constraint_dof_num = 0
        self.tot_dof_num = 0
        
        self.global_U = 0
        self.U = 0

    def apply_BC(self, msh, bid_list):
        self.tot_dof_num = int(msh.tot_node_num*2)
        self.bid_list = bid_list
        self.dof_id = np.zeros(self.tot_dof_num, dtype=int)
        self.nid_array = np.zeros((msh.tot_node_num), dtype=int)
        
        self.len_bid = self.bid_list.__len__()
        if (self.len_bid != 0):

            i = 0
            bid_list.sort()
            for bid in bid_list:
                self.nid_array[i] = bid
                i += 1
                
            m = self.len_bid
            for i in range(msh.tot_node_num):
                num = 0
                for j in range(self.len_bid):
                    if (i == self.nid_array[j]):
                        num += 1
                if (num == 0):
                    self.nid_array[m] = i
                    m += 1

            m = 0
            for i in self.nid_array:
                for node in msh.nodes:
                    if (i == node.nid):
                        self.dof_id[m] = node.gid[0]
                        self.dof_id[m+1] = node.gid[1]
                        m += 2
        else:
            from sys import exit
            print("No Boundary Condition")
            exit()

    def rearrange(self, msh, global_M, global_K, global_F):
        if (self.len_bid != 0):
            self.free_dof_num = int(np.sqrt(int(self.tot_dof_num*self.tot_dof_num)-int(self.len_bid*2*(self.tot_dof_num*2-self.len_bid*2))))
            self.constraint_dof_num = int(self.tot_dof_num-self.free_dof_num)
            self.M_ff = np.zeros((self.free_dof_num, self.free_dof_num))
            self.K_ff = np.zeros((self.free_dof_num, self.free_dof_num))
            self.F_f = np.zeros(self.free_dof_num)
        
            for i in range(self.free_dof_num):
                self.F_f[i] = global_F[self.dof_id[i+self.constraint_dof_num]]
                
                for j in range(self.free_dof_num):
                    self.M_ff[i, j] = global_M[self.dof_id[i+self.constraint_dof_num], self.dof_id[j+self.constraint_dof_num]]
                    self.K_ff[i, j] = global_K[self.dof_id[i+self.constraint_dof_num], self.dof_id[j+self.constraint_dof_num]]
            
    def update_msh_disp(self, msh):
        if (self.len_bid != 0):
            self.global_U = np.zeros(self.tot_dof_num)
            inv_K_ff = np.linalg.inv(self.K_ff)
            self.u_f = inv_K_ff.dot(self.F_f)
            for i in range(self.free_dof_num):
                self.global_U[self.dof_id[i+self.constraint_dof_num]] = self.u_f[i]
                
            i = 0
            for node in msh.nodes:
                node.displacement[0] = self.global_U[i]
                node.displacement[1] = self.global_U[i+1]
                i += 2
    
    def get_dof_id(self):
        return self.dof_id
    
    def get_free_node_num(self):
        return self.free_dof_num
    
    def get_constraint_dof_num(self):
        return self.constraint_dof_num
    
    def get_tot_dof_num(self):
        return self.tot_dof_num
    
    def get_mass_free(self):
        return self.M_ff
    
    def get_stiffness_free(self):
        return self.K_ff
    
    def get_force_free(self):
        return self.F_f
    
    def get_disp_free(self):
        return self.u_f
    
    def get_disp_vector(self, msh):
        self.U = self.global_U.reshape(msh.tot_node_num, 2)
        return self.U


if __name__ == "__main__":
    
    from time import time
    from mesh import Mesh, Material
    
    t0 = time()
    
    # set model -> plane stress
    L = 10.0 # beam length
    h = 2.0  # beam width
    w = 0.5  # beam thickness
    Nx = 10
    Ny = 2
    
    # set material
    rho = 960.0
    E = 432e+6
    v = 0.3
    
    # set external force with [apply time] and [node-ID] and [vector]
    pid = 16 
    P = [0.0, -1000.0] 
    
    # Boundary ID
    bid_list=[0, 30]
    
    elastic = Material(rho, E, v)
    elastic.cross_section(h, w)
    msh = Mesh(L, h, Nx, Ny)

    cal = Calculate(msh)
    cal.apply_info(pid, P)
    cal.single_elem_matrix(elastic)
    cal.node_f_ext(ti=0.0)
    cal.set_M_K_F()
    M = cal.get_mass_matrix()
    K = cal.get_stiffness_matrix()
    F = cal.get_force_vector()
    
    sol = Static_Solver()
    sol.apply_BC(msh, bid_list) # simply supported beam
    sol.rearrange(msh, M, K, F)
    sol.update_msh_disp(msh)
    U = sol.get_disp_vector(msh)
    
    msh.plot_fig(is_plot=False)

    print("numerical solution =", round(msh.nodes[pid].displacement[1], 8), '\n')
    print(["time-used=", time()-t0])
    