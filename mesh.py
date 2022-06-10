# -*- coding: utf-8 -*-
"""
Created on Fri May 20 22:11:34 2022

@author: user
"""


import numpy as np

class Material(object):
    def __init__(self, rho, E, poisson):
        
        # material
        self.rho = rho
        self.E   = E
        self.v   = poisson
        
        # cross section
        self.h = 0
        self.w = 0
        self.A = 0
        self.I = 0
        
        E1 = self.E/(1-self.v*self.v)
        E2 = self.v*self.E/(1-self.v*self.v)
        self.G = self.E/(2*(1+self.v))
        self.D = np.array([[E1,    E2,    0.0],
                           [E2,    E1,    0.0],
                           [0.0 , 0.0, self.G]])
    
    def cross_section(self, H, W):
        self.h = H
        self.w = W
        self.A = W*H
        self.I = W*pow(H, 3)/12.0

class Node(object):
    def __init__(self):
        self.nid = 0
        self.gid = np.zeros((2), dtype=int)
        self.position = 0
        self.displacement = np.zeros(2)
        self.velocity = 0
        self.stress = 0
        self.strain = 0
        self.f_ext = 0

class Element(object):
    def __init__(self):
        self.eid = 0
        self.n1 = None
        self.n2 = None
        self.n3 = None
        self.n4 = None
        self.me = 0
        self.ke = 0
        
class Mesh(object):
    def __init__(self, L, h, Nx, Ny):
        self.tot_elem_num = int(Nx*Ny)
        self.tot_node_num = int((Nx+1)*(Ny+1))
        self.elements = [0]*self.tot_elem_num
        self.nodes = [0]*self.tot_node_num
        
        xn = np.linspace(0, L, Nx+1)
        yn = np.linspace(0, h, Ny+1)
        node_position = np.zeros((self.tot_node_num, 2))
        for i in range(Nx+1):
            for j in range(Ny+1):
                node_position[int(i*(Ny+1)+j), 0] = xn[i]
                node_position[int(i*(Ny+1)+j), 1] = yn[j]

        for i in range(self.tot_node_num):
            self.nodes[i] = Node()
            self.nodes[i].nid = i
            self.nodes[i].gid[0] = int((self.nodes[i].nid+1)*2-1-1)
            self.nodes[i].gid[1] = int((self.nodes[i].nid+1)*2-1)
            self.nodes[i].position = node_position[i]
        
        i = 0
        for e in range(self.tot_elem_num):
            if ((e % Ny == 0) and (e != 0)):
                i += 1
            self.elements[e] = Element()
            self.elements[e].eid = e
            self.elements[e].n1 = self.nodes[e+i]
            self.elements[e].n2 = self.nodes[int(e+Ny+1+i)]
            self.elements[e].n3 = self.nodes[int(e+Ny+2+i)]
            self.elements[e].n4 = self.nodes[int(e+1+i)]
    
    def plot_fig(self, is_plot=False):
        if (is_plot):
            
            import matplotlib.pyplot as plt
            
            plt.rcParams["font.family"] = "Times New Roman"
            fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
            
            n3x = self.elements[0].n3.position
            n4x = self.elements[0].n4.position
            ax.plot(n3x[0], n3x[1], "-", lw=1, color="gray", label="undeformed")
            ax.plot(n3x[0], n3x[1], "--", lw=1, color="tab:red", label="deformed")
            
            for ie in self.elements:
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
                
            ax.legend(loc="upper left", fontsize=15, framealpha=1)
            ax.set_aspect('equal', 'box')
            plt.tick_params(labelsize=25)
            plt.show()
            return ax
     

if __name__ == "__main__":
    
    L = 10.0
    h = 2.0
    Nx = 10
    Ny = 2
    
    rho = 960.0
    E = 432e+6
    v = 0.3
    
    elastic = Material(rho, E, v)
    
    msh = Mesh(L, h, Nx, Ny)
    msh.plot_fig(is_plot=True)
    
    
    