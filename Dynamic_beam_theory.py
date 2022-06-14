# -*- coding: utf-8 -*-
"""
Created on Thu May 26 12:17:25 2022

@author: user
"""

import numpy as np

class Theory(object):
    def __init__(self, rho, A, E, I, L):
        self.rho = rho
        self.A = A
        self.E = E
        self.I = I
        self.L = L
        self.pi = np.pi
        
        self.n = 0
        self.tau = 0
        self.c = 0
        self.beta = 0
        
        self.uo = 0
        self.vo = 0
        
        self.p = 0
        self.a = 0
 
    def set_IC(self, uo, vo):
        self.uo = uo
        self.vo = vo
    
    def set_mode(self, n):
        self.n = int(n)
    
    def set_tau(self, tau):
        self.tau = tau
        
    def set_c(self):
        self.c = pow(self.E*self.I/(self.rho*self.A), 0.5)
        
    def set_beta(self, n):
        beta_square = pow(n*self.pi/self.L, 4)
        self.beta = pow(beta_square, 0.5)
        
    def set_F_ext(self, p):
        self.p = p
        
    def set_a(self, a):
        self.a = a
        
    def get_uo(self):
        return self.uo
    
    def get_vo(self):
        return self.vo
    
    def get_mode(self):
        return self.n
        
    def get_tau(self):
        return self.tau
    
    def get_c(self):
        return self.c

    def get_beta(self, n):
        self.set_beta(n)
        return self.beta
    
    def get_pforce(self):
        return self.p
    
    def get_a(self):
        return self.a

    def H(self, t):
        if (t >= self.tau):
            return 1.0
        else:
            return 0.0
        
    def Tno(self, n):
        return (2.0*self.get_uo()/(n*self.pi))*(1.0-pow(-1.0, n))
    
    def dTno(self, n):
        return (2.0*self.get_vo()/(n*self.pi))*(1.0-pow(-1.0, n))

    def Tnf(self, n, t):
        p = self.get_pforce()
        a = self.get_a()
        c = self.get_c()
        beta = self.get_beta(n)
        
        T = self.H(t)*(2.0*p/(self.rho*self.A*self.L))*np.sin(n*self.pi*a/L)/(c*beta)
        T *= np.sin(c*beta*(t-self.get_tau()))
        T += self.Tno(n)*np.cos(c*beta*t)
        T += (self.dTno(n)/(c*beta))*np.sin(c*beta*t)
        return T
    
    def mode_shape(self, n, x):
        return np.sin(n*self.pi*x/self.L)

    # total mode +=
    def u(self, x, t):
        deflection = 0
        for n in range(1, self.get_mode()+1): # modal number
            Tn = self.Tnf(n, t)
            deflection += Tn*self.mode_shape(n, x)
        return deflection
    
def rede_data(path, Nt):
    disp = np.zeros(Nt-1)
    with open(path, 'r') as ifile:
        i = 0
        for text in ifile:
            text = text.split()
            disp[i] = float(text[3])
            i += 1
    return disp


if __name__ == "__main__":
    
    from sys import exit
    
    # set material
    E   = 432e+6
    I   = 1/3
    rho = 960.0
    A   = 1.0
    L   = 10.0
    Nx  = 1001

    # set I.C
    uo = 0.0
    vo = 0.0

    # set external force -> [time && magnitude && position]
    tau = 0.0
    p   = -1000.0
    a   = 0.5*L

    # set time
    endTime = 1.0
    Nt      = 10001
    
    # set mode
    nmode = 1

    sBeam = Theory(rho, A, E, I, L)
    sBeam.set_c()
    sBeam.set_IC(uo, vo)
    sBeam.set_tau(tau)
    sBeam.set_F_ext(p)
    sBeam.set_a(a)
    sBeam.set_mode(nmode)
    
    tns = np.linspace(0.0, endTime, Nt)
    u = np.zeros(Nt)
    x, i = a, 0
    for t in tns:
        u[i] = sBeam.u(x, t)
        i += 1
        
    # READ FEM DATA
    path = r"D:\有限元素法\Final Report\FEM DYNAMICS DATA.txt"
    ana_disp = rede_data(path, Nt)
   
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    
    # deflection in x=L/2 with time history
    plt.figure(0)
    plt.plot(tns, u, "-", lw=5, color="tab:blue", label="theoretical")
    plt.plot(tns[:Nt-1], ana_disp, "-", lw=5, color="tab:orange", label="numerical")
    plt.xlabel("time (sec)", fontsize=25)
    plt.ylabel("deflection (m)", fontsize=25)
    plt.title("x = 0.5L", fontsize=25)
    plt.tick_params(labelsize=25)
    plt.xlim(0, endTime)
    plt.legend(fontsize=25, loc="upper right", framealpha=1)
    plt.grid(True)
    plt.show()
    
    # simple beam Vibrational Modes, t=0
    t, nmode = 0.0, 5
    sBeam.set_IC(uo=1.0, vo=1.0)
    fig, ax = plt.subplots()
    x = np.linspace(0, L, Nx)
    for n in range(1, nmode+1):
        Vertical_Disp = np.zeros(Nx)
        sBeam.set_mode(n)
        print("w%d ="%n, sBeam.get_c()*sBeam.get_beta(n))
        print("T%d ="%n, 2.0*np.pi/(sBeam.get_c()*sBeam.get_beta(n)), '\n')
        for i in range(Nx):
            Vertical_Disp[i] = sBeam.mode_shape(n, x[i]) * (-1.0)
        ax.plot(x, Vertical_Disp, lw=7, label="n = %d"%(n))
    
        if (n == 1):
            ax.set_title("simple beam Vibrational Modes, t=0", fontsize=25)
        if (n == 3):
            ax.set_ylabel("Vertical Displacement (m)", fontsize=25)
        if (n == 5):
            ax.set_xlabel("x (m)", fontsize=25)
        
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([-1.0, 0.0, 1.0])
        ax.tick_params(labelsize=25)
        ax.ticklabel_format(style='sci', scilimits=(-0,0), axis='y')
        ax.yaxis.get_offset_text().set_fontsize(25)
        ax.set_xlim(0, L)
        ax.grid(True)
        ax.legend(fontsize=25, loc="upper left", framealpha=1, ncol=1)
    plt.show()
    
