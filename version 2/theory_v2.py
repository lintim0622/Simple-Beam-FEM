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

        self.zeta = 0.0  # 阻尼比

    def set_IC(self, uo, vo):
        self.uo = uo
        self.vo = vo

    def set_mode(self, n):
        self.n = int(n)

    def set_tau(self, tau):
        self.tau = tau

    def set_c(self):
        self.c = pow(self.E * self.I / (self.rho * self.A), 0.5)

    def set_beta(self, n):
        beta_square = pow(n * self.pi / self.L, 4)
        self.beta = pow(beta_square, 0.5)

    def set_F_ext(self, p):
        self.p = p

    def set_a(self, a):
        self.a = a

    def set_zeta(self, zeta):
        self.zeta = zeta

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

    def get_zeta(self):
        return self.zeta

    def H(self, t):
        return 1.0 if t >= self.tau else 0.0

    def omega_d(self, omega_n):
        zeta = self.get_zeta()
        return omega_n * np.sqrt(1 - zeta ** 2)

    def Tno(self, n):
        return (2.0 * self.get_uo() / (n * self.pi)) * (1.0 - pow(-1.0, n))

    def dTno(self, n):
        return (2.0 * self.get_vo() / (n * self.pi)) * (1.0 - pow(-1.0, n))

    def Tnf(self, n, t):
        p = self.get_pforce()
        a = self.get_a()
        c = self.get_c()
        beta = self.get_beta(n)
        zeta = self.get_zeta()
        omega_n = c * beta
        omega_d = self.omega_d(omega_n)

        T = 0.0

        if zeta == 0.0:
            # 無阻尼情形
            T = self.H(t) * (2.0 * p / (self.rho * self.A * self.L)) * np.sin(n * self.pi * a / self.L) / (c * beta)
            T *= np.sin(omega_n * (t - self.get_tau()))
            T += self.Tno(n) * np.cos(omega_n * t)
            T += (self.dTno(n) / omega_n) * np.sin(omega_n * t)
        else:
            # 有阻尼模態解
            A = self.Tno(n)
            B = (self.dTno(n) + zeta * omega_n * A) / omega_d
            exp_term = np.exp(-zeta * omega_n * t)
            T = exp_term * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

            # 外力項近似（僅模擬單模態 delta 載入的衰減反應）
            if t >= self.tau:
                F_amp = (2.0 * p / (self.rho * self.A * self.L)) * np.sin(n * self.pi * a / self.L)
                T += self.H(t) * F_amp / (omega_d) * np.exp(-zeta * omega_n * (t - self.tau)) * np.sin(omega_d * (t - self.tau))

        return T

    def mode_shape(self, n, x):
        return np.sin(n * self.pi * x / self.L)

    def u(self, x, t):
        deflection = 0
        for n in range(1, self.get_mode() + 1):
            Tn = self.Tnf(n, t)
            deflection += Tn * self.mode_shape(n, x)
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
    
    # 定義物理與幾何參數
    L = 10.0
    h = 2.0
    w = 0.5
    rho = 2300.0
    E = 432e6
    v0 = 0.01
    dt = 1e-4
    t_end = 2.0
    Nt = int(t_end / dt) + 1

    A = h * w
    I = w * h**3 / 12.0

    # 讀取 FEM 中點位移歷史
    fem_data_path = "FEM_DYNAMICS_CENTER.txt"
    fem_disp = rede_data(fem_data_path, Nt)
    tns = np.linspace(0, t_end, Nt)

    # 建立理論解
    theory = Theory(rho, A, E, I, L)
    theory.set_zeta(0.03)   # 模態阻尼比 ζ = 1%
    theory.set_c()
    theory.set_IC(uo=0.0, vo=-0.01)     # 無初始速度與位移
    theory.set_tau(0.0)              # 集中力作用時間
    theory.set_F_ext(10.0)           # 外力大小（單位 N）
    theory.set_a(L / 2)              # 施力位置（梁中心）
    theory.set_mode(3)

    theory_disp = np.array([theory.u(L / 2, t) for t in tns[:-1]])

    # 繪圖比較
    from matplotlib import pyplot as plt
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(10,6))
    plt.plot(tns[:-1], theory_disp, '-', lw=3, label="Theory", color="tab:blue")
    plt.plot(tns[:-1], fem_disp, '--', lw=3, label="FEM", color="tab:orange")
    plt.xlabel("Time (s)", fontsize=18)
    plt.ylabel("Displacement $u_y$ (m)", fontsize=18)
    plt.title("Comparison of Dynamic Response at Beam Center", fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()