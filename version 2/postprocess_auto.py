# -*- coding: utf-8 -*-
"""
Created on Sat May 24 15:49:21 2025

@author: lintim0622
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from natsort import natsorted

def save_center_displacement(U_history, pid, tns, filename="FEM_DYNAMICS_CENTER.txt"):
    with open(filename, 'w') as f:
        for i, disp in enumerate(U_history):
            u = disp[pid, 0]
            v = disp[pid, 1]
            f.write(f"{i:7d} {pid:5d} {u:14.6e} {v:14.6e}\n")
    print(f"[✓] 中心節點位移儲存至: {filename}")

def save_all_figures(msh, U_history, tns, folder="fig_store", skip=200, scale=50.0):
    os.makedirs(folder, exist_ok=True)
    print(f"[*] 匯出圖片到 {folder}...")

    for i in range(0, len(U_history), skip):
        for node in msh.nodes:
            node.displacement = U_history[i][node.nid, :]
        save_single_fig(msh, tns, i, folder, scale)

    print(f"[✓] 圖片全部匯出完成")

def save_single_fig(msh, tns, i, folder, scale=50.0):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    for ele in msh.elements:
        nodes = ele.nodes
        p = [n.position for n in nodes]
        d = [n.displacement * scale for n in nodes]

        # 原始
        ax.plot([p[0][0], p[1][0]], [p[0][1], p[1][1]], '-', lw=1, color='gray')
        ax.plot([p[1][0], p[2][0]], [p[1][1], p[2][1]], '-', lw=1, color='gray')
        ax.plot([p[2][0], p[3][0]], [p[2][1], p[3][1]], '-', lw=1, color='gray')
        ax.plot([p[3][0], p[0][0]], [p[3][1], p[0][1]], '-', lw=1, color='gray')

        # 變形
        ax.plot([p[0][0]+d[0][0], p[1][0]+d[1][0]], [p[0][1]+d[0][1], p[1][1]+d[1][1]], '--', lw=1, color='tab:red')
        ax.plot([p[1][0]+d[1][0], p[2][0]+d[2][0]], [p[1][1]+d[1][1], p[2][1]+d[2][1]], '--', lw=1, color='tab:red')
        ax.plot([p[2][0]+d[2][0], p[3][0]+d[3][0]], [p[2][1]+d[2][1], p[3][1]+d[3][1]], '--', lw=1, color='tab:red')
        ax.plot([p[3][0]+d[3][0], p[0][0]+d[0][0]], [p[3][1]+d[3][1], p[0][1]+d[0][1]], '--', lw=1, color='tab:red')

    ax.set_xlim(-1, msh.L + 1)
    ax.set_ylim(-1.5, msh.h + 1.5)
    ax.set_title(f"t = {tns[i]:.2f} s")
    ax.set_aspect('equal', 'box')
    ax.set_xlabel("Length (m)")
    ax.set_ylabel("Height (m)")
    ax.legend(["undeformed", "deformed"], loc="upper left")
    plt.savefig(os.path.join(folder, f"{i+1:07d}_deformed.png"))
    plt.ioff()                   # ✅ 關閉互動模式，防止圖窗跳出
    plt.close()                 # ✅ 關閉圖窗釋放記憶體


def make_video(folder="fig_store", output="beam_deformation.mp4", fps=10):
    files = natsorted([f for f in os.listdir(folder) if f.endswith(".png")])
    if not files:
        print("[X] 無圖片可合成影片")
        return
    images = [imageio.imread(os.path.join(folder, f)) for f in files]
    imageio.mimsave(output, images, fps=fps)
    print(f"[✓] MP4 影片輸出至: {output}")

def plot_center_disp_from_txt(path="FEM_DYNAMICS_CENTER.txt", dt=1e-4):
    data = np.loadtxt(path)
    t = data[:, 0] * dt
    uy = data[:, 3]
    plt.figure(figsize=(10,4))
    plt.plot(t, uy, label="Center node $u_y$")
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement (m)")
    plt.title("Center node displacement history")
    plt.legend()
    plt.tight_layout()
    plt.savefig("center_displacement_plot.png")
    plt.show()

# 🔧 主函式整合
def auto_postprocess(msh, results, tns, skip=200, scale=50.0, dt=1e-4):
    save_center_displacement(results.U_history, results.pid, tns)
    save_all_figures(msh, results.U_history, tns, skip=skip, scale=scale)
    make_video(fps=10)
    plot_center_disp_from_txt(dt=dt)
