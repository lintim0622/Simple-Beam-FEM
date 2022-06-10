# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:22:06 2022

@author: user
"""

class Force(object):
    def __init__(self, msh):
        self.F = np.zeros(int(msh.tot_node_num*2))
            
    def apply_info(self, pid, P):
        self.pid = pid
        self.P = P
            
    def vector(self, msh):
            
        if (self.pid > msh.tot_node_num):
            from sys import exit
            print("reset external force id")
            exit()
        
        for node in msh.nodes:
            node.f_ext = np.zeros(2)
            if (node.nid == self.pid):
                node.f_ext[0] = self.P[0]
                node.f_ext[1] = self.P[1]   
            
        for ie in msh.elements:
            local_f = [ie.n1.f_ext[0], ie.n1.f_ext[1], ie.n2.f_ext[0], ie.n2.f_ext[1], ie.n3.f_ext[0], ie.n3.f_ext[1], ie.n4.f_ext[0], ie.n4.f_ext[1]]
            gid = [ie.n1.gid[0], ie.n1.gid[1], ie.n2.gid[0], ie.n2.gid[1], ie.n3.gid[0], ie.n3.gid[1], ie.n4.gid[0], ie.n4.gid[1]]
            for i in range(8):
                self.F[gid[i]] = local_f[i]
                
        return self.F