import constants
from inspect import signature

import numpy as np
import numpy.linalg as linalg
import scipy.integrate as integrate


class Config():
    pass

class Particle():
    def __init__(self, pos, q=0, type=None, name=None):
        if type=='electron':
            self.q = -constants.q_e
        elif type=='proton':
            self.q = constants.q_e
        else:
            self.q = q
        self.dimensions = len(pos)
        self.pos = np.array(pos, dtype=np.float64).reshape(1, -1)
        self.name = name
    
    def __repr__(self):
        name = self.name if self.name != None else "Particle"
        return f"{name} of charge {self.q} at position {tuple(self.pos)}"
    
    def potential(self):
        return lambda pos: 1/(4*constants.pi*constants.e0)*self.q/(np.sqrt(np.sum((pos-self.pos)**2, axis=-1)))


class Line():
    def __init__(self, start, end, q_func, name=None):
        start, end = np.array(start, dtype=np.float64), np.array(end, dtype=np.float64)
        
        if len(start) != len(end):
            raise ValueError("Start and Endpoints must be consistent!")
        
        
        # ensure q_func is a function
        f = q_func
        if not callable(q_func): f = lambda x, y: q_func
        elif len(signature(q_func).parameters) == 1: 
            f = lambda x, y: q_func(x)
        
        self.dimensions = len(start)
        self.start = start
        self.end = end
        self.length = np.sqrt(np.sum((self.start-self.end)**2))
        self.q = f
        self.name = name

    def rotation_angles(self): 
        angles = []
        phi = np.arctan((self.end[1] - self.start[1])/(self.end[0]-self.start[0]))
        angles.append(phi)

        if self.dimensions > 2:
            theta = np.arccos((self.end[2] - self.start[2])/self.length)
            angles.append(theta)
        return np.array(angles)

    def __repr__(self):
        name = self.name if self.name != None else "Line"
        return f"{name} line charge at position {tuple(self.start)} to {tuple(self.end)} with charge density {self.q}"
    
    def potential(self):
        phi, *rest = self.rotation_angles()
        # if self.dimensions == 2:
        rotation_matrix = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
        
        # rotation
        pivot = 0.5*(self.end+self.start)
        start, end = np.linalg.matmul(rotation_matrix, self.start-pivot)[0], np.linalg.matmul(rotation_matrix, self.end-pivot)[0]

        def potential_helper(pos0):
           pos0 = np.array(pos0, dtype=np.float64)
           pos0 = pos0-pivot
    
           if len(pos0.shape) == 1:
                pos0 = pos0[None]
           pos0_p = np.array([np.linalg.matmul(rotation_matrix, p) for p in pos0])

           def f(x):
               dq = self.q((x), (np.tan(phi)*x))/np.sqrt(1+np.tan(phi)**2)
               d = np.sqrt((x-pos0_p[:,0])**2 + (pos0_p[:,1])**2 + 1e-10)
               differential = 1/(4*constants.pi*constants.e0)*dq/d # pyright: ignore[reportOperatorIssue]
               return differential
           return f
        
        return lambda pos0: integrate.quad_vec(potential_helper(pos0), start, end,epsabs=1e-5, epsrel=1e-5)[0]