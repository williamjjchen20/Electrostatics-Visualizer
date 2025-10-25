import constants
import numpy as np
    
def point_potential(pos, charge):
    pos = np.array(pos, dtype=np.float64)
    potential = charge.potential()
    return potential(pos)

def electric_field(pos, particle):
    def partial(f, xs, i, h=0.001):
        xs1, xs2 = np.copy(xs), np.copy(xs)
        xs1[:,i] -= h
        xs2[:,i] += h
        return (f(xs2) - f(xs1))/(2*h)
    
    pos = np.array(pos, dtype=np.float64)
    if len(pos.shape) == 1:
        pos = pos[None]
    n = pos.shape[-1]
    #print(pos)
    potential = particle.potential()
    E = -np.array([partial(potential, pos, i) for i in range(n)])#.reshape(-1,n)
    return E 
