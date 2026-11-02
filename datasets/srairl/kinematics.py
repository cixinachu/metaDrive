"""SR-AIRL's 10 Hz backward-difference trajectory preprocessing."""
import numpy as np


def position(track,t):
    states=np.asarray(track["states"],dtype=np.float64)
    if t<states[0,0]-1e-6 or t>states[-1,0]+1e-6:return None
    return np.asarray([np.interp(t,states[:,0],states[:,i]) for i in (1,2)],dtype=np.float64)


def state(track,t,dt=.1,require_preroll=True):
    """Return [x,y,vx,vy], deriving velocity exactly as upstream diff()/0.1."""
    p=position(track,t)
    if p is None:return None
    previous=position(track,t-dt)
    if previous is None:
        if require_preroll:raise ValueError(f"track {track.get('id')} needs {dt:.1f}s pre-roll for upstream velocity differences")
        values=np.asarray(track["states"],dtype=np.float64)
        return np.asarray([*p,*[np.interp(t,values[:,0],values[:,i]) for i in (3,4)]])
    return np.asarray([*p,*((p-previous)/dt)],dtype=np.float64)


def acceleration(track,t,dt=.1):
    """Return the second backward position difference used by upstream preprocessing."""
    p=position(track,t);p1=position(track,t-dt);p2=position(track,t-2*dt)
    if p is None or p1 is None or p2 is None:
        raise ValueError(f"track {track.get('id')} needs two 0.1s pre-roll frames for upstream acceleration differences")
    return (p-2*p1+p2)/(dt*dt)
