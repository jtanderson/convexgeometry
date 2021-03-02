import numpy as np
import numpy.linalg as la

def angle2vec(theta):
    return np.array([np.cos(theta), np.sin(theta)])

# 2d for now
# Assumption: start is in the body
# memberfunc needs to be a membership oracle that gets
# called like memberfunc(query_point, *args, **kwargs)

# TODO: this could and should probably be abstracted to a master base class
# to be used for all random walks

class HitAndRunWalk:
    """ Walk through a shape, guided by a membership oracle """

    def __init__(self, memberfunc, start, space=1, *args, **kwargs):
        """
        Create the walk object, storing a reference to the membership oracle.

        Need to provide a starting point within the body; defaults to origin.

        The membership oracle should be callable as: memberfunc(x, *args, **kwargs)
        where x is the query point, and any additional arguments are passed through.
        """
        self.oracle = memberfunc
        self.oracle_args = args
        self.oracle_kwargs = kwargs
        self.dim = len(start)
        self.loc = start
        self.steps = 0
        self.space = space               # steps between each kept point
        assert(self.space > 0)
        self.record_steps = True
        self.history = np.empty((0,self.dim)) # will be N-by-dim after N steps

    def query(self, x):
        """Call the given oracle for a single point, using the stored extras."""
        # print(f"Calling oracle with x: {x}, args={self.oracle_args}, kwargs={self.oracle_kwargs}")
        return self.oracle(x, *self.oracle_args, **self.oracle_kwargs)

    def set_loc(self, loc):
        """Specify current walk location"""
        self.loc = loc

    def step(self):
        """Performs one step from the current location"""
        rtn = self.loc
        for i in range(0,self.space):
            rtn, _, _ = self._step(rtn)

        self.steps += 1
        self.loc = np.copy(rtn) # necessary?

        if self.record_steps:
            self.history = np.concatenate((self.history, [rtn]), axis=0)

        assert(self.history.shape == (self.steps, self.dim))

        return rtn

    def _step(self, start):
        """
        Internal. Do one step of hit-and-run.

        Returns:
            - Result point
            - Boundary points along the chosen segment
        """
        #angle = np.random.uniform(0,2*np.pi) # only 2-dim
        #direction = angle2vec(angle)

        angle = np.random.randn(self.dim)
        direction = angle / la.norm(angle)
        
        if not self.query(start):
            print(f"Given an invalid point! {start}")
        
        testCounter = 0
        max_iter = 1000
        
        ## Case for adding to direction ##
        high = 1
        testCounter = 0
        while(self.query(start + high*direction)):
            high = high*2
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_plus high loop with: \n\
                    high = {high}\n")
        
        low = high/2
        testCounter = 0
        while(not self.query(start + low*direction)):
            low = low/2
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_plus low loop with: \n\
                    low = {low}\n")
        
        # now we know that (start +  low * direction)  is inside
        #assert(zonoid_membership_def(A, start+low*direction))
        #         and that (start + high * direction) is outside
        #assert(not zonoid_membership_def(A, start+high*direction))
        
        tol = 1e-5
        t_plus = (high-low)/2
        old_t = 1
        current = start
        testCounter = 0
        while(abs(t_plus-old_t) > tol):
            old_t = t_plus
            t_plus = (high+low)/2
            testpoint = current + t_plus*direction
            if( self.query(testpoint) ):
                low = t_plus
            else:
                high = t_plus
            
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_plus loop with: \n\
                    t_plus = {t_plus}\n\
                    t_old = {t_old}\n\
                    high = {high}\n\
                    low = {low}\n")
        t_plus = old_t
        
        ## Case for subtracting from direction
        high = -1
        testCounter = 0
        while(self.query(start + high*direction)):
            high = high*2
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_minus high loop with: \n\
                    high = {high}\n")
        
        low = high/2
        testCounter = 0
        while(not self.query(start + low*direction)):
            low = low/2
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_minus low loop with: \n\
                    low = {low}\n")
        
        # now we know that (start +  low * direction)  is inside
        #assert(zonoid_membership_def(A, start+low*direction))
        #         and that (start + high * direction) is outside
        #assert(not zonoid_membership_def(A, start+high*direction))
        
        tol = 1e-10
        t_minus = (high-low)/2
        old_t = 1
        current = start
        testCounter = 0
        while(abs(t_minus-old_t) > tol):
            old_t = t_minus
            t_minus = (high+low)/2
            testpoint = current + t_minus*direction
            if( self.query(testpoint) ):
                low = t_minus
            else:
                high = t_minus
            
            testCounter += 1
            if testCounter > max_iter:
                print(f"Warning: Stuck in t_minus loop with: \n\
                    t_minus = {t_minus}\n\
                    t_old = {t_old}\n\
                    high = {high}\n\
                    low = {low}\n")
        t_minus = old_t
        
        # Make the step
        final_t = np.random.uniform(t_minus, t_plus)
        #print(f"Final t = {final_t}")
        
        return start + final_t*direction, start+t_plus*direction, start+t_minus*direction

