import numpy as np

def angle2vec(theta):
    return np.array([np.cos(theta), np.sin(theta)])

# 2d for now
# Assumption: start is in the body
# memberfunc needs to be a membership oracle that gets
# called like memberfunc(query_point, *args, **kwargs)
def hitrun_step(start, memberfunc, *args, **kwargs):
    angle = np.random.uniform(0,2*np.pi)
    anglev = angle2vec(angle)
    
    if not memberfunc(start, *args, **kwargs):
        print(f"Given an invalid point! {start}")
    
    testCounter = 0
    max_iter = 1000
    
    ## Case for adding to anglev ##
    high = 1
    testCounter = 0
    while(memberfunc(start + high*anglev, *args, **kwargs)):
        high = high*2
        testCounter += 1
        if testCounter > max_iter:
            print(f"Stuck in t_plus high loop with: \n\
                high = {high}\n")
    
    low = high/2
    testCounter = 0
    while(not memberfunc(start + low*anglev, *args, **kwargs)):
        low = low/2
        testCounter += 1
        if testCounter > max_iter:
            print(f"Stuck in t_plus low loop with: \n\
                low = {low}\n")
    
    # now we know that (start +  low * anglev)  is inside
    #assert(zonoid_membership_def(A, start+low*anglev))
    #         and that (start + high * anglev) is outside
    #assert(not zonoid_membership_def(A, start+high*anglev))
    
    tol = 1e-5
    t_plus = (high-low)/2
    old_t = 1
    current = start
    testCounter = 0
    while(abs(t_plus-old_t) > tol):
        old_t = t_plus
        t_plus = (high+low)/2
        testpoint = current + t_plus*anglev
        if( memberfunc(testpoint, *args, **kwargs) ):
            low = t_plus
        else:
            high = t_plus
        
        testCounter += 1
        if testCounter > max_iter:
            print(f"Stuck in t_plus loop with: \n\
                t_plus = {t_plus}\n\
                t_old = {t_old}\n\
                high = {high}\n\
                low = {low}\n")
    t_plus = old_t
    
    ## Case for subtracting from anglev
    high = -1
    testCounter = 0
    while(memberfunc(start + high*anglev, *args, **kwargs)):
        high = high*2
        testCounter += 1
        if testCounter > max_iter:
            print(f"Stuck in t_minus high loop with: \n\
                high = {high}\n")
    
    low = high/2
    testCounter = 0
    while(not memberfunc(start + low*anglev, *args, **kwargs)):
        low = low/2
        testCounter += 1
        if testCounter > max_iter:
            print(f"Stuck in t_minus low loop with: \n\
                low = {low}\n")
    
    # now we know that (start +  low * anglev)  is inside
    #assert(zonoid_membership_def(A, start+low*anglev))
    #         and that (start + high * anglev) is outside
    #assert(not zonoid_membership_def(A, start+high*anglev))
    
    tol = 1e-10
    t_minus = (high-low)/2
    old_t = 1
    current = start
    testCounter = 0
    while(abs(t_minus-old_t) > tol):
        old_t = t_minus
        t_minus = (high+low)/2
        testpoint = current + t_minus*anglev
        if( memberfunc(testpoint, *args, **kwargs) ):
            low = t_minus
        else:
            high = t_minus
        
        testCounter += 1
        if testCounter > max_iter:
            print(f"Killing t_minus loop with: \n\
                t_minus = {t_minus}\n\
                t_old = {t_old}\n\
                high = {high}\n\
                low = {low}\n")
    t_minus = old_t
    
    # Make the step
    final_t = np.random.uniform(t_minus, t_plus)
    #print(f"Final t = {final_t}")
    
    return start + final_t*anglev, start+t_plus*anglev, start+t_minus*anglev

