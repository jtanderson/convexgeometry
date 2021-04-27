import numpy as np
import numpy.linalg as l

def CentralCutEllipsoid(eps, sepK, n, R):
    """
    Solves the following problem:
    Given a rational number eps > 0 and circumscriebd convex set (K; n, R)
    given by an oracle sepK that, for any y and any rational delta > 0,
    either asserts that y in S(K, delta) or finds a vector c with norm(c, inf) = 1
    such that c.x <= c.y + delta for every x in K.

    Outputs of the following:
    a) a vector a in S(K,eps)
    b) a pd n-by-n matrix A and a n-dim point a such that K is a subset of E(A,a)
       and vol(E(A,a)) < eps

    Note: in the case of a) will also output None as the second parameter
    """
    N = int(np.ceil(5*n*np.abs(np.log2(2*R)) + 5*n*n*np.abs(np.log2(2*R))))
    p = 8*N
    delta = 2**(-p)

    a = 0
    A = R**2 @ np.eye(n,n)

    for k in range(0,N):
        c = sepK(y=a, error=delta)
        if( isinstance(c,bool) ):
            return a, None
        else:
            # c is a vector with inf norm = 1 and c.x <= c.y + delta for all x in K
            #c /= np.max(c)
            # TODO: officially supposed to round to only p digits beyond the decimal
            a = a - (1/(n+1)) * ((A@c) / np.sqrt(c.T @ A @ c))
            A = ((2*n*n + 3)/(2n*n)) * (A - (2/(n+1)) * ((A @ c @ c.T @ A)/(c.T @ A @ c)))
    return A, a


"""
Poly-time algorithm that solves weak violation problem for every circumscribed
convex body (K; n, R) given by a weak separation oracle.

Comes from proof of 4.2.2 in GLS book
"""
def WViol(c, gamma, epsilon, n, R, sepK):
    """
    Solves the following problem: given a body (K; n, R), and a 
    separation oracle, a vector c, and gamma, epsilon > 0;
    either
      i) assert that <c,x> <= gamma + epsilon for all x in S(K,-eps), or
        - <c,x> <= gamma is almost valid
      ii) find a vector y in S(K,eps) with <c,y> >= gamma - eps
        - y almost violates <c,x> <= gamma
    """

    #assume that max(c) = 1
    cscale = np.max(c)
    c /= cscale

    eps_p = epsilon / (2*n)

    def sep_kprime(y, delta, n, R):
        if not (np.dot(c, y) >= gamma + delta):
            return -c
        else:
            # second parameter is delta1
            d = sepK(y, np.min((eps_p, delta/n)))
            if isinstance(d, bool):
                # asserted that y is in S(K, delta1)
                return True
            else:
                # got a vector d so that <x,d> <= <y,d> + delta for all x in S(K, -delta1)
                return d / np.max(d) # just to make sure?

    # now, run ellipsoid algorithm with sep_kprime above and eps1 = np.min(eps_p, (eps_p/n)**n)
    eps1 = np.min(eps_p, (eps_p/n)**n)
    a, A = CentralCutEllipsoid(eps1, sepK, n, R)
    if A is None:
        # gave a point a in S(Kprime, eps1)
        return a
    else:
        # gave an ellipsoid E of volume at most eps1 containing K
        # so we assert that c.x < gamma + eps
        return True

def WVal():
    pass

def WSep():
    pass

def Wmem():
    pass
