import numpy as np
import numpy.linalg as la

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
def WSep2Viol(c, gamma, epsilon, n, R, sepK):
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

# TODO -- Theorem 4.3.2
def WMem2Viol(c, gamma, eps, memK, n, R, r, a0):
    """
    Given a body (K; n, R, r, a0), where
    memK: weak-membership oracle
    n: dimensions
    R: circumradius so S(0,r) contains K
    r: radius so that of S(a0,r) in K
    Either
       i) asserts that c.T @ x <= gamma + eps for all x in in S(K,-eps)
      ii) find a vector y in S(K, eps) with c.T @ x >= gamma - eps
    """
    pass

# Lemma 4.3.3
def StrongerMem(y, delta, memK, n, R, r, a):
    """
    Given vector y, error delta > 0,
    and weak-membership oracle
    for a body (K; n, R, r, a0), where
    n: dimensions
    R: circumradius so S(0,r) contains K
    r: radius so that of S(a0,r) in K
    Either
       i) asserts that y in S(K,delta)
      ii) asserts that y not in K
    """
    assert(delta>0)
    if la.norm(y - a, ord=2) < 2*R:
        return False

    yprime = (1 - delta/(4*R))*y + (delta*a)/(4*R)
    deltaprime = (r*delta)/(4*R)

    test = memK(yprime, deltaprime, n, R)
    assert(isinstance(test,bool))
    return test

# Lemma 4.3.4
def SeparationNonEllipsoid(y, delta, beta, memK, n, R, r, a):
    """
    Either
       i) asserts that y in S(K,delta),
      ii) finds vector c such that c != 0 and for every x in K,
          c.T @ x <= c.T @ y + (delta + beta |x-y|) |c|
    """
    if la.norm(y - a) >= 2*R:
        return (y-a) / la.norm(y - a)

    if StrongerMem(y, delta, memK, n, R, r, a):
        return True

    # StrongerMem says y not in K
    alpha = np.arctan(beta/(4*n*n))
    delta1 = (r*delta) / (R + r)
    r1 = (r*delta1) / (4*n*R)
    eps1 = (beta*beta*r1) / (16*n*n*n*n)

    in_v = np.copy(a)
    out_v = np.copy(y)

    # binary search to straddle the boundary close enough
    while la.norm(in_v - out_v) > delta1/(2*n):
        new_v = (in_v + out_v) / 2
        if memK(new_v, eps1, n, R):
            in_v = np.copy(new_v)
        else:
            out_v = np.copy(new_v)

    vpp = (1/(r + eps1)) * ((r-r1)*in_v + (r1 + eps1)*a)

    # we know S(vpp, r1) subseteq K
    # now need to re-center the algorithm at vpp so that vpp == 0...
    # TODO finish, page 109


# TODO
def WVal():
    pass

# TODO
def WSep():
    pass

# TODO
def Wmem():
    pass
