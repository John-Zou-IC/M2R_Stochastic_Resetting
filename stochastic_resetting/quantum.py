from numpy.polynomial.hermite import Hermite


class SimpleHarmonicOscillator:

    def __init__(self, alpha, r, omega_0, M):
        self.alpha = alpha
        self.r = r
        self.omega_0 = omega_0
        self.M = M

    def Hermite_n(self, n):
        coeffs = (0 for i in range(n))
        coeffs[-1] = 1
        poly = Hermite(coeffs)
        return poly
