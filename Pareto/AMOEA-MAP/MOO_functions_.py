
""" Some test functions """

import math

# ZDT1
class ZDT1_2D:
    def __init__(self, arg):
        Nvar = arg['Number of variables']
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(0.0))
            Ubounds.append(float(1.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "ZDT1"
        self.Bench_descret_matrix = {}

    def evaluate(self, chromosome):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        f1 = X_[0]
        sum_ = 0.0
        for i in range(n_var-1):
            sum_ += X_[i+1]
        g = 1.0 + (9.0 * (sum_ / (n_var-1)))
        f2 = g * (1.0 - math.sqrt(X_[0] / g))

        objective_vector.append(f1)
        objective_vector.append(f2)

        return objective_vector


# DTLZ1_3D
class DTLZ1_3D:
    # DTLZ1 mutliobjective function. It returns a tuple of obj values.
    # The individual must have at least obj elements. From: K. Deb, L. Thiele,
    # M. Laumanns and E. Zitzler. Scalable Multi-Objective Optimization Test
    # Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.
    def __init__(self, arg):
        Nvar = arg['Number of variables']
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(0.0))
            Ubounds.append(float(1.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "DTLZ1"
        self.Bench_descret_matrix = {}

    def evaluate(self, chromosome):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        k = n_var-3+1 # n-m+1
        g_sum = 0.0
        for i in range(n_var-2):
            g_sum += (X_[i+2]-0.5)**2.0-math.cos(20.0*math.pi*(X_[i+2]-0.5))

        g_sum = 100*(k+g_sum)

        objective_vector.append(0.5*(1+g_sum)*X_[0]*X_[1])
        objective_vector.append(0.5*(1+g_sum)*(1-X_[1])*X_[0])
        objective_vector.append(0.5*(1+g_sum)*(1-X_[0]))

        return objective_vector


class beam_design_2D:
    # The welded beam design problem 2D
    # Deb, Sundar, Rao KanGAL Report Number 2005012
    def __init__(self, arg):
        Nvar = arg['Number of variables']
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        Lbounds.append(float(0.125))
        Lbounds.append(float(0.1))
        Lbounds.append(float(0.1))
        Lbounds.append(float(0.125))
        Ubounds.append(float(5.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(10.0))
        Ubounds.append(float(5.0))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "Beam design"
        self.Bench_descret_matrix = {}

    def evaluate(self, chromosome):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        h = X_[0]
        l = X_[1]
        t = X_[2]
        b = X_[3]

        f1 = 1.10471*(h**2.0)*l+0.04811*t*b*(14.0+l)

        Tau1 = 6000.0/(math.sqrt(2.0)*h*l)
        Tau2 = 6000.0*(14.0+0.5*l)*math.sqrt(0.25*(l**2.0+(h+t)**2.0)) / (2.0*(0.707*h*l)*((l**2.0)/12.0+0.25*(h+t)**2.0))
        sigma = 504000.0/((t**2.0)*b)
        Pc = 64746.022*(1.0-0.0282346*t)*t*(b**3.0)

        f2 = 2.1952/((t**3.0)*b)

        Tau = math.sqrt(Tau1**2.0+Tau2**2.0+l*Tau1*Tau2/math.sqrt(0.25*(l**2.0+(h+t)**2.0)))

        # constraints
        g = []
        p1 = 100
        p2 = 0.1
        g.append((-Tau+13600.0)*p1)
        g.append((-sigma+30000.0)*p2)
        g.append(-h+b)
        g.append(Pc-6000.0)
        #g.append(-delta+0.25)

        w = []
        for i in range(4):
            if g[i]<0.0:
                w.append(abs(g[i]))
            else:
                w.append(0.0)

        constraint = sum(w)
        objective_vector.append(f1)
        objective_vector.append(f2)
        chromosome.constraint = constraint

        return objective_vector

from XFoilGame import *

# DTLZ1_3D
class XFoilPareto_3D:
    # DTLZ1 mutliobjective function. It returns a tuple of obj values.
    # The individual must have at least obj elements. From: K. Deb, L. Thiele,
    # M. Laumanns and E. Zitzler. Scalable Multi-Objective Optimization Test
    # Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.
    def __init__(self, arg):
        Nvar = arg['Number of variables']
        Lbounds = []
        Ubounds = []
        del Lbounds[:]
        del Ubounds[:]
        for i in range(Nvar):
            Lbounds.append(float(0.01))
            Ubounds.append(float(0.1))
        self.Lbounds = Lbounds
        self.Ubounds = Ubounds
        self.Name = "DTLZ1"
        self.Bench_descret_matrix = {}

    def evaluate(self, chromosome):
        objective_vector = []
        n_var = chromosome.Num_Var
        del objective_vector[:]

        X_ = []
        del X_[:]
        for i in range(n_var):
            X_.append(chromosome.variables[i])

        k = n_var-3+1 # n-m+1
        g_sum = 0.0
        for i in range(n_var-2):
            g_sum += (X_[i+2]-0.5)**2.0-math.cos(20.0*math.pi*(X_[i+2]-0.5))

        g_sum = 100*(k+g_sum)

        objective_vector.append(0.5*(1+g_sum)*X_[0]*X_[1])
        objective_vector.append(0.5*(1+g_sum)*(1-X_[1])*X_[0])
        objective_vector.append(0.5*(1+g_sum)*(1-X_[0]))
        print X_
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/Pareto/Archive")
        XFG = XFoilGame()
        objective_vector=XFG.newGame(X_)

        return objective_vector