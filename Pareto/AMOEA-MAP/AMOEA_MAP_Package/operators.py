
""" crossover and mutation operators """

import math, random, copy
from generic_calcultors import *
from SearchSpace_Partitioning import *

def crossover_operator(parents_, arg):
    method = arg["Crossover"][0]
    cross_rate = float(arg["Crossover"][1])
    Ub = arg["Ubounds"]
    Lb = arg["Lbounds"]
    cand = list(parents_)
    if len(cand) % 2 == 1:
        cand = cand[:-1]
    moms = cand[::2]
    dads = cand[1::2]
    children = []
    for i, (mom, dad) in enumerate(zip(moms, dads)):
        if method == 'SBX':
            offspring = SBX_crossover(mom, dad, cross_rate, Lb, Ub)

        # integer projection of variable space
        if arg["MAP"][0]:
            for o in offspring:
                o = Solutions_Restriction(o,arg)

        for o in offspring:
            children.append(o)
    return children

def mutation_operator(parents_, arg):
    method = arg["Mutation"][0]
    mute_rate = float(arg["Mutation"][1])
    Ub = arg["Ubounds"]
    Lb = arg["Lbounds"]
    mutants = list(parents_)
    Importance = arg['Importance']
    no_sensitive_var = []
    for i_ in range(arg['Number of variables']):
        if Importance[i_]==0:
            no_sensitive_var.append(i_)
    if no_sensitive_var:
        index_ = random.choice(range(len(parents_)))
    else:
        index_ = None
    for i, cs in enumerate(mutants):
        if method == 'Polynomial':
            mutants[i] = Polynomial_mutation(cs, mute_rate, Lb, Ub)
        if method == 'IAMO':
            mutants[i] = IAMO_mutation(cs, mute_rate, arg, Lb, Ub)

        # integer projection of variable space
        if arg["MAP"][0]:
            mutants[i] = Solutions_Restriction(mutants[i],arg)
    return mutants

## Crossover operators ---------------------------------------------------------
def SBX_crossover(mom, dad, cross_rate, Lb, Ub):
    """ This function performs simulated binary crosssover, following the
    implementation in NSGA-II
    (Deb et al., ICANNGA 1999) <http://vision.ucsd.edu/~sagarwal/icannga.pdf>
    - eta_c: the non-negative distribution index (default 10)
    A small value allows solutions far away from parents to be created as
    children solutions, while a large value restricts only near-parent solutions
    to be created as children solutions. """

    eta_c = 15
    bro = copy.copy(dad)
    sis = copy.copy(mom)
    for i, (m, d, lb, ub) in enumerate(zip(mom, dad, Lb, Ub)):
        try:
            if m > d:
                m, d = d, m
            #if abs(float(d-m))>1e-8:
            beta = 1.0 + 2 * min(m - lb, ub - d) / float(d - m)
            #else:
            #    beta = 1.0 + 2 * min(m - lb, ub - d) / 1e-8

            alpha = 2.0 - 1.0 / beta**(eta_c + 1.0)
            u = random.random()
            if u <= (1.0 / alpha):
                beta_q = (u * alpha)**(1.0 / float(eta_c + 1.0))
            else:
                beta_q = (1.0 / (2.0 - u * alpha))**(1.0 / float(eta_c + 1.0))
            bro_val = 0.5 * ((m + d) - beta_q * (d - m))
            bro_val = max(min(bro_val, ub), lb)
            sis_val = 0.5 * ((m + d) + beta_q * (d - m))
            sis_val = max(min(sis_val, ub), lb)
            if random.random() > 0.5:
                bro_val, sis_val = sis_val, bro_val
            bro[i] = bro_val
            sis[i] = sis_val
        except ZeroDivisionError:
            # The offspring already have legitimate values for every element,
            # so no need to take any special action here.
            pass
    return [bro, sis]

## Mutation operators ---------------------------------------------------------
def Polynomial_mutation(candidate, mute_rate, Lb, Ub):
    m_eta_m = 20.0
    for i, c in enumerate(candidate):
        if random.random() < mute_rate:
            y = candidate[i]
            yl = Lb[i]
            yu = Ub[i]
            delta1 = (y-yl)/(yu-yl)
            delta2 = (yu-y)/(yu-yl)
            mut_pow = 1.0/(m_eta_m+1.0)
            rnd = random.random()
            if rnd < 0.5:
                xy = 1.0-delta1
                val = 2.0*rnd+(1.0-2.0*rnd)*(xy**(m_eta_m+1.0))
                deltaq =  val**mut_pow - 1.0
            else:
                xy = 1.0-delta2
                val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(xy**(m_eta_m+1.0))
                deltaq = 1.0 - (val**mut_pow)
            y = y + deltaq*(yu-yl)
            if (y<yl):
                y = yl
            if (y>yu):
                y = yu
            new_ = y
            candidate[i] = bounded(new_, Ub[i], Lb[i])
    return candidate

def IAMO_mutation(candidate, mute_rate, arg, Lb, Ub):
    """ IAMO Mutation operator:
    Importance-based Adaptive Mutation Operator
    <Ahmadi et al., 2016. Computers and Chemical Engineering 87 (2016) 95-110."""

    Importance = arg['Importance']
    no_sensitive_var = []
    for i_ in range(arg['Number of variables']):
        if Importance[i_]<1.0:
            no_sensitive_var.append(i_)
    if no_sensitive_var:
        index_ = random.choice(no_sensitive_var)
    else:
        index_ = None

    t = float(arg["Current function calls"])
    t_max = float(arg['Max function calls'])

    m_eta_m = 20.0
    for i, c in enumerate(candidate):
        if index_ and i==index_:
            delta = 0.2*((t_max-t)/(t_max))**5.0
            u_ = float(random.choice([-1,1]))
            y = candidate[i]
            y_new = y + delta*u_*(Ub[i]-Lb[i])
            candidate[i] = bounded(y_new, Ub[i], Lb[i])
        else:
            mutation_rate = mute_rate
            if random.random() < mutation_rate:
                y = candidate[i]
                yl = Lb[i]
                yu = Ub[i]
                delta1 = (y-yl)/(yu-yl)
                delta2 = (yu-y)/(yu-yl)
                mut_pow = 1.0/(m_eta_m+1.0)
                rnd = random.random()
                if rnd < 0.5:
                    xy = 1.0-delta1
                    val = 2.0*rnd+(1.0-2.0*rnd)*(xy**(m_eta_m+1.0))
                    deltaq =  val**mut_pow - 1.0
                else:
                    xy = 1.0-delta2
                    val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(xy**(m_eta_m+1.0))
                    deltaq = 1.0 - (val**mut_pow)
                y = y + deltaq*(yu-yl)
                if (y<yl):
                    y = yl
                if (y>yu):
                    y = yu
                new_ = y
                candidate[i] = bounded(new_, Ub[i], Lb[i])
    return candidate



