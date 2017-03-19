# ------------------------------------------------------------------------------
# AMOEA-MAP - A framework for expensive multi-objective optimization
# (MAP: Memory-based Adaptive Partitioning of the search space in Pareto-based
# multi-objective optimization)
#
# Copyright (C) 2016 Aras Ahmadi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Citations:
# [1] Ahmadi A., 2016. Memory-based Adaptive Partitioning (MAP) of search space
# for the enhancement of convergence in Pareto-based multi-objective
# evolutionary algorithms, Journal of Applied Soft Computing 41 (2016) 400-417.
# ------------------------------------------------------------------------------

""" Adaptive Partitioning of Search Space """

from math import log
from generic_calcultors import *

# Restriction of solutions
def Solutions_Restriction(var,arg):
    for ind in range(arg["Number of variables"]):
        if arg['Ubounds'][ind]==arg['Lbounds'][ind]:
            var[ind] = arg['Lbounds'][ind]
        else:
            dx = (arg['Ubounds'][ind]-arg['Lbounds'][ind])/arg["MAP"][1][ind]
            temp1 = integer_adjustment((var[ind]-arg['Lbounds'][ind])/dx)
            temp2 = temp1/arg["MAP"][1][ind]
            temp2 = temp2*(arg['Ubounds'][ind]-arg['Lbounds'][ind])+arg['Lbounds'][ind]
            var[ind] = round(temp2,12)
    return var

# variable refining
def Dynamic_Parameter_Refining(R_,up_,arg):
    Lower_hist = []
    Higher_hist = []
    sensitivity_hist = []
    for v_ in range(arg["Number of variables"]):
        # variable normalizing
        VAR = []
        del VAR[:]
        for c_ in range(len(R_)):
            VAR.append((R_[c_].variables[v_]-R_[c_].Lsup[v_])/(R_[c_].Usup[v_]-R_[c_].Lsup[v_]))
        Lower_hist.append(min(VAR))
        Higher_hist.append(max(VAR))
        sensitivity_hist.append(max(VAR)-min(VAR))

    basis_ = max(sensitivity_hist)
    multiplier = 2
    Importance = []
    for v_ in range(arg["Number of variables"]):
        Importance_ = (Higher_hist[v_]-Lower_hist[v_])/max(sensitivity_hist)
        Importance.append(Importance_)

        l_int = log(arg['MAP'][2][0]/arg['MAP'][2][0])/log(multiplier)
        if up_ < arg['MAP'][2][1]:
            h_int = log(up_/arg['MAP'][2][0])/log(multiplier)
        else:
            h_int = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)
        arg["MAP"][1][v_] = int(arg['MAP'][2][0]*(multiplier**(integer_adjustment(Importance_*(h_int-l_int)) + l_int)))

    return Lower_hist, Higher_hist, Importance


