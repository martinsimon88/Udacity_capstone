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
# [1] Ahmadi et al., 2016. An archive-based multi-objective evolutionary
# algorithm with adaptive search space partitioning to deal with expensive
# optimization problems: # application to process eco-design, Computers and
# Chemical Engineering 87 (2016) 95-110.
# [2] Ahmadi A., 2016. Memory-based Adaptive Partitioning (MAP) of search space
# for the enhancement of convergence in Pareto-based multi-objective
# evolutionary algorithms, Journal of Applied Soft Computing 41 (2016) 400-417.
# ------------------------------------------------------------------------------

### D O C U M E N T A T I O N ##################################################
# https://github.com/ArasAhmadi/AMOEA-MAP
################################################################################

"""
AMOEA-MAP: main program

INPUTS (arg):
Population size, Number of variables, Number of objectives, Benchmark,
Crossover [method, probability]

Reference point for HVI(DS) [R1,R2,...]: a reference point above the optimal
Pareto front

Termination criteria: Max function calls (computational budget), Genenration Max
(maximum number of genetic generations),

Memory-based Adaptive Partitioning (MAP): True/False

Expensive optimization: True/False

Available Benchmarks: ZDT1_2D, DTLZ1_3D, beam_design_2D

OUTPUT 1: HVI(dimension sweap), function calls
OUTPUT 2: Optimal Pareto set and front in AMOEA_MAP_Pareto_Fronts.csv
"""

import os,sys
import copy
import csv
import json

dirname, filename = os.path.split(os.path.abspath(__file__))
package_path = dirname+'/AMOEA_MAP_Package'
sys.path.append(package_path)
print package_path

from generic_calcultors import *
from SearchSpace_Partitioning import *
from AMOEA_MAP import Chromosome
from AMOEA_MAP import AMOEA_MAP_framework
from MOO_functions_ import *
from AMOEA_MAP import MultiObjectives


### G E N E R A L    S E T T I N G S ###########################################
arg = {
        "Population size" : 32,
        "Number of variables" : 8,
        "Number of objectives" : 2,
        "Constraints": True,
        "Memory-based Adaptive Partitioning (MAP)": True,
        "Expensive optimization": True,
        "Genenration Max" : 30,
        "Max function calls": 2000,
        "Crossover" : ["SBX", 1.],
        "Reference point for HVI(DS)" : [10,10],
    }

### O P T I M I Z A T I O N    P R O B L E M ###################################
Benchmark = XFoilPareto_3D(arg)
arg['Ubounds'] = Benchmark.Ubounds
arg['Lbounds'] = Benchmark.Lbounds
################################################################################

# Expensive optimization
arg['Archive Pareto size'] = arg['Population size']
arg['Mutation']=[None, None]
if arg['Expensive optimization']:
    arg['Mutation'][1] = 0.05
    arg['Mutation'][0] = "IAMO"
    arg['Archive strategy'] = True
    arg["Population size"]= 8
else:
    arg['Mutation'][0] = "Polynomial"
    arg['Mutation'][1] = 1./float(arg['Number of variables'])
    arg['Archive strategy'] = False
    arg["Population size"]= arg["Archive Pareto size"]

if arg['Memory-based Adaptive Partitioning (MAP)']:
    # min and max partitioning tendencies (PT)
    arg['MAP']=[True, [], [10,320]]

# common arg
arg["Main program path"] = dirname
arg["Results"] = [0,0,0,0,[],[]]

# parameter initialization
for i_ in range(arg["Number of variables"]):
    arg["MAP"][1].append(arg['MAP'][2][0])
tot_func_calls = []
LS_func_calls = []
GD_measure = []
IH_difference = []
diversities = []
Initialization_distance = []
Initialization_IHV = []
Benchmark.Bench_descret_matrix = {}
for i_ in range(arg["Number of variables"]):
    arg['MAP'][1][i_] = arg['MAP'][2][0]

P = []
Pareto_archive = []
arg["Results"][1] = 0
arg["Results"][0] = 0
arg["Results"][2] = 0
arg["Results"][3] = 0
arg["Results"][4] = []
arg["Results"][5] = []

# framework call
Hybrid_Optimization = AMOEA_MAP_framework(arg)

# ==============================================================================
# Population initilization
def Automatic_Initialization(arg):
    pop_size = arg["Population size"]
    n_var = arg["Number of variables"]
    for k in range(pop_size):
        P.append(Chromosome(arg))
        for m in range(n_var):
            u_ = arg['Ubounds'][m]
            l_ = arg['Lbounds'][m]
            delta_ = (u_-l_)/(pop_size-1)
            P[k].variables[m]=float(l_+k*delta_)
        # integer projection of variable space
        if arg["MAP"][0]:
            P[k].variables = Solutions_Restriction(P[k].variables,arg)
    return P

P = Automatic_Initialization(arg)
for s in P:
    s.evaluation(Benchmark)
    if s.evaluated:
        arg["Results"][1] = arg["Results"][1] + 1
# ==============================================================================

# AMOEA-MAP framewor startup
print 'HVI \t\t Function calls'
P_out = Hybrid_Optimization.start(P, Pareto_archive, arg, Benchmark)
P_out.sort(key=lambda x: x.fitness[0])

# Save optimal Pareto front
csv_file = open(dirname+'/AMOEA_MAP_Pareto_Fronts.csv', 'w')
for i in range(len(P_out)):
    for k in range(arg["Number of variables"]):
        csv_file.write(" " + str(P_out[i].variables[k]) + ", ")
    for j in range(arg["Number of objectives"]):
        csv_file.write(" " + str(P_out[i].fitness.values[j]) + ", ")
    csv_file.write(" " + str(P_out[i].constraint) + "\n")
csv_file.close()

print '\nObjective function was optimized through ' + str(int(arg['Max function calls'])) + \
        ' function calls \n' +  'Optimal Pareto results in AMOEA_MAP_Pareto_Fronts.csv \n' + \
        '--------------------------------------------------------------------------- \n' + \
        'AMOEA-MAP: a tool for expensive multiobjective optimization \n[1] Ahmadi A., Applied Soft Computing 41 (2016) 400-417 \n' + \
        '[2] Ahmadi et al., Computers and Chemical Engineering 87 (2016) 95-110 \n'




