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

""" AMOEA-MAP """

import sys, random, math, copy
import json
from MOO_functions_ import *
from operators import *
from generic_calcultors import *
from SearchSpace_Partitioning import *

class Chromosome():
    def __init__(self, arg):
        self.arg = arg
        self.evaluated = None
        self.Num_Obj = arg["Number of objectives"]
        self.variables = []
        self.maximize = False
        self.Lsup = arg["Lbounds"]
        self.Usup = arg["Ubounds"]
        self.rank = sys.maxint
        self.distance = 0.0
        self.Num_Var = arg["Number of variables"]
        for i in range(self.Num_Var):
            self.variables.append(random.random()*(self.Usup[i]-self.Lsup[i]) + self.Lsup[i])
        # MAP
        if arg["MAP"][0]:
            self.variables = Solutions_Restriction(self.variables,arg)
        self.Crossover_method = arg["Crossover"][0]
        self.Mutation_method = arg["Mutation"][0]
        self.Cross_rate = float(arg["Crossover"][1])
        self.Mute_rate = float(arg["Mutation"][1])
        objective_vector = []
        del objective_vector[:]
        for k in range(self.Num_Obj):
            objective_vector.append(0.0)
        self.fitness = MultiObjectives(objective_vector,0.0)
        self.constraint = 0.0

    def __setattr__(self, name, val):
        if name == 'variables':
            self.__dict__[name] = val
            #self.fitness = None
        else:
            self.__dict__[name] = val
    def __str__(self):
        return '%s : %s' % (str(self.variables), str(self.fitness), str(self.constraint))
    def __repr__(self):
        return '<Individual: candidate = %s, fitness = %s, constraint = %s>' % ( str(self.variables), str(self.fitness), str(self.constraint))
    def __lt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            if self.maximize:
                return self.fitness < other.fitness
            else:
                return self.fitness > other.fitness
        else:
            raise Exception('fitness is not defined')
    def __le__(self, other):
        return self < other or not other < self
    def __gt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            return other < self
        else:
            raise Exception('fitness is not defined')
    def __ge__(self, other):
        return other < self or not self < other
    def __lshift__(self, other):
        return self < other
    def __rshift__(self, other):
        return other < self
    def __ilshift__(self, other):
        raise TypeError("unsupported operand type(s) for <<=: 'Individual' and 'Individual'")
    def __irshift__(self, other):
        raise TypeError("unsupported operand type(s) for >>=: 'Individual' and 'Individual'")
    def __eq__(self, other):
        return self.variables == other.variables
    def __ne__(self, other):
        return self.variables != other.variables

    def evaluation(self, Benchmark):
        string_ind = str(self.variables)
        if string_ind in Benchmark.Bench_descret_matrix.keys() and self.arg["MAP"][0]:
            # memory use
            objective_vector = Benchmark.Bench_descret_matrix[string_ind].fitness.values
            self.constraint = Benchmark.Bench_descret_matrix[string_ind].constraint
            self.fitness = MultiObjectives(objective_vector,self.constraint)
            self.evaluated = False
        else:
            # memory creation
            objective_vector = Benchmark.evaluate(self)
            self.fitness = MultiObjectives(objective_vector,self.constraint)
            Benchmark.Bench_descret_matrix[string_ind] = self
            self.evaluated = True

class MultiObjectives(object):
    def __init__(self, values=[], const=[], maximize=True):
        self.values = values
        try:
            iter(maximize)
        except TypeError:
            maximize = [maximize for v in values]
        self.maximize = maximize
        self.const = const
    def __len__(self):
        return len(self.values)
    def __getitem__(self, key):
        return self.values[key]
    def __iter__(self):
        return iter(self.values)
    def __lt__(self, other):
        if len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            not_worse = True
            strictly_better = False
            domination = False
            x_c = self.const
            y_c = other.const
            if x_c<>0 and y_c<>0:
                if x_c < y_c:
                    domination = True
            elif x_c==0 and y_c<>0:
                domination = True
            elif x_c==0 and y_c==0:
                for x, y, m in zip(self.values, other.values, self.maximize):
                    if m:
                        if x > y:
                            not_worse = False
                        elif y > x:
                            strictly_better = True
                    else:
                        if x < y:
                            not_worse = False
                        elif y < x:
                            strictly_better = True
                domination = not_worse and strictly_better
            return domination

    def __le__(self, other):
        return self < other or not other < self
    def __gt__(self, other):
        return other < self
    def __ge__(self, other):
        return other < self or not self < other
    def __eq__(self, other):
        return self.values == other.values
    def __str__(self):
        return str(self.values)
    def __repr__(self):
        return str(self.values)

class AMOEA_MAP_framework:
    def __init__(self, arg):
        self.Num_Obj = arg["Number of objectives"]
        self.memory_use = 0
        self.Mute_rate = float(arg["Mutation"][1])
        self.Cross_rate = float(arg["Crossover"][1])
        self.Pop_Size = arg["Population size"]
        self.Gen_Max = arg["Genenration Max"]
        self.Num_Var = arg["Number of variables"]
        self.Ref_point = arg["Reference point for HVI(DS)"]
        random.seed();

    def start(self, P, Pareto_archive, arg, Benchmark):
        Q = []
        fmax = []
        fmin = []
        temp = []
        HVI_DS = []
        mean1 = []
        Pareto_archive_survived = []
        rLower_b = []
        rHigher_b = []

        up_ = arg['MAP'][2][0]
        Importance = []
        for i_ in range(arg['Number of variables']):
            Importance.append(1.0)
        arg['Amplitude'] = []
        arg['memory use']=0

        ## Generations
        for i in range(self.Gen_Max):
            arg["Current generation"] = i
            R = []
            del R[:]
            R.extend(P)
            R.extend(Q)

            # bi-population strategy in AMOEA
            del P[:]
            Pareto_archive.extend(R)
            Pareto_archive = self.fast_non_dominated_sorting(Pareto_archive,arg['Archive Pareto size'])
            if arg['Archive Pareto size'] == self.Pop_Size:
                P.extend(Pareto_archive)
            else:
                Pareto_archive_survived = []
                Pareto_archive_survived = self.fast_non_dominated_sorting(Pareto_archive,self.Pop_Size)
                P.extend(Pareto_archive_survived)

            ## Pareto evolution ------------------------------------------------
            stall = 3
            if self.Num_Obj>1:
                # Hyper volume calculations via dimension-sweep algorithm
                temp_P = []
                del temp_P[:]
                if arg['Archive Pareto size'] == self.Pop_Size:
                    for x_ in P:
                        temp_P.append(x_.fitness.values)
                    hyper_volume_ind1 = hyper_volume_Dsweep(temp_P,arg)
                else:
                    for x_ in Pareto_archive:
                        temp_P.append(x_.fitness.values)
                    hyper_volume_ind1 = hyper_volume_Dsweep(temp_P,arg)
                HVI_DS.append(hyper_volume_ind1)
                mean_value = sum(HVI_DS[i-stall:])/(stall+1)
            else:
                temp_mean = 0
                for x_ in P:
                    temp_mean = temp_mean + (x_.fitness.values[0]/(len(P)))
                mean1.append(temp_mean)
                mean_value = sum(mean1[i-stall:])/(stall+1)

            # normalized standard deviation
            deviation = 0.0
            if mean_value>1.e-15 and i>=stall-1:
                if self.Num_Obj>1:
                    for nu_ in range(stall+1):
                        deviation = deviation + (mean_value-HVI_DS[i-stall+nu_])**2.0 / mean_value**2.0
                    deviation = (deviation/stall)**0.5
                else:
                    for nu_ in range(stall+1):
                        deviation = deviation + (mean_value-mean1[i-stall+nu_])**2.0 / mean_value**2.0
                    deviation = (deviation/stall)**0.5
            else:
                deviation = 1e2
            ## -----------------------------------------------------------------

            ## Dynamic adaptive partitioning -----------------------------------
            deviation_limit = 1.e-2
            if arg["Expensive optimization"]:
                if arg["MAP"][0]:
                    multiplier = 2
                    max_ = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)
                    beta_ = random.random()
                    if deviation<deviation_limit:
                        k_ = log(up_/arg['MAP'][2][0])/log(multiplier)
                        if beta_ < (0.5-0.1*k_):
                            if up_<arg['MAP'][2][1]:
                                up_ = up_*multiplier
                        else:
                            if up_>arg['MAP'][2][0]:
                                up_ = up_/multiplier
                    rLower_b, rHigher_b, Importance = Dynamic_Parameter_Refining(P,up_,arg)
            else:
                if arg["MAP"][0]:
                    multiplier = 2
                    max_ = log(arg['MAP'][2][1]/arg['MAP'][2][0])/log(multiplier)+1
                    beta_ = random.random()
                    if deviation<deviation_limit:
                        if beta_>((log(up_/arg['MAP'][2][0])/log(multiplier))/max_):
                            if up_<arg['MAP'][2][1]:
                                up_ = up_*multiplier
                        else:
                            if up_>arg['MAP'][2][0]:
                                up_ = up_/multiplier
                    rLower_b, rHigher_b, Importance = Dynamic_Parameter_Refining(P,up_,arg)

            arg['Importance'] = Importance
            ## -----------------------------------------------------------------

            if arg["Results"][1]>arg['Max function calls']:
                if self.Num_Obj>1:
                    print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1]
                    arg["Results"][4].append([hyper_volume_ind1,arg["Results"][1]])
                    break
                else:
                    print '%.4f' % temp_mean, arg["Results"][1]
                    break

            del fmax[:]
            del fmin[:]
            for j in range(self.Num_Obj):
                del temp[:]
                for x_ in P:
                    temp.append(x_.fitness.values[j])
                fmax.append(max(temp))
                fmin.append(min(temp))

            # constraint violations
            temp__ = []
            for jh_ in P:
                temp__.append(jh_.constraint)
            const_violation = sum(temp__)/self.Pop_Size

            arg["Current function calls"] = arg["Results"][1]
            if self.Num_Obj>1:
                print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1]
                arg["Results"][4].append([hyper_volume_ind1,arg["Results"][1]])
            else:
                print '%.4f' % temp_mean, arg["Results"][1]

            # Regenerating new children by means of genetic operators
            Q = self.regeneration(P, arg, Benchmark)

        if self.Num_Obj>1:
            print '%.4f' % hyper_volume_ind1,' \t ',arg["Results"][1]
            arg["Results"][3] = hyper_volume_ind1
        else:
            print '%.4f' % temp_mean, arg["Results"][1]

        if arg['Archive Pareto size'] == self.Pop_Size:
            return P
        else:
            return Pareto_archive

    def regeneration(self, P, arg, Benchmark):
        # Regenerate a new population
        Q = []
        # selection
        selected_parents = tournament_selection(P, self.Pop_Size, arg)
        selected_parents_var = [copy.deepcopy(i.variables) for i in selected_parents]
        # crossover
        children_crossedover_var = crossover_operator(selected_parents_var, arg)
        # mutation
        children_mutated_var = mutation_operator(children_crossedover_var, arg)
        for i in range(len(children_mutated_var)):
            child = Chromosome(arg)
            child.variables = children_mutated_var[i]
            child.evaluation(Benchmark)
            if child.evaluated:
                arg["Results"][1] = arg["Results"][1] + 1
            else:
                arg["memory use"] = arg["memory use"] + 1
            Q.append(child)
        return Q

    def fast_non_dominated_sorting(self, combined, archive_size):
        # Fast non-dominated sorting
        survivors = []
        fronts = []
        pop = set(range(len(combined)))
        while len(pop) > 0:
            front = []
            for p in pop:
                dominated = False
                for q in pop:
                    if combined[p] < combined[q]:
                        dominated = True
                        break
                if not dominated:
                    front.append(p)
            fronts.append([dict(individual=combined[f], index=f) for f in front])
            pop = pop - set(front)

        for i, front in enumerate(fronts):
            if len(survivors) + len(front) > archive_size:
                # Determine the crowding distance.
                distance = [0 for _ in range(len(combined))]
                individuals = front[:]
                num_individuals = len(individuals)
                num_objectives = len(individuals[0]['individual'].fitness)
                for obj in range(num_objectives):
                    individuals.sort(key=lambda x: x['individual'].fitness[obj])
                    distance[individuals[0]['index']] = float('inf')
                    distance[individuals[-1]['index']] = float('inf')
                    for i in range(1, num_individuals-1):
                        distance[individuals[i]['index']] = (distance[individuals[i]['index']] +
                                                             (individuals[i+1]['individual'].fitness[obj] -
                                                              individuals[i-1]['individual'].fitness[obj]))

                crowd = [dict(dist=distance[f['index']], index=f['index']) for f in front]
                crowd.sort(key=lambda x: x['dist'], reverse=True)
                last_rank = [combined[c['index']] for c in crowd]
                r = 0
                num_added = 0
                num_left_to_add = archive_size - len(survivors)
                while r < len(last_rank) and num_added < num_left_to_add:
                    if last_rank[r] not in survivors:
                        survivors.append(last_rank[r])
                        num_added += 1
                    r += 1
                if len(survivors) == archive_size:
                    break
            else:
                for f in front:
                    if f['individual'] not in survivors:
                        survivors.append(f['individual'])
        return survivors

# Tournament selection with constraints
def tournament_selection(pop_, pop_size, arg):
    num_selected = pop_size
    tourn_size = 2
    pop__ = list(pop_)
    selected = []
    for _ in range(num_selected):
        tourn = random.sample(pop__, tourn_size)
        if arg['Constraints']:
            if tourn[0].constraint==0 and tourn[1].constraint==0:
                selected.append(max(tourn))
            if tourn[0].constraint==0 and tourn[1].constraint<>0:
                selected.append(tourn[0])
            if tourn[0].constraint<>0 and tourn[1].constraint==0:
                selected.append(tourn[1])
            if tourn[0].constraint<>0 and tourn[1].constraint<>0:
                if tourn[0].constraint <= tourn[1].constraint:
                    selected.append(tourn[0])
                else:
                    selected.append(tourn[1])
        else:
            selected.append(max(tourn))
    return selected



