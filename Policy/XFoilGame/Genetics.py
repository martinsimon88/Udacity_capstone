
import random
from math import *
from Gen2Airfoil import *
from xfoil_dat import *
import sys
import os

# remove all files from last run
filelist = [ f for f in os.listdir(".") if f.endswith(".dat") ]
for f in filelist:
    os.remove(f)
#    print f   
filelist = [ f for f in os.listdir(".") if f.endswith(".log") ]
for f in filelist:
    os.remove(f)


# Airfoil is defined by
#  LEU = Leading edge up            LED = Leading edge down      
#  C25 = Camber at 25%              T25 = Camber at 25%
#  C50 = Camber at 50%              T50 = Camber at 50%
#  C75 = Camber at 75%              T75 = Camber at 75%

#              LEU   LED     C25   C50    C75      T25   T50   T75
genmaxs = [    0.2,  0.2,    0.2,  0.2,   0.2,     0.2,  0.2,  0.1   ]
genmins = [    0.0,  0.0,    0.0,  0.0,   0.0,     0.0,  0.0,  0.0   ]

ngen= len (genmaxs)

foilnum = 0

# ==============================================
# ============== BASIC FUNCTIONS ===============
# ==============================================

def newborn(gen):
    global foilnum
    foilnum+=1
    return [0,gen,'%06d' %foilnum]

def check_range(gen):
    for i in range(0,ngen):
        if(gen[i]>genmaxs[i]):
            gen[i]=genmaxs[i]
        if(gen[i]<genmins[i]):
            gen[i]=genmins[i]

def proximity(gen1,gen2):
    proximity = 0
    for i in range(0,ngen):
        d = gen1[i] - gen2[i]
        proximity+= d*d
    return sqrt(proximity)

def breed_random():
    child = [0]*ngen
    for i in range(0,ngen):
        child[i] = random.uniform(genmins[i],genmaxs[i])
        check_range(child)
    return newborn(child)

def breed_interpolate(mother,father,weight):
    child = [0]*ngen
    for i in range(0,ngen):
        child[i] = (1-weight)*mother[i] + weight*father[i]
    return newborn(child)

def breed_crossover(mother,father):
    child = [0]*ngen
    for i in range(0,ngen):
        if(random.random()>0.5):
            child[i] = mother[i]
        else:
            child[i] = father[i]
    return newborn(child)

def breed_mutate(mother, scale):
    child = [0]*ngen
    for i in range(0,ngen):
        child[i] = mother[i] * (1 + random.uniform(-scale,scale) )
        check_range(child)
    return newborn(child)

def gen2log(gen,logfile):
    #logfile.write(" %3.2f          " %gen[0])
    logfile.write (gen[2])
    logfile.write (' {0:10.2f}       '.format(gen[0]))
    for i in range(0,ngen):
        logfile.write(" %1.4f " %gen[1][i])
    logfile.write("\n")

#              LEU   LED     C25   C50    C75      T25   T50   C75
testgen = [    0.5,  0.1,    0.1,  0.1,   0.3,     0.5,  0.6,  0.7   ]

#print testgen
#check_range(testgen)
#print
#print testgen
#print breed_random()



# ==============================================
# ============== Algorithm ===============
# ==============================================

nsubjects = 20
nbest     = 10
name = "testfoil"

niterations = 15
Re = 1000000
Ncrit = 9.0

logfile = open("logfile.log","w")



population = [[0,[],""] for i in range(0,nsubjects)]
print len(population)
fitness = [0]*nsubjects

def populate():
    for i in range (0,nsubjects):
        #foilnum++
        population[i] = breed_random()
        #name = '%06d' %foilnum
        gen2airfoil(population[i][1],population[i][2]) # generate Airfoil shape by Bezier interpolation
        Xfoil(population[i][2],Ncrit,Re)                   # compute fittness in Xfoil
        population[i][0] = getLDmax(population[i][2])      # set fittness = LD
        print i,population[i][2],population[i][0]
        gen2log(population[i],logfile)

def eval_fitness():
    for i in range (nbest,nsubjects):          # evaluate fittens just for new ones
        gen2airfoil(population[i][1],population[i][2]) # generate Airfoil shape by Bezier interpolation
        Xfoil(population[i][2],Ncrit,Re)                   # compute fittness in Xfoil
        population[i][0] = getLDmax(population[i][2])      # set fittness = LD
        print i,population[i][0]
        gen2log(population[i],logfile)

def evolve():
    population.sort( reverse=True)
    for i in range (0,nbest):
        print "survive: ",population[i][2],"      ",population[i][0]
    population[nbest+ 0 ] =  breed_mutate(population[0][1],0.1)     # modify 1 best
    population[nbest+ 5 ] =  breed_random()     # generate new from scratch
    population[nbest+ 1 ] =  breed_crossover  ( population[0][1] ,population[1][1])      # crossover 2 best
    population[nbest+ 2 ] =  breed_interpolate( population[0][1] ,population[1][1],0.5)  # interpolation 2 best
    population[nbest+ 3 ] =  breed_crossover  ( population[0][1] ,population[random.randint(2,nbest-1)][1]    )  # crossover 1 and other
    population[nbest+ 4 ] =  breed_interpolate( population[0][1] ,population[random.randint(2,nbest-1)][1],0.5)  # interpolation 1 and other
    for i in range (nbest+ 6 ,nsubjects):
        a =  random.randint(0,nbest-1)   # choose any 2
        b =  random.randint(0,nbest-1)
        if(random.random()>0.5):   # interpolate or crossover, 1:1 chance
            population[i] = breed_crossover  ( population[a][1] ,population[b][1])
        else:
            population[i] = breed_interpolate  ( population[a][1] ,population[b][1], 0.5)

print " ===== iteration: 0"
populate()
for i in range (1,niterations):
    print " ===== iteration: ",i
    evolve()
    eval_fitness()
logfile.close()

#print "======================="
#population.sort()
#for i in range (0,nsubjects):
#    print population[i][0]
#    print population[i][1]
#    print


