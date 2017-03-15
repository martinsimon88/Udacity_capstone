# This file is to set conditions for the BWB UAV Reinforcement policy
# It doesn't matter if you use capital letters or not.
# It doesn't matter if you use underscore or camel notation for keys, e.g. episode_timeout is the same as episodeTimeout.

from vsp import *

# Input parameters
InputParameters = [
    #	PayloadWeight,
    #	PayloadDimensions,
    #    AoA
    ]


# Output parameters
OutputParameters = [
    #    Cp,
    #    Cd,
    #    Cl,
    #    Cstab
     ]

# PolicyParameters
AoAmin=0
AoAmax=20
Constraints = [
       AoAmin,
       AoAmax
]