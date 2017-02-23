from subprocess import call
#from paraview.simple import *
#import pvpython
#connection = paraview.Connect()
#import paraview
import os
os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")

#call(["pvpython"])
#call(["vsp"])

#call(["vsp", "Cone.vsppart", "-script", "example.vspscript"])

call(["vsp", "-script", "WingBody.vspscript"])
call(["vspaero", "-omp", "8", "-stab", "WingBody_DegenGeom"])
call(["vspaero", "-omp", "8", "WingBody_DegenGeom"])
call(["vspviewer", "WingBody_DegenGeom"])
call(["ls"])