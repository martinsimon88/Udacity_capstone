from subprocess import call
import os
os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
call(["ls"])

call(["vsp", "-script", "TestAll.vspscript"])
