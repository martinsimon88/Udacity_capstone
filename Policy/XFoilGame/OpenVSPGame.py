from subprocess import call
from vsp import *
import os
import config as config
import numpy as np
import vsp as vsp
import pandas as pd
import logging
testVSP= "yes"


logger = logging.getLogger(__name__)
buttons = len(config.Constraints)
print buttons



class start_gametest():

    def __init__(self):

        #self.state = self.readAoA() annab vana info

        self.steps_beyond_done = None

    def step(self, action):
        # Get new parameters for Geometry
        #file_object = open(WingBody.vspscript, w)
        #file = open("testfile.txt", "r")
        #print file.read()

        # Write new Geometry files
        # UpdateGeometry(self, actions)

        # Get new parameters for Aerodynamics setup
        # file_object = open(WingBody.vspscript, r)
        # print file.read()

        # Write new parameters for Aerodynamics setup
        self.UpdateAeroSetup(action)

        # Analysis
        #call(["vspaero", "-omp", "8", "-stab", "WingBody_DegenGeom"])
        #call(["vspaero", "-omp", "8", "WingBody_DegenGeom"])

        # Open files
        # call(["vspviewer", "WingBody_DegenGeom"])
        #call(["vsp", "WingBody.vsp3"])

        #Read reward
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        df = pd.read_csv('WingBody_DegenGeom.lod', sep='\s+', skiprows=34)
        column = np.asarray(df)

        rewardis= "reward is"
       #print rewardis, reward
        newAoA, newMach = self.readAeroData()
        reward = round(column[0, 5]/column[0, 6])*float(newAoA)
       #print "AoA and Mach number written in document is ", \
       #     newAoA, newMach

        floatAoA=float(newAoA)

        done =  floatAoA < 0 \
                or floatAoA > 25 \


        done = bool(done)
        if done is True:
            print "Hey done=done!"

        state = float(newAoA), float(newMach), 0, 0

        #print np.array(state)
        return np.array(state), reward, done, {}


    def readAeroData(self):
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        df = pd.read_csv('WingBody_DegenGeom.vspaero', sep='\s+')
        column = np.asarray(df)
        readAoA = column[6, 2]
        readMach = column[5,2]
        return readAoA, readMach


    def reset(self):
        state = np.array([6.0, 0.3, 0, 0])
        txtAoA = "6.0"
        txtMach = "0.3"
        # print "new AoA is "
        # print newAoA
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        file = open("WingBody_DegenGeom.vspaero", "w")
        file.write(
            "Sref = 143.903061\nCref = 3.918\nBref = 38.300000\nX_cg = 10.183607\nY_cg = 0.000081\nZ_cg = -0.036126\nMach = " + txtMach + "\nAoA = " + txtAoA + "\nBeta = 0.000000\nVinf = 303.8058\nRho = 0.00186846\nReCref = 6.220E+06\nClMax = -1.000000\nSymmetry = no\nFarDist = -1.000000\nNumWakeNodes = 0\nWakeIters = 5\nNumberOfRotors= 1\nPropElement_1\n1\n1.15 0 0\n1.00  0.00 0.00\n3.25\n0.5\n-2600\n0.057797173\n0.07248844  ");
        file.close()
        file = open("WingBody_DegenGeom.vspaero", "r")
        # print file.read()
        file.close()
        return state

    def UpdateAeroSetup(self, AeroSetup):
        # File basename
        AoAstepsize = 1
       #print "updating Aerosetup, old AoA and old Mach number are "
        oldAoA, oldMach = self.readAeroData()
       #print oldAoA, oldMach


        if AeroSetup == 1:
            AeroSetupV=-AoAstepsize
        elif AeroSetup==2:
            AeroSetupV=AoAstepsize
        else:
            AeroSetupV=0


       #print "AoA change is "
       #print AeroSetupV
        newMach = np.random.random_integers(2, 6)*np.random.random_integers(5, 7)/100.0
       #print "newMach"
       #print newMach
        newAoA = float(oldAoA) + AeroSetupV
        txtAoA = str(newAoA)
        txtMach = str(newMach)
       #print "new AoA is "
       #print newAoA
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        file = open("WingBody_DegenGeom.vspaero", "w")
        file.write("Sref = 143.903061\nCref = 3.918\nBref = 38.300000\nX_cg = 10.183607\nY_cg = 0.000081\nZ_cg = -0.036126\nMach = " + txtMach + "\nAoA = " + txtAoA + "\nBeta = 0.000000\nVinf = 303.8058\nRho = 0.00186846\nReCref = 6.220E+06\nClMax = -1.000000\nSymmetry = no\nFarDist = -1.000000\nNumWakeNodes = 0\nWakeIters = 5\nNumberOfRotors= 1\nPropElement_1\n1\n1.15 0 0\n1.00  0.00 0.00\n3.25\n0.5\n-2600\n0.057797173\n0.07248844  ");
        file.close()
        file = open("WingBody_DegenGeom.vspaero", "r")
       #print file.read()
        file.close()



    def UpdateGeometry(self, actions):

        # OpenVSP Script Part
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        stdout = vsp.cvar.cstdout
        errorMgr = vsp.ErrorMgrSingleton_getInstance()

        # WingBodyTestCase

        vsp.VSPRenew();
        errorMgr.PopErrorAndPrint(stdout)

        # Add Wing
        WingBody = vsp.AddGeom("WING", "")

        # Insert A Couple More Sections
        InsertXSec(WingBody, 1, XS_FOUR_SERIES);
        InsertXSec(WingBody, 1, XS_FOUR_SERIES);
        InsertXSec(WingBody, 1, XS_FOUR_SERIES);

        # Cut The Original Section
        CutXSec(WingBody, 1);

        # Change Driver
        SetDriverGroup(WingBody, 1, AREA_WSECT_DRIVER, ROOTC_WSECT_DRIVER, TIPC_WSECT_DRIVER);

        SetParmVal(WingBody, "RotateAirfoilMatchDideralFlag", "WingGeom", 1.0);

        # Change Some Parameters 1st Section

        SetParmVal(WingBody, "Root_Chord", "XSec_1", 7.0);
        SetParmVal(WingBody, "Tip_Chord", "XSec_1", 3.0);
        SetParmVal(WingBody, "Area", "XSec_1", actions[0]);
        SetParmVal(WingBody, "Sweep", "XSec_1", actions[1]);

        # Because Sections Are Connected Change One Section At A Time Then Update
        Update();

        # Change Some Parameters 2nd Section
        SetParmVal(WingBody, "Tip_Chord", "XSec_2", 2.0);
        SetParmVal(WingBody, "Sweep", "XSec_2", 60.0);
        SetParmVal(WingBody, "Dihedral", "XSec_2", 30.0);
        Update();

        # Change Some Parameters 3rd Section
        SetParmVal(WingBody, "Sweep", "XSec_3", 60.0);
        SetParmVal(WingBody, "Dihedral", "XSec_3", 80.0);
        Update();

        # Change Airfoil
        SetParmVal(WingBody, "Camber", "XSecCurve_0", 0.02);
        Update();

        ##print "All geoms in Vehicle."
        geoms = vsp.FindGeoms()
       #print geoms

        # File basename
        baseName = "WingBody";
        csvName = baseName + "_Dege   nGeom.csv";
        stlName = baseName + "_DegenGeom.stl";
        vspName = baseName + ".vsp3";

        # Set File Name
        SetComputationFileName(DEGEN_GEOM_CSV_TYPE, csvName);
        SetComputationFileName(CFD_STL_TYPE, stlName);

        WriteVSPFile(vspName)
        # ComputeDegenGeom( SET_ALL, DEGEN_GEOM_CSV_TYPE );
        # ComputeCFDMesh( SET_ALL, CFD_STL_TYPE );        # Mesh

        # Check for errors

        num_err = errorMgr.GetNumTotalErrors()
        for i in range(0, num_err):
            err = errorMgr.PopLastError()
            print "error = ", err.m_ErrorString



