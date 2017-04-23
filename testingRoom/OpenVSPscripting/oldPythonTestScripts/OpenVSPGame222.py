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

os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
df = pd.read_csv('WingBody_DegenGeom.vspaero', sep='\s+')
column = np.asarray(df)
oldAoA = column[6,2]

class start_gametest():

    def __init__(self):

        self.state = AoA

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
        reward = round(column[0, 5]/column[0, 6])
        rewardis= "reward is"
        print rewardis, reward
        print "New AoA is ", \
            self.UpdateAeroSetup.newAoA
        done = bool(1)

        '''
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)



        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        '''

        return np.array(self.state), reward, done, {}


    def reset(self):
        self.state = np.array([0,0,0,0])
        return self.state

    def UpdateAeroSetup(self, AeroSetup):
        # File basename
        print "updating Aerosetup old AoA"
        print oldAoA

        if AeroSetup == 1:
            AeroSetupV=-0.5
            print AeroSetupV
        elif AeroSetup==2:
            AeroSetupV=0.5
            print AeroSetupV
        else:
            AeroSetupV=0
            print AeroSetupV

        newAoA = float(oldAoA) + AeroSetupV
        txtAoA = str(newAoA)
        print "new AoA is "
        print newAoA
        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        file = open("WingBody_DegenGeom.vspaero", "w")
        file.write(
            "Sref = 143.903061\nCref = 3.918\nBref = 38.300000\nX_cg = 10.183607\nY_cg = 0.000081\nZ_cg = -0.036126\nMach = 0.27992\nAoA = " + txtAoA + "\nBeta = 0.000000\nVinf = 303.8058\nRho = 0.00186846\nReCref = 6.220E+06\nClMax = -1.000000\nSymmetry = no\nFarDist = -1.000000\nNumWakeNodes = 0\nWakeIters = 5\nNumberOfRotors= 1\nPropElement_1\n1\n1.15 0 0\n1.00  0.00 0.00\n3.25\n0.5\n-2600\n0.057797173\n0.07248844  ");
        file.close()
        file = open("WingBody_DegenGeom.vspaero", "r")
        print file.read()
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

        # print "All geoms in Vehicle."
        geoms = vsp.FindGeoms()
        print geoms

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



