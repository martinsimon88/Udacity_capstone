from subprocess import call
from vsp import *
import os
import config as config
import numpy as np
import vsp as vsp
import math
import logging
testVSP= "yes"

logger = logging.getLogger(__name__)
buttons = len(config.Constraints)
print buttons

#file_object = open("WingBody.vspscript", "r")
#print file_object


class start_gametest():

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        #self.action_space = spaces.Discrete(2)
        #self.observation_space = spaces.Box(-high, high)

        #self._seed()
        #self.viewer = None
        self.state = None

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
        # UpdateGeometry(self, AeroSetup)

        # Analysis
        # call(["vspaero", "-omp", "8", "-stab", "WingBody_DegenGeom"])
        # call(["vspaero", "-omp", "8", "WingBody_DegenGeom"])

        # Open files
        # call(["vspviewer", "WingBody_DegenGeom"])
        #call(["vsp", "WingBody.vsp3"])

        # call(["ls"])

        # Write results
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if action>1.1:
            reward=10000

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

        if action>0.9:
            reward=10000
        return np.array(self.state), reward, done, {}



        '''

        a=action
        s1=(a+a)*(a+a), a+a, a, -a
        reward=10-a
        if s1 < 10
            done=True

        return s1, reward, done, {}
        '''
    def reset(self):
        self.state = np.array([0,0,0,0])

        #state = np.array([-0.01968701, 0.0227853, 0.01934283, 0.02938217])
        return self.state






    #Initiate the "game":
    def start_game(self, actions):


        #Get new parameters for Geometry
        #file_object = open(WingBody.vspscript, w)
        #file = open("testfile.txt", "r")
        #print file.read()

        #Write new Geometry files
        #UpdateGeometry(self, actions)

        #Get new parameters for Aerodynamics setup
        #file_object = open(WingBody.vspscript, r)
        #print file.read()


        #Write new parameters for Aerodynamics setup
        #UpdateGeometry(self, AeroSetup)

        #Analysis
        #call(["vspaero", "-omp", "8", "-stab", "WingBody_DegenGeom"])
        #call(["vspaero", "-omp", "8", "WingBody_DegenGeom"])

        #Open files
        #call(["vspviewer", "WingBody_DegenGeom"])
        call(["vsp", "WingBody.vsp3"])

        #call(["ls"])

        #Write results



    def UpdateAeroSetup(self, AeroSetup):
        # File basename

        Sref = "Sref = " + AeroSetup[0];
        Cref = baseName + "_Dege   nGeom.csv";
        stlName = baseName + "_DegenGeom.stl";
        vspName = baseName + ".vsp3";

        Sref = 143.903061
        Cref = 3.918
        Bref = 38.300000
        X_cg = 10.183607
        Y_cg = 0.000081
        Z_cg = -0.036126
        Mach = 0.27992
        AoA = 0.23023
        Beta = 0.000000
        Vinf = 303.8058
        Rho = 0.00186846
        ReCref = 6.220E+06
        ClMax = -1.000000
        Symmetry = no
        FarDist = -1.000000
        NumWakeNodes = 0
        WakeIters = 5
        NumberOfRotors = 1
        PropElement_1
        1
        1.15
        0
        0
        1.00
        0.00
        0.00
        3.25
        0.5
        -2600
        0.057797173
        0.07248844


        os.chdir("/home/simonx/Documents/Udacity/ML/Projects/capstone/OpenVSP")
        file = open("WingBody_DegenGeom2.vspaero", "w")
        file.write("Hello World\n");
        file.write("This is our new text file\n");
        file.write("and this is another line.\n");
        file.write("Why? Because we can.\n");
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



