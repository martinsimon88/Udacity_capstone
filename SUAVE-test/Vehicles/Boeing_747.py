# Boeing_747.py
#
# Created:  Feb 2017, M. Vegh (taken from data originally in cmalpha/cmalpha.py and cnbeta/cnbeta.py)
# Modified: 

""" setup file for the Boeing 747 vehicle
note that it does not include an engine; current values only used to test stability cmalpha and cnbeta
"""

# ----------------------------------------------------------------------
#   Imports
# ----------------------------------------------------------------------

import numpy as np
import SUAVE
from SUAVE.Core import Units
from SUAVE.Core import (
    Data, Container,
)
from SUAVE.Methods.Geometry.Three_Dimensional.compute_span_location_from_chord_length import compute_span_location_from_chord_length
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.datcom import datcom
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.trapezoid_ac_x import trapezoid_ac_x
from SUAVE.Methods.Flight_Dynamics.Static_Stability.Approximations.Supporting_Functions.extend_to_ref_area import extend_to_ref_area

def vehicle_setup():

    vehicle = SUAVE.Vehicle()
    #print vehicle
    vehicle.mass_properties.max_zero_fuel=238780*Units.kg
    vehicle.mass_properties.max_takeoff  =785000.*Units.lbs
    
    # ------------------------------------------------------------------        
    #   Main Wing
    # ------------------------------------------------------------------        

    wing = SUAVE.Components.Wings.Main_Wing()
    wing.tag = 'main_wing'
    wing.areas.reference           = 5500.0 * Units.feet**2
    wing.spans.projected           = 196.0  * Units.feet
    wing.chords.mean_aerodynamic   = 27.3   * Units.feet
    wing.chords.root               = 42.9   * Units.feet  #54.5ft
    wing.chords.tip                = 14.7   * Units.feet
    wing.sweeps.quarter_chord      = 42.0   * Units.deg  # Leading edge
    wing.sweeps.leading_edge       = 42.0   * Units.deg  # Same as the quarter chord sweep (ignore why EMB)
    wing.taper                     = wing.chords.tip / wing.chords.root
    
    wing.aspect_ratio              = wing.spans.projected**2/wing.areas.reference
    wing.symmetric      = True
    wing.vertical       = False
    wing.origin         = np.array([58.6,0,3.6]) * Units.feet  
    wing.aerodynamic_center     = np.array([112.2*Units.feet,0.,0.])-wing.origin#16.16 * Units.meters,0.,0,])
    wing.dynamic_pressure_ratio = 1.0
    wing.ep_alpha               = 0.0
    
    span_location_mac                         = compute_span_location_from_chord_length(wing, wing.chords.mean_aerodynamic)
    mac_le_offset                             = .8*np.sin(wing.sweeps.leading_edge)*span_location_mac  #assume that 80% of the chord difference is from leading edge sweep
    wing.mass_properties.center_of_gravity[0] = .3*wing.chords.mean_aerodynamic+mac_le_offset
    
    
    Mach                         = np.array([0.198])
    conditions                   = Data()
    conditions.weights           = Data()
    conditions.lift_curve_slope  = datcom(wing,Mach)
    conditions.weights.total_mass=np.array([[vehicle.mass_properties.max_takeoff]]) 
   
    wing.CL_alpha                = conditions.lift_curve_slope
    vehicle.reference_area       = wing.areas.reference
    vehicle.append_component(wing)
    
    main_wing_CLa = wing.CL_alpha
    main_wing_ar  = wing.aspect_ratio
    
    # ------------------------------------------------------------------        
    #  Horizontal Stabilizer
    # ------------------------------------------------------------------        
    
    
    wing                        = SUAVE.Components.Wings.Wing()
    wing.tag                    = 'horizontal_stabilizer'
    wing.areas.reference        = 1490.55* Units.feet**2
    wing.spans.projected        = 71.6   * Units.feet
    wing.sweeps.quarter_chord   = 44.0   * Units.deg # leading edge
    wing.sweeps.leading_edge    = 44.0   * Units.deg # Same as the quarter chord sweep (ignore why EMB)
    wing.taper                  = 7.5/32.6
    wing.aspect_ratio           = wing.spans.projected**2/wing.areas.reference
    wing.origin                 = np.array([187.0,0,0])  * Units.feet
    wing.symmetric              = True
    wing.vertical               = False
    wing.dynamic_pressure_ratio = 0.95
    wing.ep_alpha               = 2.0*main_wing_CLa/np.pi/main_wing_ar    
    wing.aerodynamic_center     = [trapezoid_ac_x(wing), 0.0, 0.0]
    wing.CL_alpha               = datcom(wing,Mach)
    vehicle.append_component(wing)
    
    # ------------------------------------------------------------------
    #   Vertical Stabilizer
    # ------------------------------------------------------------------
    
    wing = SUAVE.Components.Wings.Wing()
    wing.tag                  = 'vertical_stabilizer'
    wing.spans.exposed        = 32.4  * Units.feet
    wing.chords.root          = 38.7  * Units.feet      # vertical.chords.fuselage_intersect
    wing.chords.tip           = 13.4  * Units.feet
    wing.sweeps.quarter_chord = 50.0  * Units.deg # Leading Edge
    wing.x_root_LE1           = 180.0 * Units.feet
    wing.symmetric            = False
    wing.exposed_root_chord_offset = 13.3   * Units.feet
    wing                      = extend_to_ref_area(wing)
 
    wing.areas.reference        = wing.extended.areas.reference
    wing.spans.projected        = wing.extended.spans.projected
    wing.chords.root            = 14.9612585185
    dx_LE_vert                  = wing.extended.root_LE_change
    wing.taper                  = 0.272993077083
    wing.origin                 = np.array([wing.x_root_LE1 + dx_LE_vert,0.,0.])
    wing.aspect_ratio           = (wing.spans.projected**2)/wing.areas.reference
    wing.effective_aspect_ratio = 2.2
    wing.symmetric              = False
    wing.aerodynamic_center     = np.array([trapezoid_ac_x(wing),0.0,0.0])
    wing.dynamic_pressure_ratio = .95
    Mach                        = np.array([0.198])
    wing.CL_alpha               = 0.
    wing.ep_alpha               = 0.
    vehicle.append_component(wing)
    
    
    # ------------------------------------------------------------------
    #  Fuselage
    # ------------------------------------------------------------------
    
    fuselage = SUAVE.Components.Fuselages.Fuselage()
    fuselage.tag = 'fuselage'
    
    fuselage.lengths.total                      = 229.7   * Units.feet
    fuselage.areas.side_projected               = 4696.16 * Units.feet**2 #used for cnbeta
    fuselage.heights.maximum                    = 26.9    * Units.feet    #used for cnbeta
    fuselage.heights.at_quarter_length          = 26.0    * Units.feet    #used for cnbeta
    fuselage.heights.at_three_quarters_length   = 19.7    * Units.feet    #used for cnbeta
    fuselage.heights.at_wing_root_quarter_chord = 23.8    * Units.feet    #used for cnbeta
    
    fuselage.x_root_quarter_chord               = 77.0    * Units.feet    #used for cmalpha
    fuselage.lengths.total                      = 229.7   * Units.feet
    fuselage.width                              = 20.9    * Units.feet 
    
     
    
    vehicle.append_component(fuselage)
    vehicle.mass_properties.center_of_gravity=np.array([112.2,0,0]) * Units.feet  
    
    
    
 
    #configuration.mass_properties.zero_fuel_center_of_gravity=np.array([76.5,0,0])*Units.feet #just put a number here that got the expected value output; may want to change
    fuel                                                     =SUAVE.Components.Physical_Component()
    fuel.origin                                              =wing.origin
    fuel.mass_properties.center_of_gravity                   =wing.mass_properties.center_of_gravity
    fuel.mass_properties.mass                                =vehicle.mass_properties.max_takeoff-vehicle.mass_properties.max_zero_fuel
   
    
    #find zero_fuel_center_of_gravity
    cg                   =vehicle.mass_properties.center_of_gravity
    MTOW                 =vehicle.mass_properties.max_takeoff
    fuel_cg              =fuel.origin+fuel.mass_properties.center_of_gravity
    fuel_mass            =fuel.mass_properties.mass
    
    
    sum_moments_less_fuel=(cg*MTOW-fuel_cg*fuel_mass)
    vehicle.fuel = fuel
    vehicle.mass_properties.zero_fuel_center_of_gravity = sum_moments_less_fuel/vehicle.mass_properties.max_zero_fuel
    return vehicle
  
def configs_setup(vehicle):
     # ------------------------------------------------------------------
    #   Initialize Configurations
    # ------------------------------------------------------------------
    
    configs = SUAVE.Components.Configs.Config.Container()
    
    base_config = SUAVE.Components.Configs.Config(vehicle)
    base_config.tag = 'base'
    configs.append(base_config)
    
    # ------------------------------------------------------------------
    #   Cruise Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'cruise'
    
    configs.append(config)
    
    #note: takeoff and landing configurations taken from 737 - someone should update
    # ------------------------------------------------------------------
    #   Takeoff Configuration
    # ------------------------------------------------------------------
    
    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'takeoff'
    
    config.wings['main_wing'].flaps.angle = 20. * Units.deg
    config.wings['main_wing'].slats.angle = 25. * Units.deg
    
    config.V2_VS_ratio = 1.21
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    
    # ------------------------------------------------------------------
    #   Landing Configuration
    # ------------------------------------------------------------------

    config = SUAVE.Components.Configs.Config(base_config)
    config.tag = 'landing'
    
    config.wings['main_wing'].flaps_angle = 30. * Units.deg
    config.wings['main_wing'].slats_angle = 25. * Units.deg

    config.Vref_VS_ratio = 1.23
    config.maximum_lift_coefficient = 2.
    
    configs.append(config)
    
    
    # done!
    return configs