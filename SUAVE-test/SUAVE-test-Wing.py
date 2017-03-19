
import SUAVE
from SUAVE.Core import Units

# ------------------------------------------------------------------
#   Main Wing
# ------------------------------------------------------------------
vehicle = SUAVE.Vehicle()
vehicle.tag = 'Boeing_BWB_450'

wing = SUAVE.Components.Wings.Main_Wing()
wing.tag = 'main_wing'

wing.aspect_ratio            = 289.** 2 / (7840. * 2)
wing.thickness_to_chord = 0.15
wing.taper = 0.0138
wing.span_efficiency = 0.95

wing.spans.projected = 289.0 * Units.feet

wing.chords.root = 145.0 * Units.feet
wing.chords.tip = 3.5 * Units.feet
wing.chords.mean_aerodynamic = 86. * Units.feet

wing.areas.reference = 15680. * Units.feet ** 2
wing.sweeps.quarter_chord = 33. * Units.degrees

wing.twists.root = 0.0 * Units.degrees
wing.twists.tip = 0.0 * Units.degrees
wing.dihedral = 2.5 * Units.degrees

wing.origin = [0., 0., 0]
wing.aerodynamic_center = [0, 0, 0]

wing.vertical = False
wing.symmetric = True
wing.high_lift = True

wing.dynamic_pressure_ratio = 1.0

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_1'
segment.percent_span_location = 0.0
segment.twist = 0. * Units.deg
segment.root_chord_percent = 1.
segment.dihedral_outboard = 0. * Units.degrees
segment.sweeps.quarter_chord = 30.0 * Units.degrees
segment.thickness_to_chord = 0.165
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_2'
segment.percent_span_location = 0.052
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.921
segment.dihedral_outboard = 0. * Units.degrees
segment.sweeps.quarter_chord = 52.5 * Units.degrees
segment.thickness_to_chord = 0.167
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_3'
segment.percent_span_location = 0.138
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.76
segment.dihedral_outboard = 1.85 * Units.degrees
segment.sweeps.quarter_chord = 36.9 * Units.degrees
segment.thickness_to_chord = 0.171
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_4'
segment.percent_span_location = 0.221
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.624
segment.dihedral_outboard = 1.85 * Units.degrees
segment.sweeps.quarter_chord = 30.4 * Units.degrees
segment.thickness_to_chord = 0.175
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_5'
segment.percent_span_location = 0.457
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.313
segment.dihedral_outboard = 1.85 * Units.degrees
segment.sweeps.quarter_chord = 30.85 * Units.degrees
segment.thickness_to_chord = 0.118
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_6'
segment.percent_span_location = 0.568
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.197
segment.dihedral_outboard = 1.85 * Units.degrees
segment.sweeps.quarter_chord = 34.3 * Units.degrees
segment.thickness_to_chord = 0.10
wing.Segments.append(segment)

segment = SUAVE.Components.Wings.Segment()
segment.tag = 'section_7'
segment.percent_span_location = 0.97
segment.twist = 0. * Units.deg
segment.root_chord_percent = 0.086
segment.dihedral_outboard = 73. * Units.degrees
segment.sweeps.quarter_chord = 55. * Units.degrees
segment.thickness_to_chord = 0.10
wing.Segments.append(segment)

# add to vehicle
vehicle.append_component(wing)
