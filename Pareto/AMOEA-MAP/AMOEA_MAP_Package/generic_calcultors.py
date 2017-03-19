# ------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------

""" Generic calculators """

from HV_dimension_sweep import HyperVolume

# integer adjustment
def integer_adjustment(x_in):
    up_ = 1.0
    low_ = 0.0
    [a,b] = divmod(x_in,up_)
    if (up_ - b)<(b - low_):
        x_out = a + 1
    else:
        x_out = a
    return x_out

# variable bounding
def bounded(x, Up, Lo):
    if x > Up:
        x = Up
    elif x < Lo:
        x = Lo
    return x

# Hyper volume calculations by Dimension Sweep approach
def hyper_volume_Dsweep(P_,arg):
    referencePoint = arg['Reference point for HVI(DS)']
    hyperVolume = HyperVolume(referencePoint)
    hyper_volume_DS = hyperVolume.compute(P_)
    return hyper_volume_DS



