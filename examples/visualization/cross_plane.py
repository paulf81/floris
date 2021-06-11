# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import matplotlib.pyplot as plt

import floris.tools as wfct


# Parameters
d_downstream = 5.0
D = 126.0
yaw = 0.0

# Initialize the FLORIS interface fi
fi = wfct.floris_interface.FlorisInterface("../example_input.json")

# Place single turbine at 0,0
fi.reinitialize_flow_field(layout_array=([0], [0]), wind_speed=8.0)

# Calculate wake
fi.calculate_wake(yaw_angles=[yaw])


# Get horizontal plane at default height (hub-height)
hor_plane = fi.get_cross_plane(x_loc=d_downstream * D)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
wfct.visualization.reverse_cut_plane_x_axis_in_plot(ax)
plt.show()
