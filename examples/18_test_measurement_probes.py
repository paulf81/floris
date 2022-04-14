# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import copy
import matplotlib.pyplot as plt
import numpy as np

from floris.tools import FlorisInterface


# Instantiate FLORIS using either the GCH or CC model
for model in ["gch", "cc", "jensen", "turbopark"]:
    print("Processing things for '{:s}'...".format(model))

    # Load FLORIS
    fi = FlorisInterface("inputs/{:s}.yaml".format(model))

    # Get the NREL 5MW turbine
    nrel5mw_definition = fi.floris.farm.turbine_definitions[0]

    # Define turbine locations
    turbine_locations = [[0, 700.0, 0.0, 700.0], [0.0, 0.0, 500.0, 500.0]]
    nturbs = len(turbine_locations[0])

    # Define a mesh of probes
    x = turbine_locations[0]
    y = turbine_locations[1]
    Xp, Yp = np.meshgrid(
        np.linspace(np.min(x) - 200.0, np.max(x) + 500.0, 40),
        np.linspace(np.min(y) - 200.0, np.max(y) + 200.0, 20),
    )
    probe_definition = copy.deepcopy(nrel5mw_definition)
    probe_definition["turbine_type"] = "probe"
    probe_definition["rotor_diameter"] = 1.0e-12
    probe_definition["power_thrust_table"] = {
        "wind_speed": [0.0, 100.0],
        "power": [0.0, 0.0],
        "thrust": [0.0, 0.0],
    }
    probe_locations = [list(Xp.flatten()), list(Yp.flatten())]
    nprobes = len(probe_locations[0])

    # Initialize the <nturbs> turbine farm with <nprobes> probes
    fi.reinitialize(
        layout=[
            np.hstack([turbine_locations[0], probe_locations[0]]),
            np.hstack([turbine_locations[1], probe_locations[1]]),
        ],
        turbine_type=(
            np.hstack(
                [
                    np.repeat(nrel5mw_definition, nturbs),
                    np.repeat(probe_definition, nprobes)
                ]
            )
        ),
        wind_directions=[270.0],
        wind_speeds=[8.0]
    )

    fi.calculate_wake()  # Calculate solutions
    # df = fi.get_plane_of_points()  # Collect all measurements

    fig, ax = plt.subplots()
    plt.tricontourf(
        fi.floris.grid.x.flatten(),
        fi.floris.grid.y.flatten(),
        fi.floris.flow_field.u.flatten(),
        levels=np.linspace(0.0, 8.5, 31),
    )
    ax.set_title("Model: {:s}".format(model))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    plt.colorbar()
    plt.jet()

plt.show()
