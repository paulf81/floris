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

# Serial Refine method for yaw optimization
# Adaptation of Boolean Yaw Optimization by PJ Stanley
# Update with reference


import numpy as np


def serial_refine_single_pass(
    fi, max_yaw, first_pass, yaw_init=None, num_yaw=5, allow_negative_yaw=True
):

    # Confirm number yaw is an odd integer
    if (not isinstance(num_yaw, int)) or (num_yaw % 2 == 0):
        raise ValueError("num_yaw must be an odd integer")

    # Get a list of the turbines in order of x and sort front to back
    layout_x = fi.layout_x
    ind = np.argsort(layout_x)

    # Determine min_yaw
    if allow_negative_yaw:
        min_yaw = -1 * max_yaw
    else:
        min_yaw = 0

    # Initialize the list of yaw angles to try
    # These are always num_yaw long
    # And they represent change from the initial condition
    # 0 (no change) must always be in the list

    if first_pass:
        yaw_test_list = np.linspace(min_yaw, max_yaw, num_yaw)

    else:
        point_span = (max_yaw - min_yaw) / (num_yaw - 1)
        yaw_test_list = np.linspace(point_span / -2, point_span / 2, num_yaw)

    # If yaw init is not passed in initialize
    if yaw_init is None:
        if not first_pass:
            raise ValueError("***PASS IN YAW ANGLES ON SECOND PASS DUDE!")
        yaw_angles = np.zeros_like(layout_x)
    else:
        yaw_angles = yaw_init

    # Loop through turbines
    # result_yaw = np.zeros_like(yaw_test_list)

    # Find the starting power
    fi.calculate_wake(yaw_angles=yaw_angles)
    best_power = fi.get_farm_power()

    for t_idx in ind:

        init = yaw_angles[t_idx]

        best_yaw = init  # starting assumption
        for yaw_idx, yaw_delta in enumerate(yaw_test_list):

            yaw = init + yaw_delta

            # If yaw_delta is 0, don't need to check
            if yaw_delta == 0:
                continue

            # if yaw outside range, exclude from consideration
            elif (yaw < min_yaw) or (yaw > max_yaw):
                continue

            else:
                yaw_angles[t_idx] = yaw
                fi.calculate_wake(yaw_angles=yaw_angles)
                test_power = fi.get_farm_power()
                if test_power > best_power:

                    # This is an improvement, save it
                    best_power = test_power
                    best_yaw = yaw

        # Save the best yaw before moving forward
        yaw_angles[t_idx] = best_yaw

    return yaw_angles


def serial_refine(fi, max_yaw, num_yaw=5, allow_negative_yaw=True, yaw_init=None):

    # First pass
    yaw_angles = serial_refine_single_pass(
        fi,
        max_yaw=max_yaw,
        first_pass=True,
        yaw_init=yaw_init,
        num_yaw=num_yaw,
        allow_negative_yaw=allow_negative_yaw,
    )

    # Second pass
    yaw_angles = serial_refine_single_pass(
        fi,
        max_yaw=max_yaw,
        first_pass=False,
        yaw_init=yaw_angles,
        num_yaw=num_yaw,
        allow_negative_yaw=allow_negative_yaw,
    )

    return yaw_angles
