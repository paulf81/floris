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


import os
import copy
import time
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import floris.tools as wfct
import floris.tools.visualization as vis
from floris.tools.optimization.scipy.yaw import YawOptimization
from floris.tools.optimization.other.serial_refine import serial_refine


# Make a function to define a random wind farm
def define_random(fi_base, num_turbines):

    print("### Generating layout")

    # Determine x and y ranges
    x_range_D = spacing_D * num_turbines
    y_range_D = spacing_D * (num_turbines - 1)

    n_points = num_turbines * num_turbines
    x, y = np.zeros(n_points), np.zeros(n_points)
    x[0], y[0] = (
        np.round(random.uniform(0, x_range_D * D)),
        np.round(random.uniform(0, y_range_D * D)),
    )
    min_distances = []
    i = 1
    while i < n_points:
        x_temp, y_temp = (
            np.round(random.uniform(0, x_range_D * D)),
            np.round(random.uniform(0, y_range_D * D)),
        )
        distances = []
        for j in range(0, i):
            distances.append(np.sqrt((x_temp - x[j]) ** 2 + (y_temp - y[j]) ** 2))
        min_distance = np.min(distances)
        if min_distance > min_spacing_D * D:
            min_distances.append(min_distance)
            x[i] = x_temp
            y[i] = y_temp
            i = i + 1

    fi_base.reinitialize_flow_field(layout_array=(x, y))
    fi_base.calculate_wake()
    power_base = fi_base.get_farm_power()
    show_floris(fi_base)

    print("### Generating layout -> DONE!")
    return (fi_base, power_base)


# Define a function for appending results
def append_results(
    fi,
    df,
    yaw_angles,
    num_turbines,
    offset_or_rep,
    num_yaw,
    runtime,
    power_scipy,
    runtime_scipy,
    yaw_angles_scipy,
):
    fi.calculate_wake(yaw_angles=yaw_angles)
    power = fi.get_farm_power()
    print(yaw_angles, yaw_angles_scipy),
    df_temp = pd.DataFrame(
        {
            "num_turbines": [num_turbines],
            "offset_or_rep": [offset_or_rep],
            "num_yaw": [num_yaw],
            "power_base": [power_base],
            "power": [power],
            "power_scipy": [power_scipy],
            "scipy_gain": [100 * (power_scipy - power_base) / power_base],
            "gain": [100 * (power - power_base) / power_base],
            "runtime": [runtime],
            "pow_v_scipy": [
                100 * (power - power_scipy) / power_scipy
            ],  # [100 * power/power_scipy],
            "runtime_v_scipy": [
                runtime / runtime_scipy
            ],  # [100 * runtime/runtime_scipy],
            "yaw_angles": [(yaw_angles)],
            "yaw_angles_scipy": [(yaw_angles_scipy)],
            "largest_yaw_difference": max(
                yaw_angles - yaw_angles_scipy, key=abs
            ),  # * np.sign(yaw_angles-yaw_angles_scipy)
            "largest_abs_yaw_difference": np.max(np.abs(yaw_angles - yaw_angles_scipy)),
        }
    )
    return df.append(df_temp)


# Define a function to plot floris
def show_floris(fi, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    hor_plane = fi.get_hor_plane()

    wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
    wfct.visualization.plot_turbines_with_fi(ax, fi)


# Set up FLORIS interface
fi_base = wfct.floris_interface.FlorisInterface("../../example_input.json")


# Grab D
D = fi_base.floris.farm.turbines[0].rotor_diameter

# Spacing info
spacing_D = 6
min_spacing_D = 4

# Opt constraints
min_yaw = -30.0
max_yaw = 30.0
allow_negative_yaw = True

num_turbines_to_sweep = num_turbines_to_sweep = [2, 3]  # Note these are squared
num_iter = 2
num_yaw_to_sweep = [3, 5, 7]  # [3,5,7]


rep_sweep = list(range(num_iter))

# Test the optimizations
df_result = pd.DataFrame()

for num_turbines in num_turbines_to_sweep:

    for rep in rep_sweep:

        # Update fi_base
        fi_base, power_base = define_random(fi_base, num_turbines)

        # Test the normal procedure==============================
        fi_opt = copy.deepcopy(fi_base)
        tic = time.perf_counter()

        # Instantiate the Optimization object
        yaw_opt = YawOptimization(
            fi_opt, minimum_yaw_angle=min_yaw, maximum_yaw_angle=max_yaw
        )

        # Perform optimization
        yaw_angles_scipy = yaw_opt.optimize()

        # Grab the run time
        runtime_scipy = time.perf_counter() - tic

        # Get the optimization from scipy
        fi_opt.calculate_wake(yaw_angles=yaw_angles_scipy)
        power_scipy = fi_opt.get_farm_power()

        # Test the new methods==============================

        for num_yaw in num_yaw_to_sweep:

            fi_opt = copy.deepcopy(fi_base)
            tic = time.perf_counter()

            yaw_angles_opt = serial_refine(
                fi_opt,
                max_yaw,
                num_yaw=num_yaw,
                allow_negative_yaw=allow_negative_yaw,
                yaw_init=None,
            )

            # Grab the run time
            runtime = time.perf_counter() - tic

            # Save the result
            df_result = append_results(
                fi_opt,
                df_result,
                yaw_angles_opt,
                num_turbines,
                rep,
                num_yaw,
                runtime,
                power_scipy,
                runtime_scipy,
                yaw_angles_scipy,
            )

# To be accurate now square the number of turbines
df_result["num_turbines"] = df_result.num_turbines * df_result.num_turbines

df_melt = pd.melt(
    df_result[
        [
            "num_turbines",
            "num_yaw",
            "pow_v_scipy",
            "runtime_v_scipy",
            "largest_abs_yaw_difference",
        ]
    ],
    id_vars=["num_turbines", "num_yaw"],
)

# Change num_yaw to Better name
df_melt["Number Yaw Angles"] = df_melt["num_yaw"]

g = sns.catplot(
    data=df_melt,
    kind="box",
    x="num_turbines",
    y="value",
    col="variable",
    hue="Number Yaw Angles",
    sharey=False,
    palette="viridis_r",
    aspect=1.75,
    height=2.25,
)

# Style the plot
axarr = g.axes.flatten()

# First plot
ax = axarr[0]
ax.set_title("Optimal Power Versus Default SLSQP")
ax.set_ylabel("Percent Improvement (%)")
ax.set_xlabel("Number of turbines")
ax.axhline(0, color="k", ls="--")
ax.grid(True)
# ax.text(0.1,0,'Parity with scipy')

# Second plot
ax = axarr[1]
ax.set_title("Time (Serial_Refine) / Time (Default SLSQP)")
ax.set_ylabel("(-)")
ax.set_xlabel("Number of turbines")
ax.axhline(1.0, color="k", ls="--")
ax.grid(True)
# ax.text(0.1,0,'Parity with scipy')

# Third plot
ax = axarr[2]
ax.set_title("Largest absolute difference in yaw angle")
ax.set_ylabel("Deg")
ax.set_xlabel("Number of turbines")
ax.axhline(0.0, color="k", ls="--")
ax.grid(True)

plt.show()
