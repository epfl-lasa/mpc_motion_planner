# mpc_motion_planner

A joint space motion planner based on Model Predictive Control (MPC) to find the minimum time trajectory between the current state and target state (position and velocity), while respecting box constraints on joint position, velocity, acceleration and torque.

We use [polyMPC](https://gitlab.epfl.ch/listov/polympc) as the MPC library and [Ruckig](https://github.com/pantor/ruckig) as an initial guess for the solver.

## Installation

### Requires:

- [pinocchio](https://github.com/stack-of-tasks/pinocchio)

- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (Version > 3.3)

### Installation steps

Clone the repository with the submodules, then compile the examples:

```
git clone --recurse-submodules https://github.com/AlbericDeLajarte/mpc_motion_planner.git 

cd mpc_motion_planner && mkdir build && cd build
cmake ..
make
```

## Testing the examples

### Single trajectory

Once compiled, you can launch the examples to test if everything works properly.

```
build/offline_trajectory
```
This generates one trajectory starting from the middle joint position and zero velocity, and going to a random position and velocity. The trajectory generated by Ruckig (as initial guess) and the MPC are then stored in the file `analysis/optimal_solution.txt` and can be plotted using the notebook `analysis/data_analysis.ipynb`

### Batch of trajectory

To benchmark the MPC motion planner against Ruckig, you can launch:

```
build/mpc_benchmark
```

which will generate a 1000 trajectories and record limit violations and planning statistics in `analysis/benchmark_data.txt`. You can use the notebook `analysis/benchmark_analysis.ipynb` to analyse the data.