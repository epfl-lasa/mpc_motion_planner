# mpc_motion_planner

A joint space motion planner based on Model Predictive Control (MPC) to find the minimum time trajectory between the current state and target state (position and velocity), while respecting box constraints on joint position, velocity, acceleration and torque, as well as height constraint to avoid the mounting surface of the robot.

We use [polyMPC](https://gitlab.epfl.ch/listov/polympc) as the MPC library and [Ruckig](https://github.com/pantor/ruckig) as an initial guess for the solver.

## Installation

### Requires:

- [pinocchio](https://github.com/stack-of-tasks/pinocchio)

- [eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) (**Version > 3.3**)

### Installation steps

Clone the repository with the submodules, then compile the examples:

```
git clone --recurse-submodules --branch bindings_ruckig_included https://github.com/epfl-lasa/mpc_motion_planner.git 

cd mpc_motion_planner && mkdir build && cd build
cmake ..
make
```

### Install python library

Create a virtual environment, for instance **myVenv** (`python3 -m venv myVenv`), active it (`source myVenv/bin/activate`) and run the mpc_motion_planner/pyMPC/setup.py file.

```
cd .. && python3 pyMPC/setup.py 
```

Then install the requirements : 

`pip install -r requirements.txt`

To use the library within your project, write at the top of it:

```
import sys
sys.path.append("/<path_to>/mpc_motion_planner")
from pyMPC.motion_planner import MotionPlanner, Trajectory, RobotModel
```

A [Jupyter Notebook](pyMPC/howToUse.ipynb) presents the main features of the library and how to use it.
