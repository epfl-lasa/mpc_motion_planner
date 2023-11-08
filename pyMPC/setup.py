import setuptools

# pip install .
# python3 setup.py sdist
# twine upload -r testpypi dist/* --verbose

# To install on your machine : pip install --upgrade dist/your_package_name-0.1.tar.gz

setuptools.setup(
    name="pyMPC",
    version="0.0.21",
    author="Stephen Monnet",
    description="MPC library to control 7DOFs arm robots",
    packages=setuptools.find_packages(),
    #data_files=[('.', ['motion_planning_lib.so'])]
    #package_dir={"ArmRobotMotionPlanner": 'ArmRobotMotionPlanner'},
    #package_data={'ArmRobotMotionPlanner': ['*.so']}, 
)