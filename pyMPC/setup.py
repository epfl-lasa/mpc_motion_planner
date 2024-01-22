import shutil, site, os, pathlib

site_pkg = site.getsitepackages()[0]
mpc_motion_planner_path = str(pathlib.Path(__file__).parent.parent)

print("------------ \tMain site package found : " + site_pkg + "\t -------------")

try:
    shutil.rmtree(site_pkg + '/mpc_solver')
    print('------------ \tRemoved old mpc_solver\t -------------')
except:
    print('------------ \tNo old mpc_solver found\t -------------')

shutil.copytree(mpc_motion_planner_path + '/build/mpc_solver', site_pkg + '/mpc_solver')
shutil.copytree(mpc_motion_planner_path + '/pyMPC', site_pkg + '/mpc_solver/pyMPC')
shutil.copytree(mpc_motion_planner_path + '/descriptions', site_pkg + '/mpc_solver/descriptions')
print('------------ \tCopied new mpc_solver\t -------------')