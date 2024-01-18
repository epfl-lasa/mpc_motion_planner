import shutil, site, os

site_pkg = site.getsitepackages()[0]

print("------------ \tMain site package found : " + site_pkg + "\t -------------")

try:
    shutil.rmtree(site_pkg + '/mpc_solver')
    print('------------ \tRemoved old mpc_solver\t -------------')
except:
    print('------------ \tNo old mpc_solver found\t -------------')

shutil.copytree('../build/mpc_solver', site_pkg + '/mpc_solver')
print('------------ \tCopied new mpc_solver\t -------------')