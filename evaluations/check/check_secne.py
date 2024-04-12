"""
Check the scene cut results
1. scene numbers = metadata numbers
"""


import os
import json
import sys
import subprocess

dir_path = sys.argv[1].rstrip('/')

with open(dir_path+'/metadata.json', 'r') as f:
    x = json.load(f)

print('metadata length', len(x))
cmd = f'ls -l {dir_path} | grep ^- | wc -l'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = int(result.stdout.strip())
print("file nums :", output)

if 1 <= output - len(x) <= 3:
    print('Successful. Zipping...')
    os.system(f'zip -r {os.path.join(os.path.dirname(dir_path), os.path.basename(dir_path)+".zip")} {dir_path}')
    os.system(f'rm -r {os.path.join(os.path.dirname(dir_path), "group_"+os.path.basename(dir_path).split("_")[-1])}')
else:
    print('Something is wrong. Check it out.')