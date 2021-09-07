'''
Install Blender dependencies.
Meant to be run ONCE via blender as follows
`blender --background --python install.py`

NOTE: Blender has to be added to the system path or exported as 
environment variable. (ie. add C:\Program Files\Blender Foundation\Blender 2.93 to PATH)
>> set PATH=c:\Program Files\Blender Foundation\Blender 2.93;%PATH%
AND Blender has to be started with ADMINISTRATOR rights!

>> blender --background --python blender/install.py
'''

import bpy
import sys
import subprocess
from pathlib import Path

THISDIR = Path(__file__).parent


def run(cmd):
    try:
        output = subprocess.check_output(cmd)
        print(output)
    except subprocess.CalledProcessError as e:
        print(e.output)
        sys.exit(1)


def install(name, upgrade=True, user=True, editable=False):
    cmd = [sys.executable, '-m', 'pip', 'install']
    if upgrade:
        cmd.append('--upgrade')
    if user:
        cmd.append('--user')
    if editable:
        cmd.append('-e')
    cmd.append(name)
    run(cmd)


def bootstrap(user=True):
    cmd = [sys.executable, '-m', 'ensurepip', '--upgrade']
    if user:
        cmd.append('--user')
    run(cmd)
    cmd = [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip']
    if user:
        cmd.append('--user')
    run(cmd)


def main():
    print('Installing Blender dependencies. This might take a while...')
    bootstrap(user=True)
    install('pyzmq', user=True)
    
    
main()

import zmq
