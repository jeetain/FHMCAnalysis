"""@docstring
@brief Install script for building all modules in package
@author Nathan A. Mahynski
@date 03/11/2016
@filename install.py
"""

import os
import subprocess as sp

def loop_dirs(head):
        tree = []
        for root, dirs, files in os.walk(head, topdown=False):
                for name in dirs:
                        tree.append(os.path.join(root, name))
        return tree

if __name__ == "__main__":
        tree = loop_dirs("./")
        for t in tree:
                files = os.listdir(t)
                cwd = os.getcwd()
                if ('setup.py' in files):
                        os.chdir(t)
                        module = t.split('/')[-1]
                        print '\n\n*** Building '+module+' module ***\n\n'
                        sp.check_call(['python', 'setup.py', 'build_ext', '-i'])
                        os.chdir(cwd)
