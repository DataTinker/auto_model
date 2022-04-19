import os


basePath=input('Enter your desired start path')

from subprocess import check_output


for root, dirs, files in os.walk(basePath, topdown=False):
    for name in files:
        if  name.endswith('.py'):
            fPath = os.path.join(root, name)
            cmd = 'autopep8 -i --aggressive "{}"'.format(fPath)
            result = check_output(cmd, shell=True).decode()


