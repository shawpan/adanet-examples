"""
Parse cluster specification file and run training commands
given the configuration in cluster_spec.json
"""

import os
import json
import shutil
import config

with open('cluster_spec.json', 'r') as f:
    cluster_spec = json.load(f)
    for process, value in cluster_spec.items():
        os.system('python trainer.py --task_type=' + process  + ' &')
