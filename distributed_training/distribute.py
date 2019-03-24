import os
import json

def set_tf_config(task_type="ps"):
    with open('cluster_spec.json', 'r') as f:
        cluster_spec = json.load(f)
        os.environ["TF_CONFIG"] = json.dumps(cluster_spec[task_type])
