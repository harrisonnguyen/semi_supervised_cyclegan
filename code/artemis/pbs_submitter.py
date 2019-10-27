#!/usr/bin/env python
import subprocess
import pandas as pd
import itertools

def main():
    file_name = "code/main.py"
    artemis_file = "artemis_experiments.pbs"
    param_names = [
        "--batch-size",
        "--n-epochs",
        "--cycle-loss-weight",
        "--checkpoint-dir",
        "--summary-freq",
        "--learning-rate",
        "--mod-a",
        "--mod-b",
        "--model",
        "--patch-size",
        "--experiment-id",
        "--depth",
        "--semi-loss-weight"
    ]
    param_values = [
     [2],
     [200],
     [2.0],
     ["/project/RDS-FEI-NSEG-RW/tensorflow_checkpoints/ssl/"],
     [2000],
     [2e-4],
     ["t2"],
     ["t1","flair","t1ce"],
     ["wasser"],
     [240],
     [0,1,2,3,4],
     [3],
     [1.0]
    ]

    for i, e in enumerate(itertools.product(*param_values)):
        params = ""
        for pn, pv in zip(param_names, e):
            if pv is not None:
                if isinstance(pv, str) and len(pn) == 0:
                    if pv:
                        params = params + " {}".format(pv)
                else:
                    params = params + " {}={}".format(pn, pv)
        params ="{} ".format(file_name) + params
        command = 'qsub -v line=\"{}\" {}'.format(params,artemis_file)

        #print(command)

        exit_status = subprocess.call(command, shell=True)
        if exit_status is 1:  # Check to make sure the job submitted
            print("Job {0} failed to submit".format(command))
        else:
            print("Job {0} success!".format(command))

if __name__ == "__main__":
    exit(main())  # pragma: no cover
