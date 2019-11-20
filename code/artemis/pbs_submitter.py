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
        "--experiment-id",
        "--depth",
        "--semi-loss-weight",
        "--dataset",
        "--data-dir",
        #"",
    ]
    param_values = [
     [1],
     [200],
     [10.0],
     ["/project/RDS-FEI-NSEG-RW/tensorflow_checkpoints/ssl/"],
     [2000],
     [2e-4],
     ["t2"],#["rCBF"],["t2"],
     ["t1","flair","t1ce"],#["MTT","rCBV","TTP","Tmax"],#["t1","flair","t1ce"]
     ["semi"],
     [30,31,32,33,34],
     [3],
     [2.0],
     ["brats"],#["isles"],
     ["/project/RDS-FEI-NSEG-RW/semi_supervised_cyclegan/data/brats2018/"],#["/project/RDS-FEI-NSEG-RW/semi_supervised_cyclegan/data/isles/"]
     #["--only-pair"],
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
