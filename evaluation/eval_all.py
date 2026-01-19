import os, json, sys
from pathlib import Path


SBATCH_COMMAND = """#!/bin/bash

cd pixio/evaluation

"""

baselines = [
    ("<run_dir>", "<repo_path>:<module_path>:<model_func_name>", "none"),
]


tasks = ['knn', 'monodepth/nyuv2_linear', 'monodepth/nyuv2_dpt', 'monodepth/kitti_linear', 'monodepth/kitti_dpt', 'semseg/ade20k_linear', 'semseg/ade20k_dpt', 'semseg/pascal_linear', 'semseg/pascal_dpt', 'semseg/loveda_linear', 'semseg/loveda_dpt']


def launch_jobs(model, pretrained, outdir):
    missing_result = []
    for subdir in tasks:
        result_json = f"{outdir}/{subdir}/result.json"
        if not os.path.exists(result_json):
            if 'monodepth' in subdir:
                missing_result.append(f"sbatch launch_monodepth.sh monodepth/configs/{subdir.split('/')[-1]}.yaml" + " {model} {pretrained} {outdir}")
            elif 'semseg' in subdir:
                missing_result.append(f"sbatch launch_semseg.sh semseg/configs/{subdir.split('/')[-1]}.yaml" + " {model} {pretrained} {outdir}")
            elif 'knn' in subdir:
                missing_result.append(f"sbatch launch.sh {model} {pretrained} {outdir}")
                
    if len(missing_result) > 0:
        sbatch = (SBATCH_COMMAND + "\n".join(missing_result)).format(
            model=model,
            pretrained=pretrained,
            outdir=outdir,
        )
    
        print("Writing sbatch command ...")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        with open(f"{outdir}/submit.sh", "w") as f:
            f.write(sbatch)
        os.system(f"bash {outdir}/submit.sh")


def eval_main():
    for rundir, model, pretrained in baselines:
        outdir = rundir
        if len(sys.argv) > 1:
            launch_jobs(model, pretrained, outdir)
        
        results = []
        for subdir in tasks:
            result_json = f"{outdir}/{subdir}/result.json"
            if os.path.exists(result_json):
                with open(result_json) as f:
                    result = json.load(f)
                results.append(f"{result['key_metric']:.04}")
            else:
                results.append(f"n/a")
                
        print(rundir.split('/')[-1], ',\t', ', '.join(results) )


if __name__ == "__main__":
    eval_main()
