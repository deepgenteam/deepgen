import multiprocessing as mp
import argparse
import os
from tqdm import tqdm
from glob import glob
import subprocess


def single_process(folder_list):
    for folder in tqdm(folder_list):
        subprocess.run(["tar", "-cf", f"{folder}.tar", folder])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=-1, type=int)
    parser.add_argument('--num-processes', default=8, type=int)

    args = parser.parse_args()

    folders = [path for path in sorted(glob('*')) if os.path.isdir(path)][args.start:args.end]

    num_folders = len(folders)
    num_processes = args.num_processes
    num_folders_per_process = num_folders // num_processes
    res = num_folders % num_processes
    if res > 0:
        num_processes += 1

    processes = [mp.Process(target=single_process,
                            args=(folders[process_id * num_folders_per_process:
                                            (process_id + 1) * num_folders_per_process]
                                  if process_id < num_processes - 1
                                  else folders[process_id * num_folders_per_process:],
                                  )
                            )
                 for process_id in range(num_processes)]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()
