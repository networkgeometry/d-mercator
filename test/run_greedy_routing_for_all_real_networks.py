import os
import glob
import sys


if __name__ == '__main__':
    folder = sys.argv[1]
    for edgelist in glob.glob(f"{folder}/*.edge"):
        folder_path = "".join(edgelist.split(".")[0])
        command = f"python test/greedy_routing_real_networks.py -i {edgelist} -d2 1,2,3,4,5 -o {folder_path}"
        print(command)
        os.system(command)