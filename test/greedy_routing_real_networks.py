import os
import argparse
import random
import subprocess
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    Example:

    Embedd real networks using SD model. Then run greedy routing.

    > python test/greedy_rounting_real_networks.py \
        -i /path/to/edgelist/
        -d2 1,2,3,4,5 \
        -o [output_folder]
    """))
    parser.add_argument('-i', '--input', type=str, required=True, help="Path to edgelist")
    parser.add_argument('-d2', '--dim_embedded', type=str,
                        required=True, help="Dimensions of embedded synthetic networks")
    parser.add_argument('-o', '--output_folder', type=str,
                        required=True, help="Path to output folder with all results")
    parser.add_argument('-q', '--n_runs', type=int,
                        required=False, default=100000, help="Number of greedy routing iterations.")
    args = parser.parse_args()
    return args


def embed_network(dim, edgelist_path) -> int:
    seed = random.randint(0, 999999)
    status = subprocess.Popen(
        f"./mercator -s {seed} -d {dim} -c -v {edgelist_path}; echo $?", shell=True, stdout=subprocess.PIPE)
    exit_code = status.stdout.read().decode()
    return int(exit_code)


def rerun_embed_network(dim, initial_edgelist_path) -> str:
    dirname = os.path.dirname(initial_edgelist_path)
    base_filename = os.path.split(initial_edgelist_path)[-1].split(".")[0]
    new_edgelist_path = f"{dirname}/{base_filename}_GC.edge"
    seed = random.randint(0, 999999)
    run_command = f"./mercator -s {seed} -d {dim} -c -v {new_edgelist_path}"
    os.system(run_command)
    return new_edgelist_path


def run_embedding(output_folder, edgelist_path, dim):
    tmp_directory = f"{output_folder}/eS{dim}/"
    os.makedirs(tmp_directory, exist_ok=True)
    cp_command = f"cp {edgelist_path} {tmp_directory}"
    os.system(cp_command)

    # Embed network
    tmp_edgelist = f"{tmp_directory}/{os.path.split(edgelist_path)[-1]}"
    exit_code = embed_network(dim, tmp_edgelist)
    output_filename = tmp_edgelist
    if exit_code == 12:
        output_filename = rerun_embed_network(dim, tmp_edgelist)

    coords_path = os.path.split(output_filename)[-1].split(".")[0]
    coords_path = f"{tmp_directory}/{coords_path}.inf_coord"
    return output_filename, coords_path


def run_greedy_routing(dim, coords_path, edgelist_path, n_runs, modified_version=0, suffix="gr"):
    folder_path = os.path.dirname(coords_path)
    filename = os.path.split(coords_path)[-1]
    filename = filename.split(".")[0]
    results_filename = f"{folder_path}/{filename}.{suffix}"

    command = f"""
        g++ --std=c++17 -O3 lib/greedy_routing.cpp && ./a.out {dim} {coords_path} {edgelist_path} {modified_version} {n_runs} > {results_filename}
    """
    os.system(command)


if __name__ == '__main__':
    args = parse_args()

    dims = args.dim_embedded.split(",")
    os.makedirs(args.output_folder, exist_ok=True)
        
    for d in dims:
        # 1. Embed synthetic network
        new_edgelist_path, new_coords_path = run_embedding(args.output_folder, args.input, d)

        # 2. Run greedy routing (original and modified version) on the inferred positions
        run_greedy_routing(d, new_coords_path, new_edgelist_path, args.n_runs, suffix="ogr")
        run_greedy_routing(d, new_coords_path, new_edgelist_path, args.n_runs, modified_version=1, suffix="mgr")
        