import numpy as np
import os
import argparse
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""    
    Example:

    > python test/generate_synthetic_networks.py \
        -o [output_folder]
        -b 1.5,2.5
        -d 1,2,3,4,5,6,7,8,9,10
        -g 2.1,2.7,3.5
        -s 1000
        -n 10
    """))
    parser.add_argument('-o', '--output_folder', type=str, required=True, help="Path to output folder")
    parser.add_argument('-b', '--beta', type=lambda x: [float(y) for y in x.split(',')],
                        required=True, help="Values of beta")
    parser.add_argument('-g', '--gamma', type=lambda x: [float(y) for y in x.split(',')],
                        required=True, help="Values of gamma")
    parser.add_argument('-d', '--dim', type=lambda x: [int(y) for y in x.split(',')],
                        required=True, help="Values of dimension")
    parser.add_argument('-s', '--size', type=int, required=False, 
                        default=1000, help="Size of the networks")
    parser.add_argument('-n', '--ntimes', type=int, required=False, 
                        default=10, help="Number of realizations")
    args = parser.parse_args()
    return args


def generate_kappas(n, gamma, mean_degree=10):
    kappa_0 = (
        (1 - 1 / n)
        / (1 - n ** ((2 - gamma) / (gamma - 1)))
        * (gamma - 2)
        / (gamma - 1)
        * mean_degree
    )
    kappa_c = kappa_0 * n ** (1 / (gamma - 1))

    kappas = []
    for _ in range(n):
        kappas.append(
            kappa_0
            * (1 - np.random.uniform(0, 1) * (1 - (kappa_c / kappa_0) ** (1 - gamma)))
            ** (1 / (1 - gamma))
        )
    return kappas


def generate_synthetic_network(folder, size, beta, gamma, dim, i):
    kappas = generate_kappas(size, gamma)
    path = f'{folder}/kappas_i{i}.txt'
    with open(path, 'w') as f:
        for k in kappas:
            f.write(f'{k}\n')
    
    command = f"./gen_net -b {beta*dim} -d {dim} -v {path}"
    os.system(command)


if __name__ == '__main__':
    args = parse_args()
    os.system('g++ -O3 --std=c++17 -o gen_net src/generatingSD_unix.cpp')

    for b in args.beta:
        for g in args.gamma:
            for d in args.dim:
                for i in range(args.ntimes):
                    folder = f'{args.output_folder}/beta_{str(b).replace(".", "_")}/gamma_{str(g).replace(".", "_")}/dim{d}/i{i}'
                    os.makedirs(folder)
                    generate_synthetic_network(folder, args.size, b, g, d, i)
    