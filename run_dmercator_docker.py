import os
import argparse
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    
    Wrapper to run dmercator inside Docker container

    1. Create a folder and place a network you want to embed. For instance: `data/mydata.edge`
    2. Run embeddings with attached volume (local folder with the edgelist)
    3. The embeddings will be placed in `data/` folder
    
    NOTE: by default docker will run in background `-d` flag
      You can view current jobs by typing: `docker container ls`
      To kill the job run: `docker container kill <container_id>` 

    For more detailed information about parameters see README.md 
      or include/embeddingsSD_unix.hpp help string

    Example:
    > python run_dmercator_docker.py \
        -i data/mydata.edge \
        -d 2 \
        -v 1 \ # 1 - turn on that option
        -c 1
    """))
    parser.add_argument('-i', '--input', type=str,
                        required=True, help="Path to edgelist")
    parser.add_argument('-d', '--dimension', type=int,
                        required=True, help="Dimension of embeddings")
    parser.add_argument('-b', '--beta', type=str, default='',
                        required=False, help="Value of beta. By default the program infers the value of beta")
    parser.add_argument('-a', '--screen_mode', type=str, default='',
                        required=False, help="Logs on the screen")
    parser.add_argument('-c', '--clean_mode', type=str, default='',
                        required=False, help="Writes the inferred coordinates without headers")
    parser.add_argument('-f', '--fast_mode', type=str, default='',
                        required=False, help="Fast mode. Only use LE. Suitable only for S^1")
    parser.add_argument('-k', '--no_kappa_postprocessing', type=str, default='',
                        required=False, help="No kappas postprocessing after ML.")
    parser.add_argument('-v', '--validation_mode', type=str, default='',
                        required=False, help="Validates and characterizes the inferred random network ensamble")
    parser.add_argument('-s', '--seed', type=float, default=None,
                        required=False, help="Random seed")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    folder = os.path.dirname(args.input)
    filename = os.path.basename(args.input)

    # Add additional flags
    if args.beta != '':
        args.beta = f'-b {args.beta}'
    
    if args.screen_mode:
        args.screen_mode = '-a' 

    if args.clean_mode:
        args.clean_mode = '-c'
    
    if args.fast_mode:
        args.fast_mode = '-f'
    
    if args.no_kappa_postprocessing:
        args.no_kappa_postprocessing = '-k'
        
    if args.validation_mode:
        args.validation_mode = '-v'

    if args.seed is not None:
        seed = f'-s {args.seed}'
        env_var = "-e OMP_NUM_THREADS=1"
    else:
        seed = ''
        env_var = ''

    run_command = f"""
      docker run -d --rm {env_var} -v {os.path.abspath(folder)}:/data rjankowskiub/dmercator \
        -d {args.dimension} \
        {args.screen_mode} \
        {args.beta} \
        {args.clean_mode} \
        {args.fast_mode} \
        {args.no_kappa_postprocessing} \
        {args.validation_mode} \
        {seed} \
        /data/{filename}
    """
    os.system(run_command)

