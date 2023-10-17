import os
import argparse
import textwrap


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent("""
    
    Wrapper to run dmercator inside Docker container

    0. (optional) build container or fetch already built image from the web
    1. Run embeddings with attached volume (local folder with the edgelist)

    
    NOTE: by default docker will run in background `-d` flag
      You can view current jobs by typing: `docker container ls`
      To kill the job run: `docker container kill <container_id>` 


    For more detailed information about parameters see README.md 
      or include/embeddingsSD_unix.hpp help string


    Example:
    > python run_dmercator_docker.py \
        -i path/to/edgelist.edge \
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

    parser.add_argument('-x', '--build_docker', type=str, default='',
                        required=False, help="Whether to build docker image")
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

    
    if args.build_docker:
        build_command = 'docker build -t dmercator-custom .'
        os.system(build_command)

    # TODO: publish docker image
    image_name = 'dmercator-ub' if not args.build_docker else 'dmercator-custom'
    
    run_command = f"""
      docker run -d --rm -v {folder}:/data {image_name} \
        -d {args.dimension} \
        {args.screen_mode} \
        {args.beta} \
        {args.clean_mode} \
        {args.fast_mode} \
        {args.no_kappa_postprocessing} \
        {args.validation_mode} \
        /data/{filename}
    """
    os.system(run_command)

