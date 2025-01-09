import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # three necessary component of sbatch script
    parser.add_argument("--sbatch_script", type=str, default=None)
    parser.add_argument("--logger", type=str, default=None)
    parser.add_argument("--command", type=str, default=None)

    # optional info
    parser.add_argument("--job_name", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--gres", type=str, default=None)
    parser.add_argument("--time", type=str, default=None)
    parser.add_argument("--mem_alloc", type=str, default=None)

    args = parser.parse_args()

    assert args.sbatch_script and args.logger and args.command

    #!/bin/sh
    # SBATCH --job-name=train_detm
    # SBATCH -A tlippin1_gpu
    # SBATCH --partition=a100
    # SBATCH --gres=gpu:1
    # SBATCH --time=24:00:00
    # SBATCH --output=slurm_winsize_100.out
    # SBATCH --mem=180G

    with open(args.sbatch_script, "w") as f:
        f.write("#!/bin/sh\n")
        if args.job_name:
            f.write(f"#SBATCH --job-name={args.job_name}\n")
        if args.account:
            f.write(f"#SBATCH -A {args.account}\n")
        if args.time:
            f.write(f"#SBATCH --time={args.time}\n")
        if args.mem_alloc:
            f.write(f"#SBATCH --mem={args.mem_alloc}\n")
        f.write(f"#SBATCH --output={args.logger}\n")

        if args.use_gpu is True:
            if args.partition:
                f.write(f"#SBATCH --partition={args.partition}\n")
            if args.gres:
                f.write(f"#SBATCH --gres={args.gres}\n")

        f.write("\n")
        f.write(args.command)
