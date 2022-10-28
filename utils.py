import argparse, os
import hashlib

def command(
    args: argparse.Namespace, identifier: str, n: int,
    /,
    *,
    n_tabs: int,
):
    R"""
    Command lines.
    """
    #
    dataset = "--dataset {:s}".format(str(args.dataset))
    node_embedding = "--node_embedding {:s}".format(str(args.node_embedding))
    eval_method = "--eval_method {:s}".format(str(args.eval_method))
    device = "--device {:s}".format(str(args.device))
    hidden_channels = "--hidden_channels {:s}".format(str(args.hidden_channels))
    num_layers = "--num_layers {:s}".format(str(args.num_layers))
    dropout = "--dropout {:s}".format(str(args.dropout))
    lr = "--lr {:s}".format(str(args.lr))
    batch_size = "--batch_size {:s}".format(str(args.batch_size))
    patience = "--patience {:s}".format(str(args.patience))
    factor = "--factor {:s}".format(str(args.factor))
    epochs = "--epochs {:s}".format(str(args.epochs))
    runs = "--runs {:s}".format(str(args.runs))
    positional = "--positional" if args.positional else ""
    test_type = "--test_distribution {:s}".format(str(args.test_distribution))
    test_dataset = "--test_dataset {:s}".format(str(args.test_dataset))
    subsample_method = "--subsample_method {:s}".format(str(args.test_dataset))
    # identify = "--identify" if args.identify else ""

    cmdargs = (
        [
            dataset, node_embedding, eval_method, device,
            hidden_channels, num_layers, dropout, lr,
            batch_size, patience, epochs, runs,
            positional, test_type, test_dataset, subsample_method, identify
        ]
    )

    #
    if len(args.queue) > 0:
        submit = args.queue

    #
    heads = ["#!/bin/bash"]
    if len(args.queue) > 0:
        #
        heads.append("#SBATCH -A {:s}".format(submit))
    heads.append("#SBATCH --job-name={:s}".format(identifier))
    heads.append(
        "#SBATCH --output={:s}"
        .format(os.path.join("sbatch", "{:s}.stdout.txt".format(identifier))),
    )
    heads.append(
        "#SBATCH --error={:s}"
        .format(os.path.join("sbatch", "{:s}.stderr.txt".format(identifier))),
    )
    heads.append("#SBATCH --cpus-per-task=16")
    heads.append("#SBATCH --time=1-00:00:00")
    heads.append(
        "#SBATCH --gres=gpu:{:d}".format(1 if args.device == "cuda" else 0),
    )
    #
    load_lines =(
        ["module purge",
        "module load anaconda/2020.11-py38",
        "module load use.own",
        "module load conda-env/linkpred2-py3.8.5",
        'echo -e "[linkpred2] \033[92mActivated\033[0m"'
        ]
    )
    load_lines = "\n".join(load_lines).split("\n")

    #
    lines = (
        [
            "/usr/bin/time -f \"Max CPU Memory: %M KB\nElapsed: %e sec\"",
            "python -u main.py",
        ]
    )
    for (i, argument) in enumerate(cmdargs):
        #
        if len(argument) == 0:
            #
            continue
        n_requires = (
            len(lines[-1]) + 1 + len(argument)
            + (2 if i < len(cmdargs) - 1 else 0)
        )
        if n_requires > n:
            #
            lines.append(" " * n_tabs + argument)
        else:
            #
            lines[-1] = "{:s} {:s}".format(lines[-1], argument)
    lines = " \\\n".join(lines).split("\n")
    return heads + load_lines + lines

def get_identifier(args):
    print('--------------------------------------------------------')
    print(f'dataset = {args.dataset}')
    print(f'node_embedding = {args.node_embedding}')
    print(f'eval_method = {args.eval_method}')
    print(f'device = {args.device}')
    print(f'hidden_channels = {args.hidden_channels}')
    print(f'num_layers = {args.num_layers}')
    print(f'dropout = {args.dropout}')
    print(f'learning rate = {args.lr}')
    print(f'batch_size = {args.batch_size}')
    print(f'patience = {args.patience}')
    print(f'factor = {args.factor}')
    print(f'epochs = {args.epochs}')
    print(f'runs = {args.runs}')
    print(f'positional = {args.positional}')
    print(f'test_distribution = {args.test_distribution}')
    print(f'test_dataset = {args.test_dataset}')
    print(f'subsample_method = {args.subsample_method}')
    print(f'sample_node_pct = {args.sampling_percent}')
    # print(f'identify = {args.identify}')
    print('--------------------------------------------------------')
    file_title = '{}_{}_{}'.format(args.dataset, args.node_embedding, args.test_distribution)
    if args.positional:
        identifier = '{}_pos_hc{}_nl{}_lr{}_bs{}_epo{}_run{}'.format(args.subsample_method,
                                                                args.hidden_channels,
                                                                args.num_layers,
                                                                args.lr,
                                                                args.batch_size,
                                                                args.epochs,
                                                                args.runs,
                                                                )
    else:
        identifier = '{}_struc_hc{}_nl{}_lr{}_bs{}_epo{}_run{}'.format(args.subsample_method,
                                                                args.hidden_channels,
                                                                args.num_layers,
                                                                args.lr,
                                                                args.batch_size,
                                                                args.epochs,
                                                                args.runs,
                                                                )
    # identifier = (
    #     hashlib.md5(
    #         str(
    #             (
    #                 args.dataset, args.node_embedding, args.eval_method,
    #                 args.device, args.hidden_channels, args.num_layers,
    #                 args.dropout, args.lr, args.batch_size, args.patience,
    #                 args.factor, args.epochs, args.runs, args.positional,
    #                 args.test_distribution, args.test_dataset, args.identify
    #             ),
    #         ).encode(),
    #     ).hexdigest()
    # )
    # print(
        # "\x1b[103;30mDescription Hash\x1b[0m: \x1b[102;30m{:s}\x1b[0m"
        # .format(identifier),
    # )
    return file_title, identifier

def get_identifier_PEG(args):
    print('--------------------------------------------------------')
    print(f'PE_method = {args.PE_method}')
    print(f'PE_dim = {args.PE_dim}')
    print(f'log_steps = {args.log_steps}')
    print(f'use_sage = {args.use_sage}')
    print(f'num_layers = {args.num_layers}')
    print(f'hidden_channels = {args.hidden_channels}')
    print(f'dropout = {args.dropout}')
    print(f'batch_size = {args.batch_size}')
    print(f'lr = {args.lr}')
    print(f'epochs = {args.epochs}')
    print(f'eval_steps = {args.eval_steps}')
    print(f'runs = {args.runs}')

    print(f'dataset = {args.dataset}')
    print(f'test_distribution = {args.test_distribution}')
    print(f'test_dataset = {args.test_dataset}')
    print(f'subsample_method = {args.subsample_method}')
    print(f'sample_node_pct = {args.sampling_percent}')
    print('--------------------------------------------------------')
    return None
