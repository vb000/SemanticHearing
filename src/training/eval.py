"""
Test script to evaluate the model.
"""

import argparse
import importlib
import multiprocessing
import os, glob
import logging
import json

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm  # pylint: disable=unused-import
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_distortion_ratio as si_sdr)

from src.helpers import utils

def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               n_items: int, loss_fn, metrics_fn,
               results_fn = None, results_path: str = None, output_dir: str = None,
               profiling: bool = False, epoch: int = 0,
               writer: SummaryWriter = None) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    metrics = {}
    losses = []
    runtimes = []
    results = []

    with torch.no_grad():
        for batch_idx, (inp, tgt) in \
                enumerate(tqdm(test_loader, desc='Test', ncols=100)):
            # Move data to device
            inp, tgt = test_loader.dataset.to(inp, tgt, device)

            # Run through the model
            if profiling:
                with profile(activities=[ProfilerActivity.CPU],
                            record_shapes=True) as prof:
                    with record_function("model_inference"):
                        output = model(inp, writer=writer, step=epoch, idx=batch_idx)
                if profiling:
                    logging.info(
                        prof.key_averages().table(sort_by="self_cpu_time_total",
                                                row_limit=20))
            else:
                output = model(inp, writer=writer, step=epoch, idx=batch_idx)

            # Compute loss
            loss = loss_fn(output, tgt)

            # Compute metrics
            metrics_batch = metrics_fn(inp, output, tgt)
            for k in metrics_batch.keys():
                if not k in metrics:
                    metrics[k] = metrics_batch[k]
                else:
                    metrics[k] += metrics_batch[k]

            output = test_loader.dataset.output_to(output, 'cpu')
            inp, tgt = test_loader.dataset.to(inp, tgt, 'cpu')

            # Results to save
            if results_path is not None:
                results.append(results_fn(
                    batch_idx * test_loader.batch_size,
                    inp, output, tgt, metrics_batch, output_dir=output_dir))

            losses += [loss.item()]
            if profiling:
                runtimes += [ # Runtime per sample in ms
                    prof.profiler.self_cpu_time_total / (test_loader.batch_size * 1e3)]
            else:
                runtimes += [0.0]

            output = test_loader.dataset.output_to(output, 'cpu')
            inp, tgt = test_loader.dataset.to(inp, tgt, 'cpu')
            if writer is not None:
                if batch_idx == 0:
                    test_loader.dataset.tensorboard_add_sample(
                        writer, tag='Test',
                        sample=(inp, output, tgt),
                        step=epoch)
                test_loader.dataset.tensorboard_add_metrics(
                    writer, tag='Test', metrics=metrics_batch, step=epoch)

            if n_items is not None and batch_idx == (n_items - 1):
                break

        if results_path is not None:
            torch.save(results, results_path)
            logging.info("Saved results to %s" % results_path)

        avg_metrics = {k: np.mean(metrics[k]) for k in metrics.keys()}
        avg_metrics['loss'] = np.mean(losses)
        avg_metrics['runtime'] = np.mean(runtimes)
        avg_metrics_str = "Test:"
        for m in avg_metrics.keys():
            avg_metrics_str += ' %s=%.04f' % (m, avg_metrics[m])
        logging.info(avg_metrics_str)

        return avg_metrics

def evaluate(network, args: argparse.Namespace):
    """
    Evaluate the model on a given dataset.
    """

    # Load dataset
    data_test = utils.import_attr(args.test_dataset)(**args.test_data_args)
    logging.info("Loaded test dataset %d elements" % len(data_test))

    # Set up the device and workers.
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if use_cuda:
        gpu_ids = args.gpu_ids if args.gpu_ids is not None\
                        else range(torch.cuda.device_count())
        device_ids = [_ for _ in gpu_ids]
        data_parallel = len(device_ids) > 1
        device = 'cuda:%d' % device_ids[0]
        torch.cuda.set_device(device_ids[0])
        logging.info("Using CUDA devices: %s" % str(device_ids))
    else:
        data_parallel = False
        device = torch.device('cpu')
        logging.info("Using device: CPU")

    # Set multiprocessing params
    num_workers = min(multiprocessing.cpu_count(), args.n_workers)
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': True
    } if use_cuda else {}

    # Set up data loader
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=args.eval_batch_size, collate_fn=data_test.collate_fn,
        **kwargs)

    # Set up model
    model = network.Net(**args.model_params)
    if use_cuda and data_parallel:
        model = nn.DataParallel(model, device_ids=device_ids)
        logging.info("Using data parallel model")
    model.to(device)

    # Load weights
    if args.pretrain_path == "best":
        ckpts = glob.glob(os.path.join(args.exp_dir, '*.pt'))
        ckpts.sort(
            key=lambda _: int(os.path.splitext(os.path.basename(_))[0]))
        val_metrics = torch.load(ckpts[-1])['val_metrics'][args.base_metric]
        best_epoch = max(range(len(val_metrics)), key=val_metrics.__getitem__)
        args.pretrain_path = os.path.join(args.exp_dir, '%d.pt' % best_epoch)
        logging.info(
            "Found 'best' validation %s=%.02f at %s" %
            (args.base_metric, val_metrics[best_epoch], args.pretrain_path))
    if args.pretrain_path != "":
        utils.load_checkpoint(
            args.pretrain_path, model, data_parallel=data_parallel)
        logging.info("Loaded pretrain weights from %s" % args.pretrain_path)

    # Results csv file
    results_fn = network.format_results
    results_path = os.path.join(args.exp_dir, 'results.eval.pth')
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate
    try:
        return test_epoch(
            model, device, test_loader, args.n_items, network.loss,
            network.test_metrics, results_fn, results_path, args.output_dir, args.profiling)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as _:  # pylint: disable=broad-except
        import traceback  # pylint: disable=import-outside-toplevel
        traceback.print_exc()

def get_unique_hparams(exps):
    """
    Return a list of unique hyperparameters across the set of experiments.
    """
    # Read config files into a dataframe
    configs = []
    for e in exps:
        with open(os.path.join(e, 'config.json')) as f:
            configs.append(pd.json_normalize(json.load(f)))
    configs = pd.concat(configs, ignore_index=True)

    # Remove constant colums from configs dataframe. None values are considered constant.
    configs = configs.loc[:, configs.nunique() > 1]

    return configs.to_dict('records')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Data Params
    parser.add_argument('experiments', nargs='+', type=str,
                        default=None,
                        help="List of experiments to evaluate. "
                        "Provide only one experiment when providing "
                        "pretrained path. If pretrianed path is not "
                        "provided, epoch with best validation metric "
                        "is used for evaluation.")
    parser.add_argument('--results', type=str, default="",
                        help="Path to the CSV file to store results.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Path to the directory to store outputs.")

    # System params
    parser.add_argument('--n_items', type=int, default=None,
                        help="Number of items to test.")
    parser.add_argument('--pretrain_path', type=str, default="best",
                        help="Path to pretrained weights")
    parser.add_argument('--profiling', dest='profiling', action='store_true',
                        help="Enable or disable profiling.")
    parser.add_argument('--use_cuda', dest='use_cuda', action='store_true',
                        help="Whether to use cuda")
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=None,
                        help="List of GPU ids used for training. "
                        "Eg., --gpu_ids 2 4. All GPUs are used by default.")
    args = parser.parse_args()

    results = []
    unique_hparams = get_unique_hparams(args.experiments)
    if len(unique_hparams) == 0:
        unique_hparams = [{}]

    for exp_dir, hparams in zip(args.experiments, unique_hparams):
        eval_args = argparse.Namespace(**vars(args))
        eval_args.exp_dir = exp_dir

        utils.set_logger(os.path.join(exp_dir, 'eval.log'))
        logging.info("Evaluating %s ..." % exp_dir)

        # Load model and training params
        params = utils.Params(os.path.join(exp_dir, 'config.json'))
        for k, v in params.__dict__.items():
            vars(eval_args)[k] = v

        network = importlib.import_module(eval_args.model)
        logging.info("Imported the model from '%s'." % eval_args.model)

        curr_res = evaluate(network, eval_args)
        for k, v in hparams.items():
            curr_res[k] = v
        results.append(curr_res)

        del eval_args

    if args.results != "":
        print("Writing results to %s" % args.results)
        pd.DataFrame(results).to_csv(args.results, index=False)
