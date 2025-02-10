#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys
import time
from argparse import Namespace
from itertools import chain
import numpy as np

import torch
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import metrics, progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig
import generate_pdb_file


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("Inference")


def find_zero_fragments(arr):
    zero_fragments = []
    start = None

    for i, val in enumerate(arr):
        if val == 0 and start is None:
            # Start of a new zero fragment
            start = i
        elif val != 0 and start is not None:
            # End of a zero fragment
            zero_fragments.append((start, i - 1))
            start = None

    # If the array ends with a zero fragment, add it
    if start is not None:
        zero_fragments.append((start, len(arr) - 1))

    return zero_fragments


def main(cfg: DictConfig, override_args=None):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    reset_logging()

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    if cfg.distributed_training.distributed_world_size > 1:
        data_parallel_world_size = distributed_utils.get_data_parallel_world_size()
        data_parallel_rank = distributed_utils.get_data_parallel_rank()
    else:
        data_parallel_world_size = 1
        data_parallel_rank = 0

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=overrides,
        suffix=cfg.checkpoint.checkpoint_suffix,
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(saved_cfg)

    # Build criterion
    criterion = task.build_criterion(saved_cfg.criterion)
    criterion.eval()
    start = time.time()
    total_rmsds = []
    for sampling_round in range(1, 2):
        print("sampling {}".format(sampling_round))
        try:
            task.load_dataset("test", combine=False, epoch=1, task_cfg=saved_cfg.task)
            dataset = task.dataset("test")
        except KeyError:
            raise Exception("Cannot find dataset: " + "test")

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=cfg.dataset.max_tokens,
            max_sentences=cfg.dataset.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
            seed=cfg.common.seed,
            num_shards=data_parallel_world_size,
            shard_id=data_parallel_rank,
            num_workers=cfg.dataset.num_workers,
            data_buffer_size=cfg.dataset.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            prefix=f"valid on test subset",
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        )

        log_outputs = []

        os.system("mkdir -p {}{}".format(cfg.common_eval.results_path, sampling_round))
        fw_antibody_h1 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH1.true.txt"), "w")
        fw_generation_h1 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH1.gen.txt"), "w")
        fw_antibody_h2 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH2.true.txt"), "w")
        fw_generation_h2 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH2.gen.txt"), "w")
        fw_antibody_h3 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH3.true.txt"), "w")
        fw_generation_h3 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRH3.gen.txt"), "w")
        fw_antibody_l1 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL1.true.txt"), "w")
        fw_generation_l1 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL1.gen.txt"), "w")
        fw_antibody_l2 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL2.true.txt"), "w")
        fw_generation_l2 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL2.gen.txt"), "w")
        fw_antibody_l3 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL3.true.txt"), "w")
        fw_generation_l3 = open(os.path.join("{}{}".format(cfg.common_eval.results_path, sampling_round), "CDRL3.gen.txt"), "w")

        for i, sample in enumerate(progress):

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            _loss, _sample_size, log_output, _strings, _srcs, coords, target = task.valid_step(sample, model, criterion)
            
            progress.log(log_output, step=i)
            log_outputs.append(log_output)

            for idx, tgt in enumerate(target):
                assert len(_srcs[idx]) == len(tgt) -2
                index = list(tgt.cpu()).index(0)-1

                true_cdrh1, gen_cdrh1, true_cdrh2, gen_cdrh2, true_cdrh3, gen_cdrh3 = [], [], [], [], [], []
                true_cdrl1, gen_cdrl1, true_cdrl2, gen_cdrl2, true_cdrl3, gen_cdrl3 = [], [], [], [], [], []

                fragments = find_zero_fragments(tgt)
                assert len(fragments) == 6
                
                for ind in range(fragments[0][0], fragments[0][1]+1):
                    true_cdrh1.append(_srcs[idx][ind-1])
                    gen_cdrh1.append(_strings[idx][ind-1])
                
                for ind in range(fragments[1][0], fragments[1][1]+1):
                    true_cdrh2.append(_srcs[idx][ind-1])
                    gen_cdrh2.append(_strings[idx][ind-1])
                    
                for ind in range(fragments[2][0], fragments[2][1]+1):
                    true_cdrh3.append(_srcs[idx][ind-1])
                    gen_cdrh3.append(_strings[idx][ind-1])
                
                for ind in range(fragments[3][0], fragments[3][1]+1):
                    true_cdrl1.append(_srcs[idx][ind-1])
                    gen_cdrl1.append(_strings[idx][ind-1])
                    
                for ind in range(fragments[4][0], fragments[4][1]+1):
                    true_cdrl2.append(_srcs[idx][ind-1])
                    gen_cdrl2.append(_strings[idx][ind-1])
                
                for ind in range(fragments[5][0], fragments[5][1]+1):
                    true_cdrl3.append(_srcs[idx][ind-1])
                    gen_cdrl3.append(_strings[idx][ind-1])

                fw_antibody_h1.write("".join(true_cdrh1) + "\n")
                fw_generation_h1.write("".join(gen_cdrh1) + "\n")
                fw_antibody_h2.write("".join(true_cdrh2) + "\n")
                fw_generation_h2.write("".join(gen_cdrh2) + "\n")
                fw_antibody_h3.write("".join(true_cdrh3) + "\n")
                fw_generation_h3.write("".join(gen_cdrh3) + "\n")
                fw_antibody_l1.write("".join(true_cdrl1) + "\n")
                fw_generation_l1.write("".join(gen_cdrl1) + "\n")
                fw_antibody_l2.write("".join(true_cdrl2) + "\n")
                fw_generation_l2.write("".join(gen_cdrl2) + "\n")
                fw_antibody_l3.write("".join(true_cdrl3) + "\n")
                fw_generation_l3.write("".join(gen_cdrl3) + "\n")
            fw_antibody_h1.flush()
            fw_generation_h1.flush()
            fw_antibody_h2.flush()
            fw_generation_h2.flush()
            fw_antibody_h3.flush()
            fw_generation_h3.flush()
            fw_antibody_l1.flush()
            fw_generation_l1.flush()
            fw_antibody_l2.flush()
            fw_generation_l2.flush()
            fw_antibody_l3.flush()
            fw_generation_l3.flush()

        fw_antibody_h1.close()
        fw_generation_h1.close()
        fw_antibody_h2.close()
        fw_generation_h2.close()
        fw_antibody_h3.close()
        fw_generation_h3.close()
        fw_antibody_l1.close()
        fw_generation_l1.close()
        fw_antibody_l2.close()
        fw_generation_l2.close()
        fw_antibody_l3.close()
        fw_generation_l3.close()

        if data_parallel_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=cfg.common.all_gather_list_size,
                group=distributed_utils.get_data_parallel_group(),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        print("inference time: {}".format(time.time()-start))
        progress.print(log_output, tag="test", step=i)


def cli_main():
    parser = options.get_validation_parser()
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    override_args = options.parse_args_and_arch(
        override_parser, suppress_defaults=True
    )

    # distributed_utils.call_main(
    #     convert_namespace_to_omegaconf(args), main, override_args=override_args
    # )

    main(convert_namespace_to_omegaconf(args), override_args)


if __name__ == "__main__":
    cli_main()
