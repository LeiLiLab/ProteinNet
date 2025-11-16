from dataclasses import dataclass, field
import logging
from typing import Optional
from omegaconf import II
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from fairseq import utils
from fairseq.data import (
    data_utils,
    indexed_dataset,
    NCBIDataset,
    NCBIFinetuneDataset
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.models.esm_modules import Alphabet


device = torch.device("cuda")


logger = logging.getLogger(__name__)


def load_protein_dataset(
    data_path,
    split,
    protein,
    src_dict,
    dataset_impl_source,
    dataset_impl_target,
    data_stage,
    generation,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    epoch=1,
):
    src_dataset = data_utils.load_indexed_dataset(
        data_path, src_dict, dataset_impl_source, source=True, split=split, protein=protein, data_stage=data_stage
    )

    if split == "train":
        train = True
    else:
        train = False
    motif_dataset = data_utils.load_indexed_dataset(
        data_path, src_dict, "motif", source=False, sizes=src_dataset.sizes, epoch=epoch, train=train, split=split,
        protein=protein, data_stage=data_stage
    )

    tgt_dataset = data_utils.load_indexed_dataset(
        data_path, src_dict, dataset_impl_target, source=False, motif_list=motif_dataset.motif_list, split=split,
        protein=protein, data_stage=data_stage
    )
    
    pdb_dataset = data_utils.load_indexed_dataset(data_path, dataset_impl="pdb", split=split, protein=protein, data_stage=data_stage)
    ncbi_dataset = data_utils.load_indexed_dataset(data_path, dataset_impl="ncbi", split=split, protein=protein, data_stage=data_stage)
    logger.info(
        "{} {} {} examples".format(
            data_path, split, len(src_dataset
        )
    ))
    
    if generation or data_stage == "finetuning":
        return NCBIFinetuneDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset.sizes,
        motif_dataset,
        motif_dataset.sizes,
        pdb_dataset,
        ncbi_dataset,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=None,
        eos=src_dict.eos_idx,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple)
    else:
        ligand_atom_dataset = data_utils.load_indexed_dataset(data_path, dataset_impl="ligand_atom",
                                                              split=split, protein=protein)
        ligand_coor_dataset = data_utils.load_indexed_dataset(data_path, dataset_impl="ligand_coor",
                                                              split=split, protein=protein)
        ligand_bind_dataset = data_utils.load_indexed_dataset(data_path, dataset_impl="ligand_binding",
                                                              split=split, protein=protein)
        return NCBIDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset.sizes,
            motif_dataset,
            motif_dataset.sizes,
            pdb_dataset,
            ncbi_dataset,
            ligand_atom_dataset,
            ligand_coor_dataset,
            ligand_bind_dataset,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=None,
            eos=src_dict.eos_idx,
            num_buckets=num_buckets,
            shuffle=shuffle,
            pad_to_multiple=pad_to_multiple)
    

@dataclass
class GeometricProteinDesignConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Training and validation data path"
        },
    )
    protein_task: str = field(
        default="myoglobin",
        metadata={"help": "protein task name"}
    )
    left_pad_source: bool = field(
        default=False, metadata={"help": "pad the source on the left"}
    )
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl_source: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="raw", metadata={"help": "data format of source data"}
    )
    dataset_impl_target: Optional[ChoiceEnum(get_available_dataset_impl())] = field(
        default="coor", metadata={"help": "data format of target data"}
    )
    generation: bool = field(
        default=False, metadata={
            "help": "whether design proteins in inference"
        }
    )
    decoding_strategy: str = field(
        default="greedy", metadata={
            "help": "decoding strategy in inference"
        }
    )
    topp_probability: float = field(
        default=0.2, metadata={
            "help": "p value for topp inference in inference"
        }
    )
    data_stage: str = field(
        default="pretraining-full", metadata={
            "help": "data stage to load of the current model: choose from: pretraining-mlm, pretraining-motif, pretraining-full, finetuning"
        }
    )


@register_task("geometric_protein_design", dataclass=GeometricProteinDesignConfig)
class GeometricProteinDesignTask(FairseqTask):
    """
    ProteinNet Task: trained with sequence recovery loss, structure prediction loss and
    protein-ligand binding prediciton loss to achieve co-design sequence and backbone structure
    """

    cfg: GeometricProteinDesignConfig

    def __init__(self, cfg: GeometricProteinDesignConfig, src_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.mask_idx = self.src_dict.mask_idx

    @classmethod
    def setup_task(cls, cfg: GeometricProteinDesignConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        the dictionary is composed of amino acids

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        # load dictionaries
        alphabet = Alphabet.from_architecture("ESM-1b")
        src_dict = alphabet

        return cls(cfg, src_dict)

    def load_dataset(self, split, epoch=1, combine=False, data_stage="pretraining-full", **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        data_path = self.cfg.data

        protein_task = self.cfg.protein_task
        print("data stage")
        print(data_stage)

        self.datasets[split] = load_protein_dataset(
            data_path,
            split,
            protein_task,
            self.src_dict,
            dataset_impl_source=self.cfg.dataset_impl_source,
            dataset_impl_target=self.cfg.dataset_impl_target,
            data_stage=data_stage,
            generation=self.cfg.generation,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            epoch=epoch,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        return model

    def valid_step(self, sample, model, criterion, topp_probability=0.2):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.generation:
            with torch.no_grad():
                source_input = sample["source_input"]
                src_tokens = source_input["src_tokens"]
                batch_size, n_nodes = src_tokens.size()[0], src_tokens.size()[1]

                target_input = sample["target_input"]
                motif = sample["motif"]
                output_mask = motif["output"]
                pdbs = sample["pdb"]
                centers = sample["center"]
                
                ncbi = sample["ncbi"]
                encoder_out, coords = model(src_tokens,
                                            source_input["src_lengths"],
                                            target_input["target_coor"],
                                            motif, ncbi)
                
                distribution = encoder_out
                coords = coords.reshape(batch_size, -1, 3)
                target_coor = sample["target_input"]["target_coor"]
                rmsd = torch.sqrt(torch.sum(torch.sum(torch.square(coords - target_coor), dim=-1) * output_mask, dim=-1) / torch.sum(output_mask, dim=-1))

                coords = (output_mask.unsqueeze(-1) * coords + (output_mask.unsqueeze(-1) == 0).int() * target_coor)[:, 1: -1, :]
                coords = coords + centers.unsqueeze(1)
                target_coor = target_coor + centers.unsqueeze(1)

                # top-k sampling
                if self.cfg.decoding_strategy == "top-k":
                    _, top_indices = torch.topk(encoder_out, k=3, dim=-1)
                    # indexes = top_indices[:, :, -1]
                    index_selects = torch.tensor(np.random.randint(low=0, high=3, size=(encoder_out.size(0), encoder_out.size(1)))).to(device).unsqueeze(-1)
                    indexes = top_indices.gather(index=index_selects, dim=-1).squeeze(-1)
                
                # argmax
                elif self.cfg.decoding_strategy == "greedy":
                    encoder_out[:, 1: -1, : 4] = -math.inf
                    encoder_out[:, :, 24:] = -math.inf
                    indexes = torch.argmax(encoder_out, dim=-1)   # [batch, length]
                elif self.cfg.decoding_strategy == "top-p":
                    # top-p sampling
                    encoder_out[:, 1: -1, : 4] = 0
                    encoder_out[:, :, 24:] = 0
                    sorted_logits, sorted_indices = torch.sort(encoder_out, descending=True)
                    cumulative_probs = torch.cumsum(sorted_logits, dim=-1)  # [B,L, vocab]
                    sorted_indices_to_remove = cumulative_probs > topp_probability
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
                    encoder_out[indices_to_remove] = 0
                    indexes = torch.multinomial(encoder_out.view(-1, encoder_out.size(-1)), 1).reshape(batch_size, -1)
                else:
                    raise NotImplementedError

                indexes = output_mask * indexes + (output_mask == 0).int() * source_input["src_tokens"]
                srcs = [model.encoder.alphabet.string(source_input["src_tokens"][i]) for i in range(source_input["src_tokens"].size(0))]
                strings = [model.encoder.alphabet.string(indexes[i]) for i in range(len(indexes))]
                probs = torch.gather(distribution, dim=-1, index=indexes.unsqueeze(-1)).squeeze(-1)[:, 1: -1]
                return loss, sample_size, logging_output, strings, srcs, pdbs, coords, target_coor, rmsd, probs
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    def max_positions(self):
        """Return the max length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return None