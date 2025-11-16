import json
import random
from functools import lru_cache
import numpy as np
import torch
from fairseq.dataclass.constants import DATASET_IMPL_CHOICES
from fairseq.file_io import PathManager

from . import FairseqDataset

from typing import Union

def best_fitting_int_dtype(
    max_int_to_represent,
) -> Union[np.uint16, np.uint32, np.int64]:

    if max_int_to_represent is None:
        return np.uint32  # Safe guess
    elif max_int_to_represent < 65500:
        return np.uint16
    elif max_int_to_represent < 4294967295:
        return np.uint32
    else:
        return np.int64


def get_available_dataset_impl():
    return list(map(str, DATASET_IMPL_CHOICES))


def make_dataset(path, impl, fix_lua_indexing=False, dictionary=None, source=True, sizes=None, motif_list=None,
                 epoch=1, train=True, split="train", protein=None, data_stage="pretraining-full"):
    if impl == "raw" and IndexedRawTextDataset.exists(path):
        assert dictionary is not None
        return IndexedRawTextDataset(path, dictionary, source=source, split=split, protein=protein, data_stage=data_stage)
    elif impl == "coor" and CoordinateDataset.exists(path):
        return CoordinateDataset(path, motif_list, split=split, protein=protein, data_stage=data_stage)
    elif impl == "motif" and ProteinMotifDataset.exists(path):
        return ProteinMotifDataset(path, sizes, epoch, train, split=split, protein=protein, data_stage=data_stage)
    elif impl == "pdb" and ProteinPDBDataset.exists(path):
        return ProteinPDBDataset(path, split=split, protein=protein, data_stage=data_stage)
    elif impl == "ncbi" and NCBITaxonomyDataset.exists(path):
        return NCBITaxonomyDataset(path, split=split, protein=protein, data_stage=data_stage)
    elif impl == "protein_classification" and ProteinClassificationDataset.exists(path):
        return ProteinClassificationDataset(path, dictionary, split=split)
    elif impl == "ligand_atom" and LigandAtomDataset.exists(path):
        return LigandAtomDataset(path, split=split, protein=protein)
    elif impl == "ligand_coor" and LigandCoordinateDataset.exists(path):
        return LigandCoordinateDataset(path, split=split, protein=protein)
    elif impl == "ligand_binding" and LigandBindingDataset.exists(path):
        return LigandBindingDataset(path, split=split, protein=protein)
    return None


class ProteinClassificationDataset(FairseqDataset):
    """Protein Scorer: Loading discriminator dataset"""

    def __init__(self, path, dictionary, split="train"):
        self.tokens_list = []
        self.values = []
        self.sizes = []
        self.read_data(path, dictionary, split)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary, split):
        f = open(path)
        data = json.load(f)
        lines = data[split]["seq"]

        for line in lines:
            line = line.strip()
            prepend_bos = True
            # dictionary.encode(tokens).ids
            tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=prepend_bos,
                    append_eos=True,
                    reverse_order=False,
                ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

        labels = data[split]["label"]
        for label in labels:
            self.values.append(label)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.tokens_list[i], self.values[i])

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class IndexedRawTextDataset(FairseqDataset):
    """Protein Sequence Dataset Loading"""

    def __init__(self, path, dictionary, source=True, append_eos=True, reverse_order=False, split="train", protein=None, data_stage="pretraining-full"):
        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.source = source
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary, split, protein, data_stage)
        self.size = len(self.tokens_list)

    def read_data(self, path, dictionary, split, protein, data_stage):
        f = open(path)
        data = json.load(f)

        lines = []
        # pretrain
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:
                    lines.extend(data[protein][split]["seq"])
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["seq"])
            else:
                lines = data[protein][split]["seq"]
        else:
            # finetune
            if split in ["valid", "train"]:
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["seq"])
            elif split == "test":
                for protein in data:
                    for cate in data[protein]:
                        lines.extend(data[protein][cate]["seq"])

        for line in lines:
                line = line.strip()
                self.lines.append(line)
                if self.source:
                    prepend_bos = True
                else:
                    prepend_bos = False
                tokens = dictionary.encode_line(
                    line,
                    add_if_not_exist=False,
                    prepend_bos=prepend_bos,
                    append_eos=self.append_eos,
                    reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class CoordinateDataset(FairseqDataset):
    """ Protein Structure Dataset Loading"""

    def __init__(self, path, motif_list, split="train", protein=None, data_stage="pretraining-full"):
        self.coors_list = []
        self.lines = []
        self.sizes = []
        self.centers = []
        self.read_data(path, motif_list, split, protein, data_stage)
        self.size = len(self.coors_list)

    def read_data(self, path, motif_list, split, protein, data_stage):
        f = open(path)
        data = json.load(f)
     
        lines = []
        # pretrain
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:
                    lines.extend(data[protein][split]["coor"])
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["coor"])
            else:
                lines = data[protein][split]["coor"]
        else:
            # finetune
            if split in ["valid", "train"]:
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["coor"])
            elif split == "test":
                for protein in data:
                    for cate in data[protein]:
                        lines.extend(data[protein][cate]["coor"])

        for ind, line in enumerate(lines):
                line = line.strip()
                self.lines.append(line)
                coors = line.split(",")
                protein_coor = []
                for i in range(0, len(coors), 3):
                    protein_coor.append([float(coors[i]), float(coors[i+1]), float(coors[i+2])])
                protein_coor = torch.tensor(np.array(protein_coor), dtype=torch.float32)
                mask = (motif_list[ind][1: -1] == 0).int().unsqueeze(-1)
                mean_coor = torch.sum(protein_coor * mask, dim=0) / mask.sum()
                protein_coor = protein_coor - mean_coor
                protein_coor = torch.cat([torch.tensor([[0, 0, 0]]), protein_coor, torch.tensor([[0, 0, 0]])], dim=0)
                self.coors_list.append(protein_coor)
                self.sizes.append(len(protein_coor))
                self.centers.append(mean_coor)

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return (self.coors_list[i], self.centers[i])

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class ProteinMotifDataset(FairseqDataset):
    """ Functionally Important Site Dataset Loading"""

    def __init__(self, path, dataset_sizes, epoch, train, split="train", protein=None, data_stage="pretraining-full"):
        self.motif_list = []
        self.sizes = []
        self.epoch = epoch
        self.read_data(path, dataset_sizes, self.epoch, train, split, protein, data_stage)
        self.size = len(self.motif_list)

    def read_data(self, path, dataset_sizes, epoch, train, split, protein, data_stage):
        f = open(path)
        data = json.load(f)

        lines = []
        # pretrain
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:
                    lines.extend(data[protein][split]["motif"])
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["motif"])
            else:
                lines = data[protein][split]["motif"]
        else:
            # finetune
            if split in ["valid", "train"]:
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["motif"])
            elif split == "test":
                for protein in data:
                    for cate in data[protein]:
                        lines.extend(data[protein][cate]["motif"])

        for line, size in zip(lines, dataset_sizes):
                mask = np.ones(size)
                line = line.strip()
                indexes = line.split(",")
                if line != "":
                    if data_stage == "pretraining-mlm":
                        # masked language modeling
                        indexes = random.sample(list(range(1, size-1)), int(0.8*(size-2)))
                    else:
                        # motif
                        indexes = [int(index)+1 for index in indexes]
                    
                
                if line != "":
                    for ind in indexes:
                        mask[int(ind)] = 0
                mask[0] = 0
                mask[-1] = 0
                self.motif_list.append(torch.IntTensor(mask))
                self.sizes.append(len(mask))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.motif_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class ProteinPDBDataset(FairseqDataset):
    """ Protein ID (PDB or Uniprot ID) Dataset Loading"""

    def __init__(self, path, split="train", protein=None, data_stage="pretraining-full"):
        self.pdb_list = []
        self.sizes = []
        self.read_data(path, split, protein, data_stage)
        self.size = len(self.pdb_list)

    def read_data(self, path, split, protein, data_stage):
        f = open(path)
        data = json.load(f)

        lines = []

        # pretrain
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:
                    # ncbi
                    lines.extend(data[protein][split]["protein_id"])
                    # lines.extend(data[protein][split]["pdb"])
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["protein_id"])
                        # lines.extend(data[protein][split]["pdb"])
            else:
                # lines = data[protein][split]["pdb"]
                lines = data[protein][split]["protein_id"]
        else:
            # finetune
            if split in ["valid", "train"]:
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["protein_id"])
            elif split == "test":
                for protein in data:
                    for cate in data[protein]:
                        lines.extend(data[protein][cate]["protein_id"])

        for line in lines:
            self.pdb_list.append(line.strip())
            self.sizes.append(len(line.strip()))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.pdb_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class NCBITaxonomyDataset(FairseqDataset):
    """ Protein NCBI Taxonomy Dataset Loading"""
    def __init__(self, path, split="train", protein=None, data_stage="pretraining-full"):
        self.ncbi_list = []
        self.ncbi2id = json.load(open(
            "data/ncbi2id.json", "r"))
        self.sizes = []
        self.read_data(path, split, protein, data_stage)
        self.size = len(self.ncbi_list)

    def read_data(self, path, split, protein, data_stage):
        reaction_task = protein
        f = open(path)
        data = json.load(f)

        ncbi = []
        # pretrain
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:
                    ncbi_list = [protein for _ in range(len(data[protein][split]["protein_id"]))]
                    ncbi.extend(ncbi_list)
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        ncbi_list = [protein for _ in range(len(data[protein][split]["protein_id"]))]
                        ncbi.extend(ncbi_list)
            else:
                ncbi_list = [protein for _ in range(len(data[protein][split]["protein_id"]))]
                ncbi.extend(ncbi_list)
            
            for ncbi_item in ncbi:
                self.ncbi_list.append(self.ncbi2id[ncbi_item])
                self.sizes.append(1)
        else:
            # finetune
            if split in ["valid", "train"]:
                for protein in data:
                    if split in data[protein]:
                        ncbi_list = [protein for _ in range(len(data[protein][split]["protein_id"]))]
                        ncbi.extend(ncbi_list)
            elif split == "test":
                for protein in data:
                    for cate in data[protein]:
                        ncbi_list = [protein for _ in range(len(data[protein][cate]["protein_id"]))]
                        ncbi.extend(ncbi_list)

            for ncbi_item in ncbi:
                if ncbi_item not in self.ncbi2id:
                    if reaction_task == "61444":
                        ncbi_item = "818"
                    elif reaction_task == "18421":
                        ncbi_item = "562"
                    elif reaction_task == "20245":
                        ncbi_item = "99287"
                    elif reaction_task == "Thiopurine_S_methyltransferas":
                        ncbi_item = "9606"
                self.ncbi_list.append(self.ncbi2id[ncbi_item])
                self.sizes.append(1)
        
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.ncbi_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class LigandAtomDataset(FairseqDataset):
    """Protein Binding Ligand Atom Type Dataset Loading"""

    def __init__(self, path, split="train", protein=None):
        self.tokens_list = []
        self.sizes = []
        self.read_data(path, split, protein)
        self.size = len(self.tokens_list)

    def read_data(self, path, split, protein):
        f = open(path)
        data = json.load(f)
        
        lines = []
        if split == "train":
            for protein in data:
                ligand_features = [feature[0] for feature in data[protein][split]["ligand_feat"]]
                lines.extend(ligand_features)
        elif split == "valid":
            for protein in data:
                if split in data[protein]:
                    ligand_features = [feature[0] for feature in data[protein][split]["ligand_feat"]]
                    lines.extend(ligand_features)
        else:
            ligand_features = [feature[0] for feature in data[protein][split]["ligand_feat"]]
            lines.extend(ligand_features)

        for line in lines:
            features = torch.tensor(line)

            self.tokens_list.append(features)
            self.sizes.append(len(features))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class LigandCoordinateDataset(FairseqDataset):
    """ Protein Binding Ligand Structure Dataset Loading"""

    def __init__(self, path, split="train", protein=None):
        self.coors_list = []
        self.lines = []
        self.sizes = []
        self.read_data(path, split, protein)
        self.size = len(self.coors_list)

    def read_data(self, path, split, protein):
        f = open(path)
        data = json.load(f)
        
        lines = []
        if split == "train":
            for protein in data:
                ligand_coors = [feature[0] for feature in data[protein][split]["ligand_coor"]]
                lines.extend(ligand_coors)
        elif split == "valid":
            for protein in data:
                if split in data[protein]:
                    ligand_coors = [feature[0] for feature in data[protein][split]["ligand_coor"]]
                    lines.extend(ligand_coors)
        else:
            ligand_coors = [feature[0] for feature in data[protein][split]["ligand_coor"]]
            lines.extend(ligand_coors)

        for ind, substrate_coor in enumerate(lines):
                substrate_coor = torch.tensor(substrate_coor, dtype=torch.float32)

                mean_coor = torch.sum(substrate_coor, dim=0) / len(substrate_coor)
                substrate_coor = substrate_coor - mean_coor

                self.coors_list.append(substrate_coor)
                self.sizes.append(len(substrate_coor))

        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.coors_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)


class LigandBindingDataset(FairseqDataset):
    """Protein-Ligand Binding Label Dataset Loading"""

    def __init__(self, path, split="train", protein=None):
        self.tokens_list = []
        self.sizes = []
        self.centers = []
        self.read_data(path, split, protein)
        self.size = len(self.tokens_list)

    def read_data(self, path, split, protein):
        f = open(path)
        data = json.load(f)

        lines = []
        if split == "train":
            for protein in data:
                ligand_binding = [feature[0] for feature in data[protein][split]["binding"]]
                lines.extend(ligand_binding)
        elif split == "valid":
            for protein in data:
                if split in data[protein]:
                    ligand_binding = [feature[0] for feature in data[protein][split]["binding"]]
                    lines.extend(ligand_binding)
        else:
            ligand_binding = [feature[0] for feature in data[protein][split]["binding"]]
            lines.extend(ligand_binding)

        for line in lines:
                self.tokens_list.append(line)
                self.sizes.append(1)
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError("index out of range")

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)
        return self.tokens_list[i]

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return PathManager.exists(path)