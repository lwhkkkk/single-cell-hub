from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union
from anndata import AnnData
import os
from model.cancergpt import CancerGPT
from model.data_collator import DataCollator
from model.dataset import Dataset

from .utils import load_pretrained
from model.vocab import GeneVocab
import scanpy as sc
import torch
import numpy as np
import json
from torch.utils.data import DataLoader, SequentialSampler
PathLike = Union[str, os.PathLike]


def embed(
    adata_or_file: Union[AnnData, PathLike],
    model_dir: PathLike,
    batch_key: Optional[str] = None,
    max_length: int = 1200,
    batch_size: int = 64,
    obs_to_save: Optional[list] = None,
    device: Union[str, torch.device] = "cuda",
    normalize: bool = True,
) -> AnnData:
    if isinstance(adata_or_file, AnnData):
        adata = adata_or_file
    else:
        adata = sc.read_h5ad(adata_or_file)

    if isinstance(obs_to_save, str):
        assert obs_to_save in adata.obs, f"obs_to_save {obs_to_save} not in adata.obs"
        obs_to_save = [obs_to_save]

    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Using CPU instead.")

    # LOAD MODEL
    model_dir = Path(model_dir)
    vocab_file = model_dir / "vocab.json"
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "model.pth"
    pad_token = "<pad>"
    pad_value = -2

    # vocabulary
    vocab = GeneVocab.from_file(vocab_file)

    adata.var["genes"] = adata.var.index

    adata.var["id_in_vocab"] = [
        vocab[gene] if gene in vocab else -1 for gene in adata.var["genes"]
    ]

    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    sc.pp.highly_variable_genes(
    adata, n_top_genes=max_length-1, flavor='cell_ranger', batch_key=batch_key)
    adata = adata[:, adata.var['highly_variable']]
    adata.var["genes"] = adata.var.index

    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    vocab.set_default_index(vocab["<pad>"])
    genes = adata.var["genes"].tolist()
    gene_ids = np.array(vocab(genes), dtype=int)

    model = CancerGPT(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        dropout=model_configs["dropout"],
        pad_token=pad_token,
    )

    model = load_pretrained(model, torch.load(
        model_file, map_location=device), verbose=False)

    model.to(device)
    model.eval()

    count_matrix = adata.X
    count_matrix = (
        count_matrix if isinstance(
            count_matrix, np.ndarray) else count_matrix.A
    )

    dataset = Dataset(
        count_matrix, gene_ids, vocab, pad_value=pad_value
    )
    collator = DataCollator(
        do_padding=True,
        pad_token_id=vocab[pad_token],
        pad_value=pad_value,
        do_mlm=False,
        do_binning=True,
        max_length=max_length,
        sampling=True,
        keep_first_n_tokens=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        collate_fn=collator,
        drop_last=False,
        num_workers=min(len(os.sched_getaffinity(0)), batch_size),
        pin_memory=True,
    )

    device = next(model.parameters()).device
    cell_embeddings = np.zeros(
        (len(dataset), model_configs["embsize"]), dtype=np.float32
    )
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        count = 0
        for data_dict in tqdm(data_loader, desc="Embedding cells"):
            input_gene_ids = data_dict["gene"].to(device)
            src_key_padding_mask = input_gene_ids.eq(
                vocab[pad_token]
            )

            embeddings = model.encode(
                input_gene_ids,
                data_dict["expr"].to(device),
                src_key_padding_mask=src_key_padding_mask,
            )

            # get the <cls> position embedding
            embeddings = embeddings[:, 0, :]
            embeddings = embeddings.cpu().numpy()
            cell_embeddings[count: count + len(embeddings)] = embeddings
            count += len(embeddings)
    
    if normalize:
        cell_embeddings = cell_embeddings / np.linalg.norm(
            cell_embeddings, axis=1, keepdims=True
        )

    adata.obsm["CancerGPT"] = cell_embeddings
    return adata
