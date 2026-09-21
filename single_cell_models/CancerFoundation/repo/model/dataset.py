import torch
import numpy as np


class Dataset(torch.utils.data.Dataset):
    def __init__(self, count_matrix, gene_ids, vocab, pad_value, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.pad_value = pad_value

    def __len__(self):
        return len(self.count_matrix)

    def __getitem__(self, idx):
        row = self.count_matrix[idx]
        values = row
        genes = self.gene_ids
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.pad_value)
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output
