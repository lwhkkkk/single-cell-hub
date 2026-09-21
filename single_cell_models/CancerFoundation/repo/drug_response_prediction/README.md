## Drug response prediction

### Installation

#### Prerequisites

Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine.

#### Step-by-step guide
1. **Create the Conda environment**:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate environment**:
   ```bash
   conda activate cancergpt_drp
   ```

3. **Download data**:

   Please download the pretrained model from [this link](https://polybox.ethz.ch/index.php/s/UxANnzU9q3WGlNA), unzip it, and place the three folders (cell_line, drug, embedding) in this directory.

4. **(Optional) Download results for hold-out cell lines**:
   Download the results for hold-out cell lines from [this link](https://polybox.ethz.ch/index.php/s/ir0KNi2QXQnhrN1).

### Hold-out cell line

Run the following command for drug response prediction on hold-out cell lines for **CancerGPT** embeddings.
```bash
python drp.py --embedding_path "./data/embedding/CancerGPT_embedding.csv" --gpu_id 0
```
For **scFoundation** embeddings, use:
```bash
python drp.py --embedding_path "./data/embedding/scFoundation_embedding.csv" --gpu_id 0
```
For **raw gene expression** data (DeepCDR), use:
```bash
python drp.py --embedding_path "./data/embedding/gene_expression.csv" --gpu_id 0
```

Alternatively, download the results from the link specified in Step 4. Corresponding plots can be generated using the `plot.ipynb`notebook.

### Hold-out drug

In order to derive results for all drugs and all embeddings, run the following:
```bash
   bash run_all.sh
   ```

Results will be saved in the `results` folder.