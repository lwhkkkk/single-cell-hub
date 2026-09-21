## Zero-shot batch integration

### Installation
Refer to the instructions in the parent directory in regards to the conda environment, i.e., `../README.md`.

#### Step-by-step guide

1. **Download baseline data**:

    Please download the scRNA-seq glioblastoma dataset [1], as well as the baseline embeddings from [this link](https://polybox.ethz.ch/index.php/s/ddfx3WfKLYkK712), and unzip it. Alternatively, you can generate the embeddings yourself by following the instructions in the scGPT github repository [here](https://github.com/bowang-lab/scGPT/blob/main/tutorials/zero-shot/Tutorial_ZeroShot_Integration.ipynb).

2. **Generate CancerGPT embeddings**:

    Run the following command to generate CancerGPT embeddings:
    ```bash
   python generate_embedding.py
   ```

3. **Plot the results**:

    Refer to the `plot.ipynb` notebook to generate plots for the different embeddings.       

## References
[1] Neftel, Cyril, Julie Laffy, Mariella G. Filbin, Toshiro Hara, Marni E. Shore, Gilbert J. Rahme, Alyssa R. Richman, et al. "An integrative model of cellular states, plasticity, and genetics for glioblastoma." _Cell_ 178, no. 4 (2019): 835–849. Elsevier.