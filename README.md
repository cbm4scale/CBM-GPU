# CBM-GPU
Implementation of the CBM format matrix multiplication kernels in CUDA and cuSPARSE.

> **Note:** This repository requires refactoring, and in the future it will be merged with CBM-CPU. 

## Setup

1. **Install Intel oneAPI Base Toolkit**  
   Download and install the Intel oneAPI Base Toolkit following the instructions provided [here](https://www.intel.com/content/www/us/en/docs/oneapi/installation-guide-linux/2024-2/overview.html).

2. **Create a Conda Environment**  
   Set up a new Conda environment and install the necessary dependencies:
   ```bash
   conda create -n cbm python=3.11
   conda activate cbm
   conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
   pip uninstall numpy
   pip install numpy==1.24.3
   pip install torch_geometric
   pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
   pip install ogb
   conda install cmake ninja wget prettytable scipy
    ```

3. **Clone and Install the Repository**  
   Clone the repository and set up the project:
   ```bash
   git clone https://github.com/cbm4scale/CBM-GPU.git --recursive
   cd CBM-GPU/
   git submodule init
   git submodule update
   python setup.py  # If Intel oneAPI is not installed in the default directory, use: --setvars_path PATH_TO_ONEAPI/setvars.sh
   export LD_LIBRARY_PATH=./arbok/build/:$LD_LIBRARY_PATH
   export PYTHONPATH=./:$PYTHONPATH
   ```
## Usage

### `./scripts/matmul.sh`
This script evaluates the performance of different matrix multiplication methods with both CBM and CSR formats using:  
   - `cbm/cbm4{mm,ad,dad}.py` and `cbm/mkl4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.
   - The alpha values used to convert the dataset to CBM format are defined in `benchmark/utilities.py`.

Upon completion, the script generates a results file named `results/matmul_results.txt`, which records time related metrics for matrix multiplication.

> **Note:** `cbm/cbm4ad.py` and `cbm/mkl4ad.py` contain python classes to store matrix **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 

> **Note:** `cbm/cbm4dad.py` and `cbm/mkl4dad.py` contain python classes to store matrix **D^(-1/2)** @ **A** @ **D^(-1/2)** in CBM and CSR format, and support matrix products of the form **D^(-1/2)** @ **A** @ **D^(-1/2)** @ **X**.
> Here, **A** is the adjacency matrix of the dataset, **D** is the diagonal degree matrix of **A**, and **X** is a dense real-valued matrix. 


#### How to Run:
1. Open the `scripts/matmul.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your CPU - used to accelerate the construction of the CBM format.
   
2. Run `./scripts/matmul.sh` inside the `CBM-CPU/` direction.  

Other configuration options (use default values to reproduce our experiments):    
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `WARMUPS=(...)`  
        Include in this array the number of warmup iterations to be run before recording starts.


### `./scripts/validate.sh`
This script validates the correction different matrix multiplication methods with CBM formats using: 
- `cbm/cbm4{mm,ad,dad}.py` via `benchmark/benchmark_matmul.py`.

This validation is performed by comparing the resulting matrices (element-wise) between the classes mentioned above and `cbm/mkl4{mm,ad,dad}.py`.
Again, the alpha values used are the ones set in `benchmark/utilities.py`.

#### How to Run:
1. Open the `scripts/validate.sh` and modify the following variables:
   - `MAX_THREADS=...`  
     Set this variable to the maximum number of physical cores on your CPU - used to accelerate the construction of the CBM format.
       
2. Run `./scripts/validate.sh` inside the `CBM-CPU/` direction.

Other configuration options (use default values to reproduce our experiments):  
   - `DATASETS=(...)`  
       Include in this array the datasets that should be considered..  
   
   - `NCOLUMNS=(...)`  
        Include in this array the number of columns (of the random operand matrices) you want to experiment with.
     
   - `ITERATIONS=(...)`  
        Include in this array the number of matrix multiplications to be measured.

   - `RTOL=...`  
        Specifies the relative tolerance interval to be considered during validation.

   - `ATOL=...`  
        Specifies the absolute tolerance interval to be considered in the validation.
