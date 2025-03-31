# CBM-GPU

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
