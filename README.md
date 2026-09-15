LeWorldModel with Landscape Shaping (EQM-LeWM)
Stable End-to-End Joint-Embedding Predictive Architecture with Action-Landscape Shaping
This repository extends LeWorldModel (LeWM) with Equilibrium Matching (EQM) / Landscape Shaping for latent planning. It supports training and evaluation both locally and on cloud environments like Kaggle (T4 / P100 / Dual T4).  
PDF

🔬 Method Overview
Standard Joint-Embedding Predictive Architectures (JEPAs) train a predictive world model by minimizing prediction error in latent space while applying anti-collapse regularizers such as SIGReg. However, when latents are used for gradient-based or sampling-based planning, Euclidean distance often fails to match environment topology—causing action optimization to stall at local minima or cut through infeasible transitions.  
PDF
+ 1

This implementation introduces an Equilibrium Matching (EQM) auxiliary loss (lejepa_forward) that explicitly conditions the action-energy gradient field:  
PDF

Latent Predictive Step:

L 
pred
​
 =E[∥ 
z
^
  
t+1:t+K
​
 −z 
t+1:t+K
​
 ∥ 
2
2
​
 ]
  
PDF
Action-Interpolation & Energy:
For noisy action paths a 
γ
​
 =γa+(1−γ)ϵ with ϵ∼N(0,I) and γ∼U(0,1), the prediction energy is:
  
PDF

E 
θ
​
 (a 
γ
​
 )=∥p 
ϕ
​
 (z 
c
​
 ,g 
ψ
​
 (a 
γ
​
 ))−z 
g
​
 ∥ 
2
2
​
 
  
PDF
Landscape Shaping Gradient Penalty:

L 
eqm
​
 =E 
a,ϵ,γ
​
 [∥∇ 
a 
γ
​
 
​
 E 
θ
​
 (a 
γ
​
 )−κ(1−γ)(ϵ−a)∥ 
2
2
​
 ]
  
PDF
Composite Training Objective:

L=L 
pred
​
 +αL 
eqm
​
 +λL 
sigreg
​
 
  
PDF
This regularizes local descent directions toward feasible dynamic paths, preventing pathological trajectories in topologically constrained tasks (e.g., Two Rooms) and contact-rich environments (e.g., PushT).  
PDF

📁 Repository Structure
Plaintext


├── config/
│   ├── train/            # Hydra training configs (lewm.yaml, data overrides)
│   └── eval/             # Evaluation configs (tworoom.yaml, pusht.yaml)
├── jepa.py               # JEPA architecture, rollout logic, & MPC criterion
├── module.py             # ViT, ARPredictor, Embedder, MLP, SIGReg modules
├── train.py              # Main training script (Lightning + Hydra + SPT)
├── eval.py               # Model predictive control / planner evaluation
└── utils.py              # Callbacks, image transforms, and normalizers
⚙️ Installation
Local Setup (Recommended: Python 3.10)
Bash


# Clone the repository
git clone https://github.com/RodrickKamsiyonna/le-wm_kaggle.git
cd le-wm_kaggle

# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y zstd swig

# Install package dependencies
pip install --upgrade pip
pip install hydra-core lightning omegaconf einops wandb
pip install "huggingface-hub>=0.34.0,<1.0" "datasets<3.0" "transformers>=4.45.0"
pip install "stable-worldmodel[all]==0.1.1" "stable-pretraining==0.1.7"
📦 Data Preparation
Data is stored as compressed HDF5 archives (.tar.zst). Set $STABLEWM_HOME to specify the local cache directory:

Bash


export STABLEWM_HOME="$HOME/.stable-wm"
mkdir -p "$STABLEWM_HOME"

# Example: Download and extract Two Rooms
python -c "
from huggingface_hub import hf_hub_download
import os
path = hf_hub_download(repo_id='quentinll/lewm-tworooms', filename='tworoom.tar.zst', repo_type='dataset')
os.system(f'tar --zstd -xvf {path} -C $STABLEWM_HOME')
"
🏋️ Training
Local Run
Run the default training pipeline with standard LeJEPA or EQM loss enabled:

Bash


export HYDRA_FULL_ERROR=1
python train.py data=tworoom
To configure hyperparameters or ablation weights directly from the CLI:

Bash


# Disable EQM loss (baseline ablation)
python train.py data=tworoom loss.eqm_pred_weight=0.0

# Disable SIGReg (lambda=0 ablation)
python train.py data=tworoom loss.sigreg.weight=0.0
Kaggle GPU Execution
Open a new Kaggle Notebook and attach an accelerator (GPU T4 x 2 or P100).

Set up environment variables and directories:

Python


import os
os.environ["STABLEWM_HOME"] = "/kaggle/working/stablewm"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.makedirs("/kaggle/working/stablewm", exist_ok=True)
os.makedirs("/kaggle/data", exist_ok=True)
Download dependencies & data:

Bash


!apt-get update && !apt-get install -y zstd swig
!pip install -q hydra-core lightning omegaconf einops wandb
!pip install -q "huggingface-hub>=0.34.0,<1.0" "datasets<3.0" "transformers>=4.45.0"
!pip install -q "stable-worldmodel[all]==0.1.1" "stable-pretraining==0.1.7"

# Fetch data directly
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='quentinll/lewm-tworooms', filename='tworoom.tar.zst', repo_type='dataset', local_dir='/kaggle/data')
"
!tar --zstd -xvf /kaggle/data/tworoom.tar.zst -C /kaggle/working/stablewm/
Authenticate WandB and launch:

Python


from kaggle_secrets import UserSecretsClient
import wandb

user_secrets = UserSecretsClient()
wandb.login(key=user_secrets.get_secret("wandb_key"))
Bash


!python train.py data=tworoom
🎯 Evaluation & Planning
Evaluation configs reside in config/eval/. Provide the policy path relative to $STABLEWM_HOME (omit _object.ckpt), or supply the direct filepath:

Bash


# Using policy checkpoint path
python eval.py --config-name=tworoom.yaml policy=/kaggle/working/lewm_run/lewm_epoch_10_object
Python API Inference
To load a serialized cost model inside a custom model predictive control (MPC) loop:

Python


import stable_worldmodel as swm

# Load serialized object policy
cost_model = swm.policy.AutoCostModel("pusht/lewm")
📚 Citation
If you use this repository or its extensions in your research, please cite:

Code snippet


@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}

@article{landscapeshape2026,
  title={Landscape Shaping for Stable Latent Planning},
  journal={Working Paper},
  year={2026}
}
