# Single Image Dehazing using Untrained Neural Network (SID-UNN)

This repository is an implementation of the paper [Looking Beyond the Veil: Single Image Dehazing using Untrained Neural Network (SID-UNN)](https://www.deanfrancispress.com/index.php/te/article/download/3111/TE008662.pdf/12666)

### Abstract
Haze and fog refer to the suspension of atmospheric particles that significantly diminish visibility, which has always been a concerning issue in our daily lives. Performances of critical computer vision systems are often limited by the hazy weather, posing threats to security and road safety. However, many current dehazing methods rely on complex network or data prior from massive paired datasets which are difficult and costly to obtain. In this paper, I propose an improved dehazing model, SID-UNN, which uses an unsupervised network that requires neither pretraining nor data prior, with the network architecture adopted from Deep Image Prior (DIP). The model incorporates physical priors in estimating airtight and initializing transmission maps where the parameters are further optimized together with hyperparameters from the network, treating single image dehazing as a nonlinear optimization problem. This unique structure ensures that the estimation does not rely on handcrafted parameters, thus allowing better generalization ability and robustness in the model. Weighted Least Square filtering and smoothing constraints are innovatively applied so that artifacts like halo and noise can be mitigated. Moreover, a benchmarking dataset that includes hazy images in multiple conditions is created. Experiment results on the self-created and synthetic datasets show that SID-UNN has outstanding dehazing ability regarding image details and artefacts that outperformed other methods including the traditional Deep Channel Prior (DCP).

## Our Contribution
This section is a reproduction of the results from the research paper.


## Tool Requirements

The following libraries need to be installed to execute the codes:
* python = 3.7
* pytorch = 0.4
* numpy
* matplotlib

The experiments in the previous section were conducted on two 80-core RTX-A6000 GPUs with 128GB memory

## Repository Structure
```
.
├─ README.md
├─ SID.py (Implementations of our contribution to the Dehazing Algorithm)
├─ SID_DCP.py (Implementations of DIP + DCP + guided filtering for reference)
├─ hazy input (Selected images for testing use)
│   └── Real-world Multiple-condition Grading Dataset (Self-created indoor hazy image dataset)
├─ models (Neural network model adopted from DIP)
│   └── Unet_new.py (Unet.py with pad=1)
├─ utils 
│   └── dehaze.py (Physical-prior-related utilities)
│   └── common_utils.py (Utilities)
```