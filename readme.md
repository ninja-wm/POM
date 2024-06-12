# Code implementation of *"Pretrained Optimization Model for Zero-Shot Black Box Optimization"*

## Introduction
This is an open source implementation of the paper code. Only one version is provided here for research. It is worth mentioning that in order to adjust the performance of POM, the weight file provided here is the one we adjusted and retrained. 

The POM in this repository is not the latest version, nor the best version. This means that this code repository will continue to be updated, and we will continue to launch more powerful POMs.

Welcome to pay attention to this code. If you find any problems, please raise questions. Thank you!

## Requirements
We need to make sure pytorch>1.12. Additionally, install missing packages if they are not available in your environment.

To ensure that the code can run, please run it in an Ubuntu 20.04 environment.
## Installation
### 1 Install BBOB
```bash
cd BBOB_pkg
./install.sh
```

### 2 Install POM
```bash
cd GLHF_pkg
./install.sh
```

## Reproduce the results of BBOB
```bash
chmod +x demo.sh
./demo.sh
```


