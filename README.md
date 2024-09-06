# AppleDetector

Creating a new conda environment and installing the required packages:

```shell
conda create -n detect python=3.9 -y
conda activate detect
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install jupyter -y
```


Installing the fastsam:

```shell
git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
```

```shell
cd FastSAM
pip install -r requirements.txt
```

Installing the CLIP:

```shell
pip install git+https://github.com/openai/CLIP.git