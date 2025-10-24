# Federated Learning with Feature Space Separation for Speaker Recognition

#### FedFSS
![Alt Text](picture/github.png)

#### Training
python train_fed_anchor_lama_gamma.py

#### Environment
torch 1.12.1; torchaudio 0.12.1; torchvision 0.13.1  
pip install soundfile, tqdm, matplotlib, Pillow, scikit-learn, numpy

#### Result
EER（%）
| Method | Client1 | Client2 | Client3 | Client4 |
|Trainning on Vox1|---------|---------|
| baseline|6.45 | 6.79 |6.29|6.99|
| FedFSS|4.72| 4.65 |4.57|5.07|








