# Federated Learning with Feature Space Separation for Speaker Recognition

### FedFSS
![Alt Text](picture/github.png)

### Training
python train_fed_anchor_lama_gamma.py

### Environment
torch 1.12.1; torchaudio 0.12.1; torchvision 0.13.1  
pip install soundfile, tqdm, matplotlib, Pillow, scikit-learn, numpy

### Result
#### Training on Vox1

| Method | Client1 | Client2 | Client3 | Client4 |
|--------|---------|---------|---------|---------|
| baseline | 6.45 | 6.79 | 6.29 | 6.99 |
| FedFSS | 4.72 | 4.65 | 4.57 | 5.07 |

### Citation

## Reference

```bibtex
@inproceedings{fang2023robust,
  title={Robust Training for Speaker Verification against Noisy Labels},
  author={Fang, Zhihua and He, Liang and Ma, Hanhan and Guo, Xiaochen and Li, Lin},
  booktitle={Proc. INTERSPEECH 2023},
  pages={3192--3196},
  year={2023},
  doi={10.21437/Interspeech.2023-452}
}

## I'm so glad that our sharing was useful to you.


















