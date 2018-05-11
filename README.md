# Adversarial Neural Cryotography (WIP)
Reproducing Andersan et al. (Google Brain)'s "LEARNING TO PROTECT COMMUNICATIONS WITH ADVERSARIAL NEURAL CRYPTOGRAPHY" in Pytorch.
---
This work is an independent project to get handson experience on Adversarial Neural Nets and GANs.

## Requirements
1. Python 3.6+
2. PyTorch

## Usage
1. Run `python main.py --save {--gpu}`. This will create `errors.csv` in `data/`. Essentially this is the base Adversarial Neural CryptoNets[1].

## References
1. Andersan et al. "LEARNING TO PROTECT COMMUNICATIONS WITH ADVERSARIAL NEURAL CRYPTOGRAPHY" International Conference on Machine Learning. 2017.  [(pdf)](https://arxiv.org/pdf/1610.06918.pdf)

## TODO List
- [x] Build Neural Agents based on Fully Connected layers
- [ ] Design and Implement Cryptography Algorithms with Neural Nets.
- [ ] Train the Adversarial agents based on Convolutional Architectures.
- [ ] Reinforcement Learning based Adversarial Examples.
