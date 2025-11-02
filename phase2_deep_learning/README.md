# 🏛️ PHASE 2 — Deep Learning Core

**Goal:** Learn how neural nets represent & optimize complex functions.
**Length:** 6–8 weeks

## 1. Feedforward Neural Net (from-scratch with NumPy)

* **Implement:** dense layers, activations (ReLU, sigmoid, tanh), forward/backward pass, mini-batch SGD.
* **Concepts:** weight init, vanishing/exploding gradients, normalization basics.
* **Tools / Alts:** pure NumPy; then PyTorch for scale.
* **Deliverable:** train a 2-layer net on MNIST subset; compare to PyTorch implementation.

* **Online Resources:** Neural Networks and Deep Learning by Michael Nielsen ([neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com/))

## 2. Backpropagation Visualization & Implementation

* **Implement:** compute gradients manually for simple networks; visualize gradient flow and gradients per-layer.
* **Explain:** chain rule intuitions; autograd vs manual.
* **Deliverable:** notebook with visualizations (gradient norms, gradient histograms).

## 3. CNNs (Conv, Pooling, Padding — manual)

* **Implement:** convolution operation, stride, padding, max/avg pooling, forward/backprop (NumPy).
* **Concepts:** receptive field, parameter sharing, common architectures (LeNet, VGG, ResNet overview).
* **Tools / Alts:** PyTorch / Keras; torchvision for datasets.
* **Deliverable:** implement conv layer, build tiny CNN, train on CIFAR-10 subset.

## 4. RNN, GRU, LSTM (Sequential Models)

* **Implement:** vanilla RNN, GRU, and LSTM cell implementations (forward + backward).
* **Concepts:** gating mechanisms, truncation through time, teacher forcing.
* **Tools / Alts:** PyTorch nn.RNN/LSTM/GRU.
* **Deliverable:** small char-level language model; next-character generation.

## 5. Autoencoders + Variational Autoencoders (VAE)

* **Implement:** plain autoencoder, VAE reparameterization trick, ELBO.
* **Concepts:** latent variable modeling, sampling, reconstruction vs regularization.
* **Tools / Alts:** PyTorch implementations; TensorFlow/Keras alternatives.
* **Deliverable:** train VAE on MNIST and visualize latent traversals.

## 6. GANs (Generator vs Discriminator Dynamics)

* **Implement:** vanilla GAN training loop, stabilizing tricks (label smoothing, gradient penalty).
* **Concepts:** min-max optimization, mode collapse, evaluation metrics (FID, IS).
* **Tools / Alts:** PyTorch, TensorFlow GAN libs.
* **Deliverable:** train on simple image dataset (e.g., CelebA-small or MNIST variants).

## 7. Attention Mechanisms (Scaled Dot-Product Attention)

* **Implement:** single-head scaled dot-product attention; visualize attention maps.
* **Explainers:** Distill.pub style attention visual guides and Jay Alammar's work (see Phase 3).
* **Tools / Alts:** PyTorch implementations; compare to torch.nn.MultiheadAttention.
* **Deliverable:** attention visualization demo on toy sequences.

## Learning Objectives
- Understand neural network architectures and training dynamics
- Master backpropagation and gradient flow
- Implement convolutional and recurrent neural networks from scratch
- Learn generative modeling techniques (autoencoders, GANs)
- Visualize and interpret neural network behavior

## Nigerian Context
- **Healthcare:** Medical image analysis with CNNs for disease detection
- **Agriculture:** Satellite image processing for crop monitoring
- **Education:** Handwriting recognition with RNNs for OAU student assessments
- **Language:** Yoruba language modeling and text generation

## Assessment Structure
- **Implementation Notebooks:** 7 comprehensive neural network implementations
- **Visualization Tasks:** Gradient flow, attention maps, latent space traversals
- **Comparative Analysis:** NumPy vs PyTorch implementations
- **Generative Projects:** VAE and GAN training with evaluation metrics

## Resources
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) (Michael Nielsen)
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow et al.)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) (Stanford)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Distill.pub](https://distill.pub/) (Interactive ML explanations)
