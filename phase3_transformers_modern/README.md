# 🧠 PHASE 3 — Transformers & Modern Architectures

**Goal:** Build the foundation of modern LLMs and multimodal models.
**Length:** 8–12 weeks (heavy; expect most effort here)

> Key paper: *Attention Is All You Need* (Transformer) — foundational. Read & implement the transformer block from scratch, then scale to BERT / GPT-style tasks. ([arXiv][1])
> Canonical explainer: *The Illustrated Transformer* (Jay Alammar) — a must-read for intuition & visuals. ([Jay Alammar][2])

## 1. Transformer (Attention Is All You Need)

* **Implement:** scaled dot-product attention, positional encodings, multi-head attention, feed-forward, layer-norm, residuals.
* **Deliverable:** single-layer transformer block in NumPy + PyTorch; attention visualizer on sample text.

## 2. BERT Mini (Masked Language Modeling)

* **Implement:** masked language modeling objective, tokenization (BPE / WordPiece), pre-training loop for MLM on a small corpus.
* **Concepts:** next-sentence prediction (historically), fine-tuning for classification tasks.
* **Tools / Alts:** Hugging Face Transformers (BERT implementations), tokenizers library.
* **Deliverable:** fine-tune on a small classification task (e.g., SST-2 small).

## 3. GPT Mini (Autoregressive LM)

* **Implement:** causal self-attention, autoregressive decoding loop, sampling strategies (greedy, top-k, nucleus).
* **Tools / Alts:** nanoGPT, minGPT tutorials; Hugging Face GPT-2 small for comparisons.
* **Deliverable:** small GPT trained on tiny corpus; interactive text-generation demo.

## 4. T5 / BART (Encoder–Decoder seq2seq)

* **Implement:** encoder-decoder attention, task prefixing (T5), denoising objectives (BART).
* **Tools / Alts:** Hugging Face sequence-to-sequence models; Fairseq.
* **Deliverable:** small seq2seq model for summarization or translation toy tasks.

## 5. Vision Transformers (ViT)

* **Implement:** patch embedding, positional encodings for patches, transformer stack for images.
* **Tools / Alts:** timm (PyTorch Image Models), huggingface/vision-transformers.
* **Deliverable:** ViT on CIFAR-10 or tiny image dataset; analyze learned attention patterns.

## 6. CLIP (Text–Image Joint Embedding)

* **Implement:** contrastive image-text pretraining (InfoNCE), separate encoders for image & text, cosine similarity objective.
* **Tools / Alts:** OpenCLIP, Hugging Face CLIP models.
* **Deliverable:** retrieval demo (image → text or text → image) with a small paired dataset.

## 7. Whisper Mini (Speech Recognition Transformer)

* **Implement / Explore:** transformer-based ASR: spectrogram input, encoder-decoder LM.
* **Tools / Alts:** OpenAI Whisper repo, wav2vec 2.0 as alternative.
* **Deliverable:** run small ASR pipeline on toy audio; compare with Whisper pretrained.

## 8. Diffusion Models (Image Generation via Denoising)

* **Implement:** DDPM forward/backward noising, UNet denoiser, sampling loop.
* **Papers / Explainers:** DDPM papers and high-level tutorials.
* **Tools / Alts:** guided-diffusion implementations, Stable Diffusion codebases.
* **Deliverable:** tiny diffusion model generating low-res images; visualize denoising steps.

## Learning Objectives
- Master transformer architecture and attention mechanisms
- Understand modern language model training objectives
- Implement multimodal models (vision-language, speech)
- Learn generative modeling with diffusion models
- Apply transformers to various domains and modalities

## Nigerian Context
- **Language:** Yoruba language models, multilingual transformers for Nigerian languages
- **Healthcare:** Medical image analysis with vision transformers
- **Education:** Automated essay grading and educational content generation
- **Agriculture:** Satellite imagery analysis for crop disease detection
- **Speech:** Voice recognition systems for Nigerian languages and accents

## Assessment Structure
- **Architecture Implementations:** 8 transformer-based model implementations
- **Training Objectives:** MLM, autoregressive, contrastive learning
- **Multimodal Projects:** Vision-language and speech recognition demos
- **Generative Tasks:** Text generation, image generation, speech synthesis
- **Analysis:** Attention visualization and model interpretability

## Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer paper)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) (Jay Alammar)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [nanoGPT](https://github.com/karpathy/nanoGPT) (Andrej Karpathy)
- [minGPT](https://github.com/karpathy/minGPT) (Andrej Karpathy)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)

[1]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[2]: https://jalammar.github.io/illustrated-transformer/ "The Illustrated Transformer - Jay Alammar"
