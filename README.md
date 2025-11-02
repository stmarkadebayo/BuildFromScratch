# Build From Scrath #ScratchToProd — Detailed AI/ML Curriculum

A comprehensive learning path to master artificial intelligence and machine learning by building everything from scratch. This curriculum covers machine learning fundamentals through advanced AI systems, emphasizing understanding over APIs.

## Overview

This repository contains a structured AI/ML curriculum divided into **9 phases** plus **2 interludes**:

### Core Phases
1. **🧩 Phase 1 — Core Machine Learning Foundations** - Linear/Logistic Regression, Decision Trees, KNN/K-Means, Naive Bayes/SVM, PCA, Regularization, Gradient Variants
2. **🏛️ Phase 2 — Deep Learning Core** - Feedforward NNs, CNNs, RNNs/LSTMs, Autoencoders, GANs, Attention Mechanisms
3. **🧠 Phase 3 — Transformers & Modern Architectures** - Transformer, BERT mini, GPT mini, ViT, CLIP, Whisper, Diffusion Models
4. **🧩 Phase 4 — Retrieval, Reasoning & Grounded AI** - RAG, Vector Databases, OCR, Multimodal RAG
5. **⚙️ Phase 5 — Scalable & Efficient Models** - MoE, LoRA, Quantization, Pruning, Speculative Decoding
6. **🤖 Phase 6 — Agents, Reasoning & Self-Improving Systems** - RL, PPO, RLHF, Tool-Using Agents
7. **🧪 Phase 7 — AI Systems Engineering & Deployment** - MLOps, Experiment Tracking, Model Serving
8. **🧬 Phase 8 — The 2025 AI Frontier** - Advanced innovations and ongoing projects
9. **🛡️ Phase 9 — Responsible AI, Interpretability & Production** - Ethics, Bias Detection, Interpretability

### Interludes
- **🏗️ DSA Fundamentals** - Graph Algorithms, Dynamic Programming, String Algorithms, Advanced Trees
- **🏛️ System Design** - ML Platform Design, Real-time Feature Engineering, Multi-tenant Systems

## Key Features

- ✅ **From Scratch Implementation** - NumPy-first approaches before frameworks
- 📚 **Academic Rigor** - Paper-to-code methodology with proper citations
- 🚀 **Production Ready** - Includes deployment, monitoring, and best practices
- 🏗️ **Scalable Structure** - Monorepo with clean organization per phase
- 🔧 **Modern Tooling** - GitHub Actions CI/CD, pre-commit hooks, type checking
- 🌍 **African Context** - Nigerian applications in healthcare, agriculture, education

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd build-from-scratch

# Install dependencies
pip install -e ".[dev,ml]"

# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files
pre-commit run --all-files

# Setup environment for development
pip install -e ".[serve]"  # For demo deployments
```

## Project Structure

```
.
├── interlude_dsa_fundamentals/     # DSA for ML engineers
├── interlude_system_design/        # ML platform design patterns
├── phase1_core_ml/                 # ML fundamentals (8 topics)
├── phase2_deep_learning/           # Neural networks & architectures
├── phase3_transformers_modern/     # Transformers & modern AI
├── phase4_retrieval_grounded_ai/   # Retrieval & multimodal
├── phase5_scalable_efficient/      # Efficient model techniques
├── phase6_agents_reasoning/        # Agent systems & reasoning
├── phase7_systems_deployment/      # MLOps & production
├── phase8_ai_frontier/             # Advanced/ongoing projects
├── phase9_responsible_ai/          # Ethics & interpretability
├── docs/                           # Documentation
├── tests/                          # Unit tests
├── demos/                          # Deployment demos
└── src/                            # Shared utilities
```

Each phase/interlude contains:
- `notebooks/` - Jupyter notebooks with implementations
- `src/` - Python source code
- `demos/` - Deployed demos (Streamlit/Gradio)
- `papers/` - Key paper references

## Learning Path

1. **Start Simple** - Begin with NumPy implementations in Phase 1 & 2
2. **Build Complexity** - Progress through architectures and algorithms
3. **Apply Practically** - Use frameworks (PyTorch, Hugging Face) for scale
4. **Ship Products** - Deploy working AI applications
5. **Think Ethically** - Incorporate responsible AI practices

## Prerequisites

- Python 3.8+
- Basic mathematical understanding (linear algebra, calculus)
- Familiarity with programming concepts
- Optional: Experience with Jupyter notebooks

## Resources

- [Full Syllabus](syllabus.md) - Detailed learning plan with resources
- [Progress Tracking](myprogress.md) - Personal progress log
- [Jay Alammar's Transformer Blog](https://jalammar.github.io/illustrated-transformer/)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)

## Contributing

This is an educational curriculum. Contributions welcome:
- Implementation improvements
- Additional resources
- Documentation enhancements
- New phase content

## License

MIT License - Free for educational use.

## Acknowledgments

Inspired by the AI/ML community's emphasis on building understanding through implementation. Grateful to the open-source community for models, datasets, and educational resources.
