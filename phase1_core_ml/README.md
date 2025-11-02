# 🧩 PHASE 1 — Core Machine Learning Foundations

**Goal:** Build intuition for optimization, generalization, and algorithmic design.
**Length:** 8–12 weeks (suggested: 1.5–2 weeks per major topic; adjust for microbiology→CS transition)
**Assessment Checkpoints:** End-of-phase quiz covering bias-variance tradeoff derivation, gradient descent convergence analysis, and XGBoost vs Random Forest trade-offs

## 1. Linear Regression (Batch + SGD)

* **What you'll implement:** Ordinary least squares closed-form; gradient descent; stochastic & mini-batch SGD; MSE, R².
* **Math / Concept Focus:** derivation of normal equations, gradients, learning rate selection, convergence diagnostics.
* **Papers / Explainers:** classic stats texts (OLS derivation) + blog explainers (e.g., Andrew Ng notes).
* **Tools / Libs / Alts:** NumPy (from-scratch), scikit-learn (LinearRegression, SGDRegressor), statsmodels for inference.
* **Exercises / Deliverables:** notebook implementing OLS & SGD, plots of loss vs steps, compare to scikit-learn results.

* **Online Resources:** Coursera Machine Learning by Andrew Ng ([coursera.org/learn/machine-learning](https://www.coursera.org/learn/machine-learning)), 3Blue1Brown Linear Algebra series ([3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra))

## 2. Logistic Regression (Binary & Multiclass)

* **Implement:** Binary logistic regression via gradient descent, softmax multiclass, log-loss, regularization.
* **Math / Concept Focus:** cross-entropy loss, probability interpretation, calibration.
* **Tools:** NumPy, scikit-learn (LogisticRegression).
* **Deliverable:** classifier on a simple dataset (e.g., Iris, binary subset), ROC curve, decision boundary visualizations.

* **Online Resources:**
  - GeeksforGeeks: [Implementation of Logistic Regression from Scratch](https://www.geeksforgeeks.org/implementation-of-logistic-regression-from-scratch-using-python/)
  - RealPython: [Logistic Regression in Python](https://realpython.com/logistic-regression-python/)
  - YouTube: [Logistic Regression From Scratch in Python (Mathematical)](https://www.youtube.com/watch?v=dMuH9qjTXNY)
  - Machine Learning Mastery: [How To Implement Logistic Regression From Scratch](https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/)

## 3. Decision Trees & Random Forests

* **Implement:** CART splits (Gini, entropy), tree building, pruning heuristics, bootstrap aggregation (bagging).
* **Concepts:** bias-variance tradeoff, overfitting, feature importance.
* **Papers / Explainers:** Breiman's Random Forests papers and standard ML textbooks.
* **Tools / Alts:** scikit-learn (DecisionTreeClassifier, RandomForest), XGBoost / LightGBM (for gradient boosting alternatives).
* **Deliverable:** implement split criterion + a simple random forest ensemble wrapper; compare with scikit-learn.

* **Online Resources:**
  - GeeksforGeeks: [Decision Tree Implementation](https://www.geeksforgeeks.org/decision-tree-implementation-python/)
  - Towards Data Science: [Building Decision Tree Algorithm from Scratch](https://towardsdatascience.com/decision-tree-algorithm-in-python-from-scratch-8c43f445b2ee)
  - YouTube: StatQuest [Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
  - Machine Learning Mastery: [How to Build Random Forest Tutorial](https://machinelearningmastery.com/implement-random-forest-scratch-python/)

## 4. KNN, K-Means, Hierarchical Clustering

* **Implement:** KNN classifier, K-means (Lloyd's), hierarchical agglomerative clustering.
* **Concepts:** metrics, elbow method, silhouette score, linkage choices.
* **Tools / Alts:** scikit-learn, scipy.cluster, faiss for large-scale neighbors (see Phase 4).
* **Deliverable:** cluster analysis on a toy dataset; cluster visualization & evaluation.

* **Online Resources:**
  - GeeksforGeeks: [KNN Algorithm Implementation](https://www.geeksforgeeks.org/k-nearest-neighbors-with-python-ml/)
  - YouTube: StatQuest [K-nearest neighbors](https://www.youtube.com/watch?v=HVXime0nQeI)
  - Towards Data Science: [K-Means Clustering from Scratch](https://towardsdatascience.com/k-means-clustering-from-scratch-6a9d19cafc25)
  - Andrew Ng: [Unsupervised Learning](https://www.youtube.com/watch?v=hClHNWz42dw)

## 5. Naive Bayes & SVM

* **Implement:** Gaussian / Multinomial Naive Bayes; linear SVM via primal gradient descent (or hinge loss).
* **Concepts:** generative vs discriminative models, kernel trick overview.
* **Tools / Alts:** scikit-learn (GaussianNB, MultinomialNB, SVC), LIBSVM.
* **Deliverable:** text classification baseline (Naive Bayes) and compare with SVM.

## 6. PCA & Dimensionality Reduction

* **Implement:** PCA via SVD, eigen-decomposition, explained variance. t-SNE / UMAP overview.
* **Tools / Alts:** scikit-learn PCA, sklearn.manifold.TSNE, umap-learn.
* **Deliverable:** visualization of high-dimensional datasets (MNIST / word embeddings).

## 7. Regularization: L1, L2, Dropout

* **Implement:** ridge and lasso (analytic + gradient solutions), dropout from scratch in a small neural net.
* **Concepts:** sparsity, model complexity control, under/overfitting tradeoffs.
* **Tools / Alts:** scikit-learn (Ridge, Lasso), PyTorch / TensorFlow for dropout implementations.
* **Deliverable:** ablation study showing effects of different regularizers.

## 8. Gradient Descent Variants: Momentum, Adam, RMSProp

* **Implement:** vanilla GD, momentum, Nesterov, RMSProp, Adam — track updates step-by-step.
* **Explainers:** read Adam paper and practical notes on convergence/hyperparameters.
* **Tools / Alts:** use PyTorch/TF optimizers for comparison.
* **Deliverable:** train a small NN with each optimizer and compare speed & stability.

## Learning Objectives
- Master mathematical foundations of core ML algorithms
- Understand optimization techniques and convergence properties
- Implement algorithms from scratch using NumPy
- Compare custom implementations with production libraries
- Analyze bias-variance tradeoffs and regularization techniques

## Nigerian Context
- **Healthcare:** Disease prediction using logistic regression on patient data
- **Agriculture:** Crop yield prediction with linear regression models
- **Education:** Student performance analysis using decision trees
- **Finance:** Credit scoring with ensemble methods

## Assessment Structure
- **Weekly Notebooks:** 8 comprehensive implementations with detailed explanations
- **Mathematical Derivations:** Understanding of gradients, loss functions, optimization
- **Comparative Analysis:** Performance comparison with scikit-learn implementations
- **Final Quiz:** Theoretical understanding of bias-variance, convergence, trade-offs

## Resources
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (Bishop)
- [Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/) (Hastie et al.)
- [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow et al.)
- [Coursera Machine Learning](https://www.coursera.org/learn/machine-learning) (Andrew Ng)
