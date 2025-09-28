# Research & Concepts

In this section we will reflect on our results and cover key concepts regarding the models we worked with.

---

## A.) Feature Scaling: Why is important for gradient-based algorithms?

Feature scaling is crucial for gradient-based learners (e.g., Adaline, logistic regression, linear SVM with GD) **because it leads to faster, more stable optimization and balances each feature’s influence** on learning. In short, it prevents the algorithm from “chasing” large-scale features while ignoring smaller-scale ones.

**Key Reasons**
- **Faster & More Stable Convergence**
  - Gradient updates use a single learning rate for all weights. When features are on comparable scales, a single `η` works well, steps are well-sized, and the model converges in fewer epochs (less overshoot/undershoot).
- **Prevents Zig-Zagging From ill-Conditioning**
  - Unscaled features produce **elongated, elliptical** loss contours → steepest-descent “zig-zags” across narrow valleys. Scaling makes contours more circular, so steps point more directly toward the minimum.
- **Equal contribution & Fair Regularization**
  - Gradients for each weight are proportional to feature magnitude. If one feature has a much larger range, it dominates updates and drowns out others. Regularization isn't biased by raw units. 

When features are unscaled, large-range variables (e.g., capital-gain ≈ $100,000) dominate updates relative to smaller-range ones (e.g., age ≈ 25). This slows learning and makes optimization inefficient, yielding an elongated cost-contour rather than a more symmetric one.

---

## B.) Gradient Descent Variants: Batch vs Stochastic Gradient Descent
- **Batch GD (BGD)**: computes the gradient on all N samples each step → smooth, stable updates but expensive per step. 
- **Stochastic GD (SGD)**: updates using one sample at a time → noisy path, cheap per update, scales to streaming data; often better generalization with a good schedule.
- **Mini-batch GD (MBGD)**: computes on a small batch (e.g., 32–1024) → practical middle ground and the default in most libraries.

Batch GD computes the gradient over the entire dataset before taking a step, so the loss curve is smooth and steps are stable—but each step is expensive. Stochastic GD updates after each single example, so updates are cheap and frequent, but the path is noisy. Mini-batch sits in between and is what most libraries use. In our Adult-Income results you can see the contrast: Adaline-GD has a clean, monotonic MSE decline, while Adaline-SGD is noisier yet still converges and slightly improved recall on the minority class. Use batch when the data are small and you want stability; use SGD/mini-batch for large or streaming data and when you want faster iterations and often better generalization.

An important note is that you need to **shuffle** data each epoch for SGD/mini-batch and use **feature scaling** to improve conditioning and speed. 

--- 

## C.) Scikit-learn vs Book Implementations: Why does scikit-learn outperform book code?

The reason scikit-learn’s versions are faster mainly because the hot path is written in Cython. Cython is a superset of Python that lets you add static types and compiles your code to C, so loops and dot-products run as compiled code instead of through the Python interpreter. sklearn uses this for the SGD inner loop (`sgd_fast.pyx`) and can also tap **BLAS** (or Basic Linear Algebra Subprograms) and even **OpenMP** (API for shared-memory parallel programming) for parallelism.

On top of raw speed, sklearn’s SGD engine gives you learning-rate schedules, L1/L2/ElasticNet, shuffling, early stopping, class weights, and even weight averaging—all of which make training faster and more stable. 

By contrast, our scratch Perceptron iterates in pure Python, paying per-iteration overhead, and our `AdalineGD` does full-batch updates, which are costlier per epoch. That’s why, on the Adult-Income data, the sklearn Perceptron/‘Adaline’ reached similar or better accuracy faster, with fewer numerical issues and less tuning. 

The takeaway: for production-style speed and stability, we prefer sklearn’s compiled SGD stack; and we use the scratch code for learning and transparency.

### References
- **Perceptron wraps SGDClassifier** (parameters & equivalence): [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html). 
- **SGDClassifier/Regressor options** (losses, penalties, learning-rate schedules, early stopping, sparse input, partial_fit): [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html). 
- **Stochastic Gradient Descent user guide** (sparse support, implementation notes): [scikit-learn](https://scikit-learn.org/stable/modules/sgd.html?utm_source=chatgpt.com).
- **Cython / compiled performance** (why scikit-learn is fast): scikit-learn computational performance page; [Cython/OpenMP developer notes](https://scikit-learn.org/stable/computing/computational_performance.html?utm_source=chatgpt.com). 
- **Cython inner loop location (`sgd_fast.pyx`) used by SGD estimators**: [scikit-learn issue & developer discussion](https://github.com/scikit-learn/scikit-learn/issues/15123?utm_source=chatgpt.com). 
- `safe_sparse_dot` utility for efficient dense/sparse multiplication: [scikit-learn API docs](https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.safe_sparse_dot.html?utm_source=chatgpt.com). 

---

## D.) Decision Boundaries: Comparing Logsitic Regression & SVM

Understanding decision boundaries is key in evaluating how different models classify data. This reflection compares **Logistic Regression** and **Support Vector Machines (SVM)**, highlighting their strengths, limitations, and the implications of their decision boundaries on classification tasks.  

### Logistic Regression
- **Linear decision boundaries only:**  
  Logistic regression models separate classes using a straight line (in 2D) or a hyperplane (in higher dimensions).  
- **Limitation with overlapping data:**  
  In datasets where the classes overlap, logistic regression struggles. This is evident in the visual representation: the red and blue points cannot be cleanly separated by a single line.  
- **Interpretability:**  
  Logistic regression is often preferred in scenarios where model transparency is important, as coefficients can be directly interpreted as feature importance.  
- **When it works best:**  
  Performs well when data is approximately linearly separable and when simplicity and interpretability outweigh the need for capturing complex patterns.  

### Support Vector Machine (SVM)
- **Kernel flexibility (RBF in this case):**  
  Unlike logistic regression, SVM can use kernel functions (e.g., radial basis function) to project the data into higher-dimensional space. This enables it to draw nonlinear boundaries that better capture the true structure of the data.  
- **Nonlinearity:**  
  The decision boundary in the SVM example adapts to the data’s shape, allowing it to create curved or irregular separation regions that logistic regression cannot achieve.  
- **Handling overlap:**  
  SVM does not force a straight-line separation. Instead, it maximizes the margin between classes while tolerating misclassifications in overlapping regions. This produces broader, more adaptive decision boundaries (e.g., the “broader tail” effect seen in the figure).  
- **Trade-offs:**  
  While powerful, SVMs are more computationally expensive and less interpretable compared to logistic regression.  

### Comparative Insights
- **Flexibility:**  
  Logistic regression is constrained to linear boundaries, whereas SVM can model nonlinear relationships via kernels.  
- **Data Fit:**  
  Logistic regression struggles when classes overlap significantly, but SVM adapts better by expanding the feature space.  
- **Complexity vs. Interpretability:**  
  Logistic regression is lightweight and interpretable, but limited. SVM is more flexible and accurate for complex data, but less interpretable.  

### Key Takeaways
1. Logistic regression demonstrates the limitations of linear models when data is not perfectly separable.  
2. SVM, especially with RBF kernels, shows how nonlinear approaches can better capture patterns and improve classification performance.  
3. The choice between Logistic Regression and SVM depends on the problem:  
   - Choose **Logistic Regression** for simpler, linearly separable data or when interpretability is essential.  
   - Choose **SVM** when the data is complex, nonlinear, or has overlapping classes and higher accuracy is required.  

---

## E.) Regularization: Preventing Overfitting in Machine Learning


---

## F.) Impact of the `C` Parameter: Logistic Regression & Linear SVC: 0.01, 1.0, 100.0

