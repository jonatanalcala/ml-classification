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

### References
Perceptron wraps SGDClassifier (parameters & equivalence): [scikit-learn docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html). 
SGDClassifier/Regressor options (losses, penalties, learning-rate schedules, early stopping, sparse input, partial_fit): scikit-learn docs. 
Stochastic Gradient Descent user guide (sparse support, implementation notes): scikit-learn. 
Cython / compiled performance (why scikit-learn is fast): scikit-learn computational performance page; Cython/OpenMP developer notes. 
Cython inner loop location (sgd_fast.pyx) used by SGD estimators: scikit-learn issue & developer discussion. 
safe_sparse_dot utility for efficient dense/sparse multiplication: scikit-learn API docs. 

---

## D.) Decision Boundaries: Comparing Logsitic Regression & SVM


---

## E.) Regularization: Preventing Overfitting in Machine Learning


---

## F.) Impact of the `C` Parameter: Logistic Regression & Linear SVC: 0.01, 1.0, 100.0

