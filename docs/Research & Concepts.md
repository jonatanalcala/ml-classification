# Research & Concepts

In this section we will reflect on our results and cover key concepts regarding the models we worked with.

---

## A.) Feature Scaling: Why is important for gradient-based algorithms?

Feature scaling is crucial for gradient-based learners (e.g., Adaline, logistic regression, linear SVM with GD) **because it leads to faster, more stable optimization and balances each feature’s influence** on learning. In short, it prevents the algorithm from “chasing” large-scale features while ignoring smaller-scale ones.

**Key Reasons**
- Faster & More Stable Convergence
  Gradient updates use a single learning rate for all weights. When features are on comparable scales, a single `η` works well, steps are well-sized, and the model converges in fewer epochs (less overshoot/undershoot).
- Prevents Zig-Zagging From Ill-Conditioning
  Unscaled features produce **elongated, elliptical** loss contours → steepest-descent “zig-zags” across narrow valleys. Scaling makes contours more circular, so steps point more directly toward the minimum.
- Equal contribution & Fair Regularization
  Gradients for each weight are proportional to feature magnitude. If one feature has a much larger range, it dominates updates and drowns out others. Regularization isn't biased by raw units. 

When features are unscaled, large-range variables (e.g., capital-gain ≈ $100,000) dominate updates relative to smaller-range ones (e.g., age ≈ 25). This slows learning and makes optimization inefficient, yielding an elongated cost-contour rather than a more symmetric one.

---

## B.) Gradient Descent Variants: Batch vs Stochastic Gradient Descent


--- 

## C.) Scikit-learn vs Book Implementations: Why does scikit-learn outperform book code?


---

## D.) Decision Boundaries: Comparing Logsitic Regression & SVM


---

## E.) Regularization: Preventing Overfitting in Machine Learning


---

## F.) Impact of the `C` Parameter: Logistic Regression & Linear SVC: 0.01, 1.0, 100.0

