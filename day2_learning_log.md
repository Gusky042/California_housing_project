## Day 2 - Cost Function for Logistic Regression

### What I Learned:
- Logistic regression uses Log Loss (Binary Cross-Entropy), not MSE
- Cost is high when predictions are confidently wrong
- Cost is low when predictions are confidently correct
- Formula: -1/m * Σ [ y*log(p) + (1-y)*log(1-p) ]

### Code I Wrote:
- logistic_cost() function from scratch
- Visualization showing cost behavior


### Tomorrow's Goal:
- Gradient descent for logistic regression (learning w and b)
