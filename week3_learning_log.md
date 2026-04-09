# Learning Log - Andrew Ng Week 3

## Day 1 - April 9, 2026

### What I Learned Today:
- Logistic regression predicts probabilities (0 to 1)
- Sigmoid function converts any number to probability
- Formula: g(z) = 1 / (1 + e^(-z))

### Code I Wrote:
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
