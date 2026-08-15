# Machine Learning Notes

## Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to produce a stronger prediction. Each tree is trained on a random subset of the data and features.

Random Forest reduces variance by averaging predictions across trees, which generally improves generalization and helps reduce overfitting compared to a single deep decision tree.

Key properties:
- Uses bootstrap aggregating (bagging)
- Random feature selection at each split
- Robust to outliers and high-dimensional data

## Overfitting

Overfitting occurs when a model learns noise in the training data rather than the underlying pattern. Signs include high training accuracy but poor validation performance.

Techniques to reduce overfitting:
1. Cross-validation
2. Regularization
3. Ensemble methods like Random Forest
4. More training data
5. Feature selection

## Ensemble Methods

Ensemble methods combine multiple models to improve predictive performance. Common approaches include bagging, boosting, and stacking.

Bagging (Bootstrap Aggregating) trains models on random subsets and averages predictions. Random Forest is a bagging method applied to decision trees.

Boosting trains models sequentially, each correcting errors of the previous model. Gradient Boosting and AdaBoost are popular boosting algorithms.

## Decision Trees

A decision tree splits data based on feature values to make predictions. Trees are interpretable but prone to overfitting when grown too deep.

Pruning, maximum depth limits, and minimum sample requirements help control tree complexity.
