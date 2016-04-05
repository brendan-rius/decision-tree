# Decision trees

Basic implementation of a decision tree classifier.


Example of code:

```python
clf = DecisionTreeClassifier()
# Training feature vectors
x = [
    [1, 2, 3, 1, 2, 1],
    [1, 4, 3, 1, 2, 3],
    [5, 1, 3, 3, 2, 5],
    [1, 2, 3, 1, 3, 1],
    [1, 2, 3, 1, 2, 3],
    [1, 2, 5, 1, 5, 1],
    [5, 2, 8, 5, 2, 5],
    [5, 4, 8, 5, 2, 8],
    [5, 5, 8, 8, 2, 5],
    [5, 2, 8, 3, 8, 5],
    [5, 2, 8, 5, 2, 8],
    [5, 2, 5, 5, 5, 5],
]
# Training labels
y = [
    "A",
    "A",
    "A",
    "A",
    "A",
    "A",
    "B",
    "B",
    "B",
    "B",
    "B",
    "B",
]
clf.fit(x, y)  # "Learning" step
print(clf.predict([1, 2, 5, 1, 2, 3]))
```

Output:

```
A
```

## Improvements

 * Find a better way to make continuous features values discrete. For now, the only possible values are the ones in the
 training set, that means if you have a feature `age` going from 0 to 100 but there is no training example with age
 `97`, then the algorithm cannot predict the label when `age` == 97.
 * Use numpy instead of native Python lists.
