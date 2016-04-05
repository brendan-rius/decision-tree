# Decision trees

Basic implementation of a decision tree classifier.

This is a work in progress and is not intended to be used yet.

## Improvements

 * Find a better way to make continuous features values discrete. For now, the only possible values are the ones in the
 training set, that means if you have a feature `age` going from 0 to 100 but there is no training example with age
 `97`, then the algorithm cannot predict the label when `age` == 97.
 * Use numpy instead of native Python lists.
