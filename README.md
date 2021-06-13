# zombies
Simple bootstrapping.

## Usage

Simply call the `confidence_interval` function, which takes a function and dataset as inputs.

The rules are:

1. The data set must be 1-D or greater.
2. The function must produce a scalar or 1-D output.

```python
import numpy as np
import pandas as pd
from zombies import confidence_interval

iris = pd.read_csv(
    "https://raw.githubusercontent.com/"
    "mwaskom/seaborn-data/master/iris.csv"
    )
ci = confidence_interval(silent=True)(np.mean, iris)
print(ci)

#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.712617     2.987397      3.465284     1.071816
# 1      5.960700     3.116750      3.985333     1.300000
```

Bootstrapping is stochastic, so repeating this outright will give slighly different values:

```python
ci2 = confidence_interval(silent=True)(np.mean, iris)
print(ci2)

#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.706378        2.984      3.478191     1.078486
# 1      5.982794        3.132      4.039455     1.319758
```

You can make the bootstrapping reproducible if you can use a seed.

```python
ci3 = confidence_interval(silent=True, seed=42069)(np.mean, iris)
ci4 = confidence_interval(silent=True, seed=42069)(np.mean, iris)
print(ci3)
print()
print(ci4)

#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.723856     2.990000      3.485461     1.079327
# 1      5.950049     3.114633      3.993417     1.294886
# 
#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.723856     2.990000      3.485461     1.079327
# 1      5.950049     3.114633      3.993417     1.294886
```
