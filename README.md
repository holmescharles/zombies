# zombies
Simple bootstrapping.

## Usage

Simply make a bootstrapper object and call the `conf_int` function, which takes a function and dataset as inputs.

The rules are:

1. The data set must be 1-D or greater.
2. The function must produce a scalar or 1-D output.

```python
import numpy as np
import pandas as pd
import zombies as zb

iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
bs = zb.Bootstrapper()

ci = bs.conf_int(np.mean, iris)

# Output:
#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.720377     2.983206      3.484000     1.080667
# 1      5.976667     3.124667      4.032289     1.313751
```

Bootstrapping is stochastic, so repeating this outright will give slighly different values:

```python
ci2 = bs.conf_int(np.mean, iris)

#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.706378        2.984      3.478191     1.078486
# 1      5.982794        3.132      4.039455     1.319758
```

You can make the bootstrapping reproducible if you initiate the bootstrapper with a seed.

```python
bs2 = zb.Bootstrapper(seed=5555)

ci3 = bs2.conf_int(np.mean, iris)
#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.715333     2.987333      3.492056     1.084667
# 1      5.964199     3.129927      4.031421     1.319473

ci4 = bs2.conf_int(np.mean, iris)
#    sepal_length  sepal_width  petal_length  petal_width
# 0      5.715333     2.987333      3.492056     1.084667
# 1      5.964199     3.129927      4.031421     1.319473
```

Sometimes you may want to bootstrap two different data sets where the data indexes are the same, but you want to make sure bootstrap resampling is identical across data sets. E.g., lets say you had two data sets, one for sepal data and one for petal data. You could do:

```python

iris_sepal = iris[['sepal_length', 'sepal_width']]
iris_petal = iris[['petal_length', 'petal_width']]

bs3 = zb.Bootstrapper(seed=5555, strict=True)

ci_sepal = bs3.conf_int(np.mean, iris_sepal)
# Output:
#    sepal_length  sepal_width
# 0      5.715835     2.987607
# 1      5.964634     3.130667

ci_petal = bs3.conf_int(np.mean, iris_petal)
#    petal_length  petal_width
# 0      3.492056     1.084723
# 1      4.031421     1.319807
```

If you were to try to use a dataset with a different number of data, then an error would be thrown.
