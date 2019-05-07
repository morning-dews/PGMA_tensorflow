# Parameter Generation and Model Adaptation(PGMA)

This is an implementation of [Overcoming Catastrophic Forgetting for Continual Learning via Model Adaptation(Hu et al.,2018)](https://openreview.net/forum?id=ryGvcoA5YX) in disjoint MNIST setting.

#### Requirements
- Python 3.5
- tensorflow 1.4

#### Explanations

`wae.py` is based on the implementation of WAE. It defines the network structure and training procedure of our model.

`configs.py` defines the configurations in our model.

`setdata.py` loads MNIST data.

`run.py` is the main script to train and test our model.

`ops.py` and `models.py` define operations and functions utilized in `wae.py`.
