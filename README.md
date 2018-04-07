# nets

Testing ground for big ones, small ones, recurrent and even convolutional ones for language modelling. Everything can be trivially extended to seq2seq models.

-----------------------------
## Instructions
```
# Check command line arguments
$ python3 train.py -h
# Run
$ python3 train.py -opt momentum --name my_network
```

The network architecture is kept modular. To swap out the network for your custom one, create a `@staticmethod` under the `Network` class in `network.py`:

```python
@staticmethod
def my_network(x, config, training, **kwargs):
    """
    Inputs:
    x: example data
    config: class defining hyperparameter values
    training: Placeholder boolean tensor to distinguish between training/prediction

    Returns:
    network logits
    """

    # To prevent overfitting, we don't even look at the inputs!
    return tf.random_uniform(x.shape[0], minval=0, maxval=config.n_classes, dtype=tf.int32, seed=42)
```
Now open model.py and edit the first line under the Model init:
```python
class Model():
    def __init__(self, **kwargs):

        arch = Network.my_network
        # Define the rest of the computational graph
```

## Dependencies
* Python 3.6
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.7](https://www.tensorflow.org/)
