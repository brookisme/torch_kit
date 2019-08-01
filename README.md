### Torch Kit: pytorch utilities


TorchKit provides a Pytorch training loop with callbacks, custom loss functions and metrics, along with various helper utilities. 

- [example](#example): A quick example illustrating usage
- [train](#train): Trainer class for training pytorch-models
- [callbacks](#callbacks): ...
- [helpers](#helpers): ...
- [metrics](#metrics): ...
 

---

<a name='install'></a>
##### INSTALL

```bash
git clone https://github.com/brookisme/torch_kit.git
cd torch_kit
pip install -e .
```

---

<a name='example'></a>
##### Example 

Perhaps the best way to get started to walk through a simple example. 

In the scripts below the following variables have already been set:

- `NNModel`: A model class such that `model=NNModel(**model_config)` creates an instance of a pytorch model
- `model_config`: A kwarg dictionary for NNModel
- `train_dl`/`valid_dl`: Dataloaders for training/validation
- `criterion`/`optimizer`: pytorch criterion (loss function) / optimizer

First we'll use the helpers module to get the device and set up our model

```python

...

import torch_kit.train as train
import torch_kit.helpers as H

# get device
DEVICE=H.get_device()

# initialize model
model=H.get_model(
    net=NNModel,
    config=model_config,
    device=DEVICE )

""" or """

# initialize model with weight initializer 
model=H.get_model(
    net=NNModel,
    config=model_config,
    device=DEVICE,
    weight_initializer=weights_init)

""" or """

# initialize model with saved weights
WEIGHTS_PATH='path/to/weights/nnmodel_weights.p'
model=H.get_model(
    net=NNModel,
    config=model_config,
    device=DEVICE,
    init_weights=WEIGHTS_PATH)
```

Now we create a instance of `train.Trainer`. Here we are passing `name='example'` so that our weights will be saved to `weights/example.<TIME_STAMP>` and `weights/example.best.<TIME_STAMP>`.

```python
trainer=train.Trainer(
    model,
    name='example',
    criterion=criterion,
    optimizer=optimizer)
```

Now we set the callbacks. By default the only callback is the History callback which is used to print to the screen during training. `kwargs` for the History callback can be passed directly to the `.set_callbacks` method.

Here:
    * `save=True`: saves a pickled dictionary of the training/validation accuracy/loss history
    * `name='history_example'`: sets the name of the history-callback to 'history_example' and in turn, the path of the pickled file to `history/history_example.<TIME_STAMP>`
    * `noise_reducer=2`: reduces the number of lines printed to the screen. In this case every other epoch will be printed the screen.

```python
trainer.set_callbacks(
    save=True,
    name='history_example',
    noise_reducer=2 )
```

Finally its time to train our model.  As mentioned above `train_dl`/`valid_dl` are our dataloaders, and:
    * `nb_epochs`: the max number of epochs to be run. since we are using `early_stopping` the nb_epochs might be less
    * `early_stopping=True`: stop after the validation loss did not improve for `patience` number of epochs
    * `patience=2`: the number of epochs to wait before stopping

```python
trainer.fit(
    train_dl,
    valid_loader=valid_dl,
    nb_epochs=50,
    early_stopping=True,
    patience=2 )

""" output
---------------------------------------------------------------------------
  epoch      batch    | batch_loss    loss    | batch_acc     acc    
---------------------------------------------------------------------------
    1          84     |  0.11298    0.39952   |  0.23185    0.28084  
    ... 
    10         84     |  0.10835    0.09091   |  0.74068    0.79894  
    -          24     |  0.18172    0.13673   |  0.61542    0.70239  
Trainer.example.2019-03-29T20:16:40:
     best_epoch: 8
     best_loss: 0.1169810232400894
     weights/example.best.2019-03-29T20:16:40.p
CPU times: user 24min 54s, sys: 3min 42s, total: 28min 36s
Wall time: 9min 20s
"""
```

Now we can plot our training history:

```python
# get the history callback:
hist=trainer.callbacks.get('history_example')

# plot the epoch history:
hist.plot()

# plot the batch history:
hist.plot(batch=True)
```

Let's change the learning rate and do some more training.  First,  the model weights are currently set to the last epoch, but since we are using a non zero patience we need to load the best weights.

```
trainer.load_weights()
```

Now we can update our optimizer and train some more

```python
# here `new_opt` is our optimizer with a lower learning rate
trainer.compile(optimizer=new_opt)
trainer.fit(
    train_dl,
    valid_loader=valid_dl,
    nb_epochs=50,
    early_stopping=True,
    patience=2 )
```

And thats it. Now lets move on to the docs...


---

<a name='train'></a>
##### Trainer

```python
""" pytorch training loop

Usage:
    * create instance: trn=Trainer(...) 
     * set_callbacks: trn.set_callbacks(...)
    * train: trn.fit(...)
    * load best weights to model: trn.load_weights()
    * plot history: trn.callbacks.get('history').plot()

Args:
    model<nn.Module>: pytorch model
    criterion<criterion|None>: 
        - pytorch criterion
        - can be set later with .compile(...)
        - if criterion and optimizer are provide .compile() is automatically called
    optimizer<optimizer|None>: 
        - pytorch optimizer
        - can be set later with .compile(...)
        - if criterion and optimizer are provide .compile() is automatically called
    name<str>: weights are saved to files containing this name and a timestamp
    weights_dir<str>: 
        - directory for weights
        - defaults to `weights/` in current directory
    force_cpu<bool>: if true run on cpu even if gpu exists
""" 
```

---

<a name='callbacks'></a>
##### Callbacks

...

---

<a name='helpers'></a>
##### Helpers

...

---

<a name='metrics'></a>
##### Metrics

...


