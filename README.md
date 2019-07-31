### Torch Kit: pytorch utilities


TorchKit provides a Pytorch training loop with callbacks, custom loss functions and metrics, along with various helper utilities. 

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

<a name='train'></a>
##### Trainer

```python
""" pytorch training loop

Usage:
    * create instance: trn=Trainer(...) 
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


