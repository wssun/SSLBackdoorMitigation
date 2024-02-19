# <center>README</center>

## 1. overview

 This is the repo for our paper *On the Effectiveness of Distillation in Mitigating Backdoors in Pre-trained Encoder*.

 The repo consists of 3 modules:

- DATASETS: the dataloader module, STL10, CIFAR10, GTSRB, SVHN are supported
- KDistill_ZOO: the distillation loss module,  attention-based loss, layer-based loss and feature-based loss are supported.
- MODELS: the module to implement model architecture.
- EVALUATION: the module to evaluate the encoder performance.



## 2. Usage

```python
python distillation.py --student <your_model_path> --teacher <your_model_path> --results_dir <your_saved_dir> --distill_way 'ATD'
```

