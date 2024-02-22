# wsol-apple
This project explores weakly supervised object localization techniques on a class-label dataset of apple images. 

# [Object Detection Model](https://github.com/Ka0Ri/Pytorch-pretrained-models)

- Try this code to run witl multi GPUs
```Python
def get_gpu_settings(gpu_ids: list[int]) -> Tuple[str, int, str]:
    '''
    Get GPU settings for PyTorch Lightning Trainer:
    Args:
        gpu_ids (list[int])
        n_gpu (int)
    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    '''
    if not torch.cuda.is_available():
        return "cpu", None, None

    n_gpu = len(gpu_ids)
    mapping = {
        'devices': gpu_ids if gpu_ids is not None else n_gpu if n_gpu is not None else 1,
        'strategy': 'ddp' if (gpu_ids or n_gpu) and (len(gpu_ids) > 1 or n_gpu > 1) else 'auto'
    }

    return "gpu", mapping['devices'], mapping['strategy']
```
