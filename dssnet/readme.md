# PyTorch Semantic Segmentation

### Introduction

This repository is a PyTorch implementation for DSSNet. The code is easy to use for training and testing on various datasets. And multiprocessing training is supported, tested with pytorch 1.6.0.


### Usage

1. Highlight:

   - Fast multiprocessing training ([nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html)) with official [nn.SyncBatchNorm](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm).
   - Better reimplementation results with well designed code structures.

2. Requirement:

   - Hardware: 4 GPUs (better with >=11G GPU memory)
   - Software: PyTorch>=1.6.0, Python3, [tensorboardX](https://github.com/lanpa/tensorboardX), 

3. Clone the repository:

   ```shell
   git clone https://github.com/ldrunning/DSSNet
   ```

4. Train:

   - Download related datasets and put them under folder specified in config or modify the specified paths.

      python  train_dss.py
	  
	  
5. Test:

   - Download trained segmentation models and put them under folder specified in config or modify the specified paths.
     
	 python  detect_images.py


### Performance

Description: **mIoU/mAcc/aAcc** stands for mean IoU, mean accuracy of each class and all pixel accuracy respectively. **ss** denotes single scale testing and **ms** indicates multi-scale testing. Training time is measured on a sever with GeForce RTX 1080 Ti. General parameters cross different datasets are listed below:

- Train Parameters:  aux_weight(0.4, 0.25), batch_size(16), base_lr(1e-2), power(0.9), momentum(0.9), weight_decay(1e-4).
- Test Parameters: scales(single: [1.0]).


### Citation

If you find the code or trained models useful, please consider citing:

```
@misc{DSSNet2021,
  author={Die Luo},
  title={DSSNet},
  howpublished={\url{https://github.com/ldrunning/DSSNet}},
  year={2021}
}

### Question

You are welcome to send pull requests or give some advices. Contact information: `luodie@hust.edu.cn`.

