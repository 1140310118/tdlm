# tdlm
实现了Transformer中的几种位置编码方案


位置编码的介绍与实验结果见 [知乎-位置编码上的一次实验](https://zhuanlan.zhihu.com/p/415020704)。


## 运行

设置好data目录和输出目录后运行

bash/pretrain.sh -c 0 -l 20 -p 'layerwise_rope' -e transformer

### requirements

- python3
- pytorch-1.8.1
- pytorch_lightning-1.4.9
- transformers-4.6.1
- einops-0.3.2
- performer_pytorch-1.1.0
