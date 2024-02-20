# 1. 数据集

[Dove Dataset](https://aistudio.baidu.com/datasetdetail/259489)

# 2. 指标

| model | config | mIoU | Background | Dove |
| :---: | :---: | :---: | :---: | :---: |
| UNet | [config](../configs/dove_120ep_224_unet_warmup.yaml) | 83.35 | 98.66 | 68.05 |
| UNet | [config](../configs/dove_120ep_224_unet_warmup_deconv.yaml) | 85.41 | 98.93 | 71.90 |
| UNet | [config](../configs/dove_120ep_224_unet_warmup_deconv_1gpu.yaml) | 84.41 | 98.81 | 70.01 |
