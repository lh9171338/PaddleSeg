# 1. 数据集

[Dove Dataset](https://aistudio.baidu.com/datasetdetail/259489)

# 2. 指标

| model | config | mIoU | Background | Dove |
| :---: | :---: | :---: | :---: | :---: |
| UNet | [config](../configs/dove_120ep_224_unet.yaml) | 88.22 | 99.27 | 77.18 |
| DeepLabV3 | [config](../configs/dove_120ep_224_deeplabv3_resnet50_vd.yaml) | 85.03 | 98.99 | 71.07 |
| DeepLabV3P | [config](../configs/dove_120ep_224_deeplabv3p_resnet50_vd.yaml) | 89.67 | 99.34 | 80.00 |
| SegFormer-B0 | [config](../configs/dove_120ep_224_segformer_b0.yaml) | 86.92 | 99.17 | 74.67 |
| SegFormer-B5 | [config](../configs/dove_120ep_224_segformer_b5.yaml) | 88.55 | 99.28 | 77.82 |
