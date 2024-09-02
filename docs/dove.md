# 1. 数据集

[Dove Dataset](https://aistudio.baidu.com/datasetdetail/259489)

# 2. 指标

| model | Loss | config | mIoU | Background | Dove |
| :---: | :---: | :---: | :---: | :---: | :---: |
| UNet | CELoss | [config](../configs/dove_120ep_224_unet.yaml) | 88.22 | 99.27 | 77.18 |
| UNet | CELoss + LovaszSoftmax | [config](../configs/dove_120ep_224_unet_lovasz_loss.yaml) | 88.35 | 99.22 | 77.47 |
| UNet | CELoss + LovaszHingeLoss | [config](../configs/dove_120ep_224_unet_lovasz_hinge_loss.yaml) | 90.32 | 99.37 | 81.27 |
| DeepLabV3 | CELoss | [config](../configs/dove_120ep_224_deeplabv3_resnet50_vd.yaml) | 85.03 | 98.99 | 71.07 |
| DeepLabV3P | CELoss | [config](../configs/dove_120ep_224_deeplabv3p_resnet50_vd.yaml) | 89.25 | 99.29 | 79.21 |
| DeepLabV3P | CELoss + LovaszSoftmax | [config](../configs/dove_120ep_224_deeplabv3p_resnet50_vd_lovasz_loss.yaml) | 89.51 | 99.30 | 79.72 |
| DeepLabV3P | CELoss + LovaszHingeLoss | [config](../configs/dove_120ep_224_deeplabv3p_resnet50_vd_lovasz_hinge_loss.yaml) | 89.62 | 99.31 | 79.93 |
| DeepLabV3P | CELoss + FocalLoss | [config](../configs/dove_120ep_224_deeplabv3p_resnet50_vd_focal_loss.yaml) | 88.89 | 99.25 | 78.52 |
| SegFormer-B0 | CELoss | [config](../configs/dove_120ep_224_segformer_b0.yaml) | 86.92 | 99.17 | 74.67 |
| SegFormer-B5 | CELoss | [config](../configs/dove_120ep_224_segformer_b5.yaml) | 88.55 | 99.28 | 77.82 |
