[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) PaddleSeg
===

This is a image segmentation package based on PaddlePaddle.

# Model
- [x] UNet
- [x] DeepLabV3P, DeepLabV3
- [x] SegFormer

# Install
```shell
git clone https://github.com/lh9171338/PaddleSeg.git
cd PaddleSeg
python -m pip install -r requirements.txt
python -m pip install -v -e .
```
# Train
```shell
sh train.sh <config>
```

# Test
```shell
sh test.sh <config>
```
