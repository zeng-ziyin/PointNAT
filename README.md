# PointNAT: Large Scale Point Cloud Semantic Segmentation via Neighbor Aggregation with Transformer

Here is the PyTorch implementation of the paper **_PointNAT: Large Scale Point Cloud Semantic Segmentation via Neighbor Aggregation with Transformer_**. 



- Dataset
```
mkdir -p data/S3DIS/
cd data/S3DIS
gdown https://drive.google.com/uc?id=1MX3ZCnwqyRztG1vFRiHkKTz68ZJeHS4Y
tar -xvf s3disfull.tar
cd ../../
```

### S3DIS
- Train
```
CUDA_VISIBLE_DEVICES='0' python examples/segmentation/main.py  --cfg cfgs/s3dis/spotr.yaml
```
- Inference
```
CUDA_VISIBLE_DEVICES='0' python examples/segmentation/main.py  --cfg cfgs/s3dis/spotr.yaml  mode=test --pretrained_path your/ckpt/path.pth
```


## Acknowledgement
This repo is built upon OpenPoints.
```
https://github.com/guochengqian/openpoints
```
