## SimMIM

    self-supervised
    reconstruction task: pixel-level loss
    high mask ratio
    large patch size
    super-light 1-layer-linear head


## model zoo

    pre-trained at low resolution, finetuining at same/higher resolution:
    - Swin-B: 192 / [192,224]
    - Swin-L: 192 / 224
    - SwinV2-Huge: 192 / [224,512]
    - ViT-B: 224 / 224

    一个问题：192的时候window size是6，224的时候window size是7，这里面涉及了window pe的bilinear interpolation，不是会掉点吗？


## configs

    pretrain和finetune参数上的差别：
    - window size：随着input resolution变化
    - mask settings：mask ratio=0.6，patch size=32
    - DROP_PATH_RATE: pretrain=0., finetune=0.1（finetune的有标签数据量较少，需要正则）
    - training settings：
        -- pretrain 800个epoch(10 warmups)，finetune 100个epoch(20 warmups)
        -- lr scheduler
        -- layer decay


## pretraining details

    输入 x [b,h,w,3] & mask [b,h//s,w//s], encoder stride: swin32/vit16
    输出 predict x


    ********** mask token **********
    - learnable
    - embedding dim: 初始每个token的维度是[1,1,embedding_dim], normal initial
    - broadcast到全图: [b,L,embedding_dim]，因为所有mask的representation都是一致的


    ********** encoder **********
    - swin / vit
    - 输出最后一层的feature map：x32/x16，[b,h,w,c]


    ********** decoder **********
    - 线性层：1x1 conv，[b,h,w,s*s*3]，feature map上一个像素对应s*s*3的RGB patch, [b,h/32,w/32,32*32*3]
    - nn.PixelShuffle: 就是transpose+reshape，恢复成[b,h,w,3]

    ********** loss **********
    - 只计算masking tokens的loss
    - repeat_interleave: 对mask进行nearest上采样，到input resolution
    - mean of L1 on mask pixels








































