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


    ********** hypers **********
    - swinB：input 192x192，window size=6
    - dataset：ImageNet-1K，a light data augmentation (random resize cropping/random flipping /color normalization)
    - AdamW：weight decay=0.05，beta=[0.9,0.999]
    - cosine LR scheduler：100 epochs (warmup 10 ep)，baseLR=8e-4
    - batch size：2048
    - random masking：mask ratio=0.6，mask patch size=32


    ********** grad_norm **********
    配置文件里面没有CLIP_GRAD参数，所以没有做梯度截断
    torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=2.0): 在对传入参数进行clip_by_global_norm以后，返回the Total norm
    源代码用默认自定义的utils.py/get_grad_norm：norm_type=2，计算了所有梯度的global norm，只用来输出没有真正进行梯度截断，用于观测训练状况


    ********** a light data augmentation **********
    - random resize cropping with scale range of [0.67, 1]
    - a aspect ratio range of [3/4, 4/3]
    - a random flipping
    - a color normalization steps


    初步实验下来发现模型输出棋盘格:
    - 因为pixel recover是通过reshape得到的，每个feature pixel负责一个x32的patch
    - 邻接patch的信息这个咋弄？




## finetuning settings
    - AdamW、batch size、masking 参数与pretrain一致
    - cosine LR：baseLR=5e-3
    - a stochastic depth rate：0.1
    - a layer-wise learning rate decay：0.9
    - strong data augmentation：RandAug，Mixup，Cutmix，label smoothing，random erasing







































