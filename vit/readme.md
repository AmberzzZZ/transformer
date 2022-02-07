
## vision transformer (ViT)
    
    paper: AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE
    官方repo: https://github.com/google-research/vision_transformer
    third repo: https://github.com/lucidrains/vit-pytorch

    task: supervised classification, visual版的BERT

    inputs: 将图片切成不重叠的16x16块，然后flatten，然后用learnable的线性层降维，然后添加cls token，然后加上PE
        * image patch sequence & trainable linear projection: conv2d, stride=kernel=patch_size, n_filters=emb_dim
        * cls_token: trainable [1,1,emb_dim], tile by b-axis
        * [cls_token, *patch_embeddings] + PEs
        * PE: trainable, normal(stddev=0.02), from BERT

    model: transformer encoder
        * patch_size
        * hidden_size: through all
        * MSA layer: attn_drop是相似度的dropout，ffn_drop是最后那个dense proj的dropout，都是常规dropout(drop on last dim), qkv_bias注意在small和huge上是False

    MLP head: 
        * 一种是用cls_token(BERT)
        * 一种是用mean on sequence
        * MLP在pre-training阶段是一个fc-tanh-fc，在fine-tuning阶段只有一层fc层，hidden_dim=emb_dim

    GeLU:
        Gaussian error linear unit: x * P(X <= x), where P(X) ~ N(0, 1)
        if approx:
            y = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
        else:
            y = 0.5 * x * (1 + erf(x / sqrt(2)))

    LN:
        https://www.geek-book.com/src/docs/keras/keras/keras.io/api/layers/normalization_layers/layer_normalization/index.html
        对该层的所有神经元求mean & var，所以mean & var的shape都是(b,hwd,1)
        trainable的情况下，given inputs [b,(hwd),c]，参数量是weight & bias (c,)

    ViT family:
    |   model   | depth |  hidden_size  | mlp_size | n_heads | Params
    | ViT-Base  |   12  |      768      |   3072   |    12   |  86M
    | ViT-Large |   24  |      1024     |   4096   |    16   |  307M
    | ViT-Huge  |   32  |      1280     |   5120   |    16   |  632M

    training details:
    -- train on scratch: strong regularization --
    * Adam: beta1=0.9, beta2=0.999, clip_norm=1
    * batch size: 4096
    * high weight decay: 0.1
    * linear lr: warmup & linear decay
    * resolution: 224
    -- finetuning --
    * SGD: momentum=0.9, clip_norm=1
    * no weight decay
    * batch size: 512
    * lr: larger, cosine decay 
    * resolution: 384, 这时sequence length变长了，要对PE基于其在原图上的位置做2D interpolation
    【实际实验下来】
    * ViT收敛比较慢，一个简单的6分类任务，resnet需要10个epoch，vit需要50个epoch
    * 即使有预训练权重，在transfer task上也需要一定epoch(20)的warmup，1e-6开始
    * warmup以后可以给到较大的lr(5e-4)会加速收敛，1e-3就不行了(batch size是32)
    * 后续的lr decay也很重要，acc到了92左右需要lr减小2个数量级，接近1e-6，不然不好收敛了


    子类继承模型：class ***(keras.Model)
        * init里面定义层不能复用
        * 批量定义的layer list里面每个layer必须在self作用空间下声明
        * checkpoint只能save_weights不能save_model，因为不支持get_config()和序列化

    主要缺陷：
    * 模型量级太大，batch size大，tpu级别训练
    * 训练数据量必须要大，不大精度不行，基本没办法在自己的数据集上train from scratch

    基于ViT的提升有：DeiT, T2T-ViT, LV-ViT
    


## LV-ViT

    paper: Token Labeling: Training an 85.4% Top-1 Accuracy Vision Transformer with 56M Parameters on ImageNet
    官方repo: https://github.com/zihangJiang/TokenLabeling
    
    patch embedding
    4-layer conv, kernel size [7,3,3,8], stride [2,1,1,8], filters 64
    [conv-bn-relu]-[conv-bn-relu]-[conv-bn-relu]-[conv-bias]

    re-labeling
    用另一个模型inference training set，给出一个K-dim dense score map
    在训练我们的模型的时候，random crop以后，基于cropped score map重新计算label

    token labeling
    基于re-labeling的dense score map，我们能够进一步地给到每一个token一个独立的K-dim label
    每个token的label和prediction能够独立计算一个CE：auxiliary token labeling loss

    mixtoken
    针对token grids，以cutMix的形式(crop box)，而不是noisy drop
    crop box的长宽是服从beta分布的（大概率落在较小值，从而保证总体的label是beta分布）
    token label是token individual的，所以mixtoken不影响每个token的label学习，所以源代码在计算token loss之前，将crop patches复原，这样token gt labels就不用转换了
    mixtoken本质上还是在augment原图，所以只影响cls token的prediction，cls token要基于随机mask重新计算

    loss
    cls_token对应的out embedding [b,D]接上MLP prediction head，预测总体的类别概率
    其他token对应的out embedding [b,N,D]接上shared MLP prediction head，学习每个token的类别预测，求全部tokens的平均
    再加权求和：cls_loss + 0.5*token_loss

    encoder block
    * stochastic depth (dropblock): random drop by sample
    * residual_scale: 给residual downscale有提升，scale=2

    training details
    * lr: linear scaling by batch 1e-3*batch_size/1024, 5 warmup epochs + cosine decay
    * AdamW: weight_decay=5e-2
    * batch_size: 1024
    * dropout = 0.
    * dropconnect = .1
    * randAug, mixup


## MAE
    
    paper: Masked Autoencoders Are Scalable Vision Learners
    official repo: https://github.com/facebookresearch/mae, torch, 据说有个tf/tpu版本，但是没开源
    3rd keras re-implementation: https://keras.io/examples/vision/masked_image_modeling/

    是一种reconstruct-based的自监督训练方法，用于提升encoder(backbone)模型的泛化性能

    -------- encoder-decoder ---------
    * encoder就是我们要预训练的模型，如ViT，输入是random sampled patches，[b,N1,emb_dim]
    * decoder是负责重建的模块，输入是full set patches(encoded & masked)，[b,N,emb_dim]
    * mask_ratio=.75，N1=0.25N
    * cls_token
    * mask_token
    * 都是self-attention block

    ------ input proj -------
    RandomMask
    trainable cls token
    constant sin2d PE

    --------- loss --------
    per-pixel MSE: pixel是patch pixel


    ViT的结构跟原始一致，训练的lr schedule有改变：per epoch改成per step，整体还是linear warmup + cosine decay
    * epoch = current_epoch + current_step/len(dataloader), 是个小数
    * before warmup_epochs: lr = args.lr * epoch / warmup_epochs
    * after warmup_epochs: lr = args.min_lr + (args.lr-args.min_lr) * 0.5 * (1+cos(pi/(total_epoch-warmup_epochs)*(epoch-warmup_epochs))), 一次cosine decay，no cycle，scale是max-min
    * base_lr = 1e-3, 对应的batch size是256，与实际的batch size成比例放缩
    * min_lr = 1e-6
    * warmup_epochs = 5
    * total_epochs = 50
    * weight_decay = 0.05
    * smoothing = 0.1






    











    












