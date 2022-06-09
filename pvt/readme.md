
## PVT

    Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions
    repo: https://github.com/whai362/PVT


## SegFormer

    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    repo: https://github.com/NVlabs/SegFormer
    pretrained weights: https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia, Mit b0-b5
    trained weights: https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA, SegFormer b0-b5

    是基于PVT-back的novel分割框架
    * 对PVT attention block有改进：MLP里面夹了conv层
    * 对分割的decoder有新设计：使用light-weight but efficient MLP，而非stacking CNN


    * drops
    drop_attn: MSA中qk similarity map的dropout
    drop_mlp: MSA & MLP中fc的dropout
    drop_path: MSA和MLP block的residual drop，dbr


    * Mix Transformer encoders (MiT)

        ******* overlap patch embedding ********
        - patch size K=7/3
        - stride S=4/2
        - padding size P=3/1 (valid padding)
        - 通过卷积操作来实现
        - 在每个stage开始之前，[x4,x2,x2,x2]

        ******* efficient self attention block ********
        - 没有positional encoding，看起来非常清爽
        - sr_ratio，给K降维
        - MLP里面有个卷积


    * MLP decoder

        ******** 特征unify ********
        - 用fc层
        - 随着模型scaleup，这个fc的decoder dim在变大的
        - b0,b1: 256, b2,b3,b4,b5: 768,


        ******** 特征fusion ********
        - concat
        - conv-bn-relu
        - dropout
        - seg head：conv















