import torch
import numpy as np

torch_model = torch.load("weights/mae_pretrain_vit_base.pth")
model_weights = torch_model['model']   # OrderedDict

# given input (224,224,3), full sequence: 224/16*224/16=196, patch dim: 16*16*3=768

# trainable cls_token: (1,1,768)
cls_token = {k:v for k,v in model_weights.items() if 'cls_token' in k}
print(cls_token.keys(), len(cls_token.keys()))

# trainable encoder_pe: (1,197,768)
pos_embed = {k:v for k,v in model_weights.items() if 'pos_embed' in k}
print(pos_embed.keys(), len(pos_embed.keys()))

# patch_embed: conv weight & bias
patch_embed = {k:v for k,v in model_weights.items() if 'patch_embed.proj' in k}
print(patch_embed.keys(), len(patch_embed.keys()))

# encoder block: LN-MSA(dense-dense)-LN-FFN(fc-fc), num_layers=12
blocks = {k:v for k,v in model_weights.items() if 'blocks' in k}
print(len(blocks.keys()))



from MAE import MAE_VIT
from VIT import VIT
vit_depth = 12
emb_dim = 768
# mae_model = MAE_VIT(input_shape=(224,224,3), patch_size=16, mask_ratio=.75, use_cls_token=True,
#                       emb_dim=768, depth=12, n_heads=16, mlp_ratio=4,
#                       decoder_emb_dim=512, decoder_depth=8, decoder_n_heads=16,
#                       norm_pix_loss=False)
# for layer in mae_model.layers:
#     if not layer.weights:
#         continue
#     print(layer.name)

#     if 'proj' in layer.name:  # conv-bias
#         torch_weights = [np.transpose(v,(2,3,1,0)) if 'weight' in k else v for k,v in patch_embed.items()]
#         layer.set_weights(torch_weights)

#     if 'add_token_1' in layer.name:   # cls_token / mask_token
#         torch_weights = [v for k,v in cls_token.items()]
#         layer.set_weights(torch_weights)

#     if 'vitattentionblock' in layer.name:
#         idx = int(layer.name.strip('vitattentionblock_'))
#         if idx>=vit_depth:   # decoder block
#             continue
#         torch_weights = [np.transpose(v,(1,0)) if len(v.shape)>1 else v for k,v in blocks.items() if 'blocks.%d'%idx in k]
#         # fix order issue in the origin OrderDict
#         qkv_bias = torch_weights.pop()
#         torch_weights.insert(3, qkv_bias)
#         layer.set_weights(torch_weights)

# mae_model.save_weights("weights/mae_vit_base.h5")

vit_model = VIT(input_shape=(224,224,3), n_classes=1000, patch_size=16, use_cls_token=True,
                emb_dim=768, depth=12, n_heads=12, mlp_ratio=4,
                att_drop_rate=0., drop_rate=0.1)
for layer in vit_model.layers:
    if not layer.weights:
        continue
    print(layer.name)

    if 'proj' in layer.name:  # conv-bias
        torch_weights = [np.transpose(v,(2,3,1,0)) if 'weight' in k else v for k,v in patch_embed.items()]
        layer.set_weights(torch_weights)

    if 'cls_token' in layer.name:   # cls_token
        torch_weights = [v for k,v in cls_token.items()]
        layer.set_weights(torch_weights)

    if 'encoder_pe' in layer.name:  # encoder pe
        torch_weights = [v for k,v in pos_embed.items()]
        layer.set_weights(torch_weights)

    if 'vitencoderblock' in layer.name:   # [vitencoderblock/vitattentionblock]
        idx = int(layer.name.strip('vitencoderblock_'))
        if idx>=vit_depth:   # decoder block
            continue
        torch_weights = [np.transpose(v,(1,0)) if len(v.shape)>1 else v for k,v in blocks.items() if 'blocks.%d'%idx in k]
        # fix order issue in the origin OrderDict
        qkv_bias = torch_weights.pop()
        torch_weights.insert(3, qkv_bias)
        layer.set_weights(torch_weights)

vit_model.save_weights("weights/vit_base.h5")



