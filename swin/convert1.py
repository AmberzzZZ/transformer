# interpolate the window pe to inference/finetune on a different resolution
import numpy as np
import cv2
from swin import SwinTransformer
import pandas as pd
import pickle
import keras.backend as K


# # given a low-resolution small-window pretrained model
# model = SwinTransformer(input_shape=(224,224,3), patch_size=4, emb_dim=128, n_classes=1000,
#                         num_layers=[2,2,18,2], num_heads=[4,8,16,32], window_size=7,
#                         residual_drop=0.5)
# model.load_weights("weights/swin_base_patch4_window7_224_22k.h5", by_name=True, skip_mismatch=True)  # skip cls fc
# weight_dict = {}
# for layer in model.layers:
#     if 'STB' in layer.name:
#         print(layer.name)
#         for sub_l in layer.layers:
#             if 'windowmultiheadattention' in sub_l.name:
#                 print(sub_l.name)
#                 for wmsa_l in sub_l.layers:
#                     if 'relative_position_bias' in wmsa_l.name:
#                         print(wmsa_l.name, wmsa_l.weights[0].shape)
#                         tmp = K.get_value(wmsa_l.weights[0])
#                         weight_dict[wmsa_l.name] = tmp
#                     elif 'dense' in wmsa_l.name:   # for check
#                         print(wmsa_l.name, wmsa_l.weights[0].shape)
#                         tmp = K.get_value(wmsa_l.weights[0])
#                         weight_dict[wmsa_l.name] = tmp

# print(weight_dict.keys())
# with open('weight_dict.pkl', 'wb') as f:
#     pickle.dump(weight_dict, f)


weight_dict = pd.read_pickle("weight_dict.pkl")
model = SwinTransformer(input_shape=(384,384,3), patch_size=4, emb_dim=128, n_classes=1000,
                        num_layers=[2,2,18,2], num_heads=[4,8,16,32], window_size=12,
                        residual_drop=0.5)
model.load_weights("weights/swin_base_patch4_window7_224_22k.h5", by_name=True, skip_mismatch=True)

num_layers = [2,2,18,2]
num_heads = [4,8,16,32]
window_size = 12

stage_ref = {}
for stage_idx, n_blocks in enumerate(num_layers):
    for i in range(n_blocks):
        STB_idx = (sum(num_layers[:stage_idx])+i) // 2
        stage_ref[STB_idx] = stage_idx
print(stage_ref)

for layer in model.layers:
    if not layer.get_weights():
        continue
    if 'STB' in layer.name:
        # each STB is a WMSA-SWMSA
        block_idx = int(layer.name.split('_')[-1])
        stage_id = stage_ref[block_idx]
        print('------------- layer name: %s -------------' % layer.name, 'stage', stage_id)
        for sub_l in layer.layers:
            if 'windowmultiheadattention' in sub_l.name:
                # WMSA & SWMSA: dense,dense,win_pe
                print(sub_l.name)
                for wmsa_l in sub_l.layers:
                    if 'dense' in wmsa_l.name:     # use to check layers
                        orig_weights = weight_dict[wmsa_l.name]
                        current_weights = K.get_value(wmsa_l.weights[0])
                        print(np.array_equal(orig_weights, current_weights))
                    elif 'relative_position_bias' in wmsa_l.name:   # ((2*h-1)*(2*w-1),num_heads)
                        print(wmsa_l.name)
                        # bicubic
                        orig_weights = weight_dict[wmsa_l.name]
                        h = w = window_size
                        target_shape = ((2*h-1)*(2*w-1), num_heads[stage_id])  # target h,w
                        if not np.array_equal(orig_weights.shape, np.array(target_shape)):
                            print('orig shape: ', orig_weights.shape, 'target shape: ', target_shape)
                            target_bias = cv2.resize(orig_weights, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)   # # (w,h) for cv2.resize
                            model.get_layer(layer.name).get_layer(sub_l.name).get_layer(wmsa_l.name).set_weights([target_bias])

model.save_weights('swin_base_patch4_window7_224_22k.h5'.replace("window7", "window12"))






