import os
import json
import yaml
import shutil
import random
from PIL import Image, ImageOps
import numpy as np

LM_PATH = 'output/bop_data/lm'
MODEL_PATH = 'lm/models/obj_000009.ply'
MODELS_PATH = 'lm/models/'
OUTPUT_PATH = 'synthetic_preprocessed'
TEST_PORTION = 0.2

# scene_gt.json -> gt.yml
# gt.yml also has obj_bb parameter, which is the 2D bounding box, TODO we generate it
with open(os.path.join(LM_PATH, 'train_pbr/000000/scene_gt.json')) as json_file:
    yml_fname = os.path.join(OUTPUT_PATH, 'data/09/gt.yml')
    os.makedirs(os.path.dirname(yml_fname), exist_ok=True)
    with open(yml_fname, "w") as yaml_file:
        data = json.load(json_file)
        yaml_file.write(yaml.safe_dump(data, default_flow_style=None))

# scene_camera.json -> info.yml
# info.yml doesn't contain cam_R_w2c or cam_t_w2c, so we throw them out
with open(os.path.join(LM_PATH, 'train_pbr/000000/scene_camera.json')) as json_file:
    yml_fname = os.path.join(OUTPUT_PATH, 'data/09/info.yml')
    os.makedirs(os.path.dirname(yml_fname), exist_ok=True)

    with open(yml_fname, "w") as yaml_file:

        data = json.load(json_file)

        # for k1, v1 in data.items():
        #     for k2, v2 in v1.items():
        #         if(k2 != "cam_R_w2c" and k2 != "cam_t_w2c"):
        #             print(k2)
        #             del data[k1][k2]

        yaml_file.write(yaml.safe_dump(data, default_flow_style=None))


# generate masks (they are used: https://github.com/ybkscht/EfficientPose/issues/8#issuecomment-770874205)
# depth images are not used by efficientpose, but they're useful for generating masks
depth_dir = os.path.join(LM_PATH, 'train_pbr/000000/depth/')
os.makedirs(os.path.dirname(os.path.join(OUTPUT_PATH, 'data/09/mask/')), exist_ok=True)
for fpath in os.listdir(depth_dir):
    img = Image.open(os.path.join(depth_dir, fpath))

    # img_arr = np.array(img)
    # img_arr = 65535 - img_arr

    img_arr = (np.array(img) == 65535)*65535
    img_arr = 65535 - img_arr

    print(img_arr.dtype)

    # img = img.convert('L')
    # inverted_img = ImageOps.invert(img)
    # img = img.convert('1')

    inverted_img = Image.fromarray(img_arr.astype(np.int32))
    inverted_img.save(os.path.join(OUTPUT_PATH, 'data/09/mask', fpath))

# rearrange files
# LM_PATH/train_pbr/000000/ -> OUTPUT_PATH/data/09/
dest_path = os.path.join(OUTPUT_PATH, 'data/09/rgb/')
source_path = os.path.join(LM_PATH, 'train_pbr/000000/rgb')

os.makedirs(os.path.dirname(dest_path), exist_ok=True)

for fpath in os.listdir(source_path):
    shutil.copy(os.path.join(source_path, fpath), os.path.join(dest_path))

# generate test/train split
# test.txt contains file names (excluding .png) of test data, line separated
# similar for train.txt
ids = [x.split(".")[0] for x in os.listdir(dest_path)] # no need to shuffle, already randomly chosen
test_ids = ids[:int(len(ids)*TEST_PORTION)]
train_ids = ids[int(len(ids)*TEST_PORTION):]

with open(os.path.join(OUTPUT_PATH, 'data/09/test.txt'), "w") as f:
    for x in test_ids:
        f.write(x + "\n")

with open(os.path.join(OUTPUT_PATH, 'data/09/train.txt'), "w") as f:
    for x in train_ids:
        f.write(x + "\n")

# copy over model 09 to models directory
dest_path = os.path.join(OUTPUT_PATH, 'models/obj_09.ply')
os.makedirs(os.path.dirname(dest_path), exist_ok=True)
shutil.copyfile(
    MODEL_PATH,
    os.path.join(dest_path)
)

# models_info.json -> models_info.yml
with open(os.path.join(MODELS_PATH, 'models_info.json')) as json_file:
    yml_fname = os.path.join(OUTPUT_PATH, 'models/models_info.yml')
    os.makedirs(os.path.dirname(yml_fname), exist_ok=True)
    with open(yml_fname, "w") as yaml_file:
        data = json.load(json_file)
        yaml_file.write(yaml.safe_dump(data, default_flow_style=None))


# steps to do manually afterwards:
# - Copy over models_info.yml from the preprocessed linemod dataset (might not need to do manually?)
# - find and replace all instance of a single quote (') in gt.yml and info.yml with nothing
# - change test/train split if you want to test all the images in the folder 
#   (just copy lines around, alternatively you can make TEST_PORTION = 1 before running)