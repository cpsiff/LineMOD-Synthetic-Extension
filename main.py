import blenderproc as bproc
import argparse
import os
import numpy as np
import random
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

"""
https://github.com/DLR-RM/BlenderProc/tree/main/examples/datasets/bop_object_pose_sampling
"""

# total images rendered will be NUM_SCENES*CAM_POSES_PER_SCENE
NUM_SCENES = 250
CAM_POSES_PER_SCENE = 10
OBJ_ID = 9 #ID 9 is duck

parser = argparse.ArgumentParser()
parser.add_argument('bop_parent_path', nargs='?', help="Path to the bop datasets parent directory")
parser.add_argument('bop_dataset_name', nargs='?', help="Main BOP dataset")
parser.add_argument('output_dir', nargs='?', help="Path to where the final files will be saved ")
parser.add_argument("-b", "--backgrounds", type=str, help="Path to background images to paste on.")
args = parser.parse_args()

bproc.init()

# load specified bop objects into the scene
bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name),
                          mm2m = True,
                          obj_ids = [OBJ_ID])

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(args.bop_parent_path, args.bop_dataset_name))  

# set shading
for j, obj in enumerate(bop_objs):
    obj.set_shading_mode('auto')
        
# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(500)
location = bproc.sampler.shell(center = [0, 0, -0.8], radius_min = 1, radius_max = 4,
                        elevation_min = 40, elevation_max = 75, uniform_volume = False)
light_point.set_location(location)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    obj.set_location(np.random.uniform([-0.2, -0.2, -0.2],[0.2, 0.2, 0.2]))
    obj.set_rotation_euler(bproc.sampler.uniformSO3(around_x=False, around_y=False, around_z=True))
    
# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

# Render five different scenes
for _ in range(NUM_SCENES):
    
    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(bop_objs)

    poses = 0
    # Render CAM_POSES_PER_SCENE camera poses
    while poses < CAM_POSES_PER_SCENE:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.2,
                                radius_max = 0.5,
                                elevation_min = 1,
                                elevation_max = 89,
                                uniform_volume = False)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(bop_objs)
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-0.7854, 0.7854))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, 
                                          frame = poses)
            poses += 1

    # Enable transparency so the background becomes transparent
    bproc.renderer.set_output_format(enable_transparency=True)

    # render the cameras of the current scene
    data = bproc.renderer.render()
    # seg_data = bproc.renderer.render_segmap(map_by = ["instance", "class", "cp_bop_dataset_name"], 
    #                                         default_values = {"class": 0, "cp_bop_dataset_name": None})


    d = data["depth"][0]

    # what blenderproc is doing (https://github.com/DLR-RM/BlenderProc/blob/53a72ad84e6974a6941556cf26b13989d0c845bb/blenderproc/python/writer/BopWriterUtility.py#L350)
    # is taking 4 1 7 0 -> 0 7 1 4, which is rgba -> abgr, completely reversing
    # the channel order, which works for an rgb image, but not for an rgba
    # image. As a hack (I should really just make a PR with blenderproc), 
    # we can transform the input ourselves to argb, which will get reversed
    # into bgra by blenderproc

    # data["colors"] has dimensions num_images x height x width x channels

    colors = np.array(data["colors"])
    r_channel = colors[:,:,:,0]
    g_channel = colors[:,:,:,1]
    b_channel = colors[:,:,:,2]
    a_channel = colors[:,:,:,3]
    colors = np.stack([a_channel, r_channel, g_channel, b_channel], 3)
    colors = [x for x in colors] # convert back to a list of np.arrays

    # Write data to bop format
    bproc.writer.write_bop(os.path.join(args.output_dir, 'bop_data'),
                           dataset = args.bop_dataset_name,
                           depths = data["depth"],
                           depth_scale = 0.05, 
                           colors = colors, 
                           color_file_format = "PNG", 
                           append_to_existing_output = True,
                           frames_per_chunk=100000) # idk why you'd want different chunks

    # paste backgrounds onto completed images
    print("adding random backgrounds")
    images_path = os.path.join(args.output_dir, 'bop_data', 'lm', 'train_pbr', '000000', 'rgb')
    for file_name in os.listdir(images_path):
        img_path = os.path.join(images_path, file_name)
        img = Image.open(img_path)
        img_w, img_h = img.size

        background_path = random.choice([
            os.path.join(args.backgrounds, p)
            for p in os.listdir(args.backgrounds)
        ])

        with Image.open(background_path).convert('RGB') as background:
            background = background.resize([img_w, img_h])
            # Pasting the current image on the selected background
            background.paste(img, mask=img.convert('RGBA'))

            background.save(img_path)