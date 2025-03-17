import os

import cv2
import imageio.v2 as imageio
import json
import numpy as np
import torch


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]
]).float()

# Row-Major
rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]
]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]
]).float()

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius) # distance z
    c2w = rot_phi(phi/180.*np.pi) @ c2w # @: matrix multiplication operator
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w # -x

    return c2w

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]   # all image counts  ( [nerf_synthetic:lego, skip == 8] => [0, 100, 113, 138] )

    for s in splits: # ['train', 'val', 'test']
        meta = metas[s] # all data in s.json
        imgs = []
        poses = []

        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip # default == 8

        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png') # .\data\nerf_synthetic\lego\train\r_0.png
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32) # float64 to float32
        counts.append(counts[-1] + imgs.shape[0])

        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)] # [ [0, 1, ..., 98, 99], [100, 101, ..., 112], [113, ..., 137] ]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas['test']['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x) # Calculate the focal length using the horizontal FoV. (Perspective Projection)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40+1)[:-1]], 0)

    if half_res:
        H = H//2 # int
        W = W//2 # int
        focal = focal / 2 # float64

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4)) # (138 x 400 x 400 x 4) size 0 matrix
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res

    return imgs, poses, render_poses, [H, W, focal], i_split
