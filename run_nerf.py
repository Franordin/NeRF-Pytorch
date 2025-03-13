import os

import numpy as np
import torch

from load_llff    import load_llff_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_deepvoxels import load_dv_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()

    parser.add_argument('--config', is_config_file=True,
                        help="config file path")
    parser.add_argument('--datadir', type=str, default='./data/llff/fern',
                        help="input data directory")
    parser.add_argument('--basedir', type=str, default='./logs/',
                        help="where to store ckpts and logs")
    parser.add_argument('--expname', type=str,
                        help='experiment name')

    # dataset options
    parser.add_argument('--dataset_type', type=str, default='llff',
                        help="options: llff / blender / deepvoxels")
    parser.add_argument('--testskip', type=int, default=8,
                        help="will load 1/N images from test/val sets, useful for large datasets like deepvoxels")

    ## llff flags
    parser.add_argument('--factor', type=int, default=8,
                        help="downsample factor for LLFF images")
    parser.add_argument('--spherify', action='store_true',
                        help="set for spherical 360 scenes")
    parser.add_argument('--llffhold', type=int, default=8,
                        help="will take every 1/N images as LLFF test set, paper uses 8")
    parser.add_argument('--no_ndc', action='store_true',
                        help="do not use normalized device coordinates (set for non-forward facing scenes)")

    ## blender flags
    parser.add_argument('--half_res', action='store_true',
                        help="load blender synthetic data at 400x400 instead of 800x800")
    parser.add_argument('--white_bkgd', action='store_true',
                        help="set to render synthetic data on a white bkgd (always use for dvoxels)")

    ## deepvoxels flags
    parser.add_argument('--shape', type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # rendering options
    parser.add_argument('--render_test', action='store_true',
                        help='render the test set instead of render_poses path')

    return parser


def train():
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75, spherify=args.spherify)
        hwf = poses[0,:3,-1]
        poses = poses[:,:3,:4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if (i not in i_test and i not in i_val)])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.-images[..., -1:]) # alpha blending with white background
        else:
            images = images[..., :3]
    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[...,:3] * images[...,-1] + (1. - images[...,-1:])
        else:
            images = images[...,:3]
    elif args.dataset_type == 'deepvoxels':
        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print ('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal,     0, 0.5 * W],
            [    0, focal, 0.5 * H],
            [    0,     0,       1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
