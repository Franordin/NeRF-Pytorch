import os
import time

import imageio
import torch.nn.functional as F
from tqdm import tqdm

from load_llff    import load_llff_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_deepvoxels import load_dv_data

from run_nerf_helpers import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42)
DEBUG = False


def batchify(fn, chunk):
    """ Constructs a version of 'fn' that applies to smaller batches. """
    if chunk is None:
        return fn

    def ret(inputs):
        # fn == model_fine == Nerf => run "def forward(self, x)"
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def create_nerf(args):
    """ Instantiate NeRF's MLP model. """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) # 21,63 = get_embedder(10, 0)
    # embed_fn : 21 lambda functions (1 returns the input value as is, and the rest perform Positional Encoding
    # input_ch : 63 integers (3 is the num of camera coordinates, 60 is the num of arguments on which PE is performed

    embeddirs_fn = None
    input_ch_views = 0
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed) # 9,27 = get_embedder(4, 0)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, # 8
                 W=args.netwidth, # 256
                 input_ch=input_ch, # 63
                 output_ch=output_ch, # 5
                 skips=skips, # [4]
                 input_ch_views=input_ch_views, # 27
                 use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, # 8
                          W=args.netwidth_fine, # 256
                          input_ch=input_ch, # 63
                          output_ch=output_ch, # 5
                          skips=skips, # [4]
                          input_ch_views=input_ch_views, # 27
                          use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                         embed_fn=embed_fn, embeddirs_fn=embeddirs_fn,
                                                                         netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    #Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


# TODO
def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """ Volumetric rendering.
    :param ray_batch: arrayof shape [batch_size, ...].
                      All information necessary for sampling along a ray.
                      Including: ray origin, ray direction, min dist, max dist, and unit-magnitude viewing direction.
    :param network_fn: function. Model for predicting RGB and density at each point in space.
    :param network_query_fn: function used for passing queries to network_fn.
    :param N_samples: int. Number of different times to sample along each ray.
    :param retraw: bool. If True, include model's raw, unprocessed predictions.
    :param lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
    :param perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified random points in time.
    :param N_importance: int. Number of additional times to sample along each ray.
                         These samples are only passed to network_fine.
    :param network_fine: "fine" network with same spec as network_fn.
    :param white_bkgd: bool. If True, assume a white background.
    :param raw_noise_std: ...
    :param verbose: bool. If True, print more debugging info.
    :return:
        rgb_map:  [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
        disp_map: [num_rays]. Disparity map. 1 / depth.
        acc_map:  [num_rays]. Accumulated opacity along each ray. Comes from fine model.
        raw: [num_rays, num_samples, 4]. Raw predictions from model.
        rgb0:  See rgb_map.  Output for coarse model.
        disp0: See disp_map. Output for coarse model.
        acc0:  See acc_map.  Output for coarse model.
        z_std: [num_rays]. Standard deviation of distances along ray for each sample.
    """

    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1, 1]

    t_vals = torch.linspace(0., 1., steps=N_samples) # Divide the number of samples btw 0 and 1
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals) # Applying the num of samples btw near and far values (linear interpolation)
    else:
        z_vals = 1. / ((1./near) * (1. - t_vals) + far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples]) # Duplicate the sampled ray by the number of rays

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(42)
            t_rand = np.random.rand(*list(z_vals.shape)) # (160000, 11) -> list[160000, 11] -> random(0~1) 160K by 11 matrix
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    # raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0: # if this is a fine network TODO
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach() #TODO

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False) # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def understanding_perturb():
    '''
        Run this function in main function
        N_samples == 5
        num of rays == 1
    '''
    t_vals = torch.linspace(0., 1., steps=5) # Divide the number of samples btw 0 and 1

    z_vals = 2. * (1. - t_vals) + 6. * (t_vals) # Applying the num of samples btw near and far values (linear interpolation)
    z_vals = z_vals.expand([1, 5]) # Duplicate the sampled ray by the number of rays
    print(z_vals)
    print(z_vals[...,1:])
    print(z_vals[...,:-1])

    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    print(' mids', mids)
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    print('upper', upper)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    print('lower', lower)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape)
    print(t_rand)

    print(upper-lower)
    print(upper-lower * t_rand)

    z_vals = lower + (upper - lower) * t_rand
    print(z_vals)


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """ Render rays in smaller minibatches to avoid OOM (Out of Memory) """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk): # for 0 to 160000
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)

        for k in ret: # 5K times (160000 / 1024 * 32)
            #print(k) TODO
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render(H, W, K,
           chunk=1024*32, rays=None, c2w=None,
           ndc=True, near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """
    :param H: int. Height of image in pixels.
    :param W: int.  Width of image in pixels.
    :param focal: float. Focal length of pinhole camera
    :param chunk: int. Maximum number of rays to process simultaneously.
                       Used to control maximum memory usage. Does not affect final results.
    :param rays: array of shape [2, batch_size, 3].
                 Ray origin and direction for each example in batch.
    :param c2W: array of shape[3, 4]. Camera-to-world transformation matrix.
    :param ndc: bool. If True, represent ray origin, direction in NDC coordinates.
    :param near: float or array of shape [batch_size].  Nearest distance for a ray.
    :param far:  float or array of shape [batch_size]. Farthest distance for a ray.
    :param use_viewdirs: bool. If True, use viewing direction of a point in space in model.
    :param c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix
                            for camera while using other c2w argument for viewing directions.
    :return:
        rbg_map:  [batch_size, 3]. Predicted RGB values for rays.
        disp_map: [batch_size]. Disparity map. Inverse of depth.
        acc_map:  [batch_size]. Accumulated opacity (alpha) along a ray.
        extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)

        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape
    # TODO print(sh)

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # shape == torch.Size([160000, 11])
    """
    rays = [ rays_o_x,   rays_o_y,   rays_o_z,
             rays_d_x,   rays_d_y,   rays_o_z,
             near,       far,
             viewdirs_x, viewdirs_y, viewdirs_z ]
    """

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal/render_factor

    rgbs = []
    disps = [] # disparities

    t = time.time()

    print('render_poses', render_poses.shape)

    for i, c2w in enumerate(tqdm(render_poses)): # visualize the process
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        # TODO
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i == 0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


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
    parser.add_argument('--lindisp', action='store_true',
                        help="sampling linearly in disparity rather than depth")

    ## blender flags
    parser.add_argument('--half_res', action='store_true',
                        help="load blender synthetic data at 400x400 instead of 800x800")
    parser.add_argument('--white_bkgd', action='store_true',
                        help="set to render synthetic data on a white bkgd (always use for dvoxels)")

    ## deepvoxels flags
    parser.add_argument('--shape', type=str, default='greek',
                        help="options : armchair / cube / greek / vase")

    # rendering options
    parser.add_argument('--render_test', action='store_true',
                        help="render the test set instead of render_poses path")
    parser.add_argument('--render_only', action='store_true',
                        help="do not optimize, reload weights and render out render_poses path")
    parser.add_argument('--render_factor', type=int, default=0,
                        help="downsampling factor to speed up rendering, set 4 or 8 for fast preview")

    parser.add_argument('--multires', type=int, default=10,
                        help="log2 of max freq for positional encoding (3D location)")
    parser.add_argument('--multires_views', type=int, default=4,
                        help="log2 of max freq for positional encoding (2D direction)")

    parser.add_argument('--i_embed', type=int, default=0,
                        help="set 0 for default positional encoding, -1 for none")
    parser.add_argument('--N_importance', type=int, default=0,
                        help="number of additional fine samples per ray")
    parser.add_argument('--N_samples', type=int, default=64,
                        help="number of coarse samples per ray")

    parser.add_argument('--use_viewdirs', action='store_true',
                        help="use full 5D input instead of 3D")

    parser.add_argument('--perturb', type=float, default=1.,
                        help="set to 0. for no jitter, 1. for jitter")
    parser.add_argument('--raw_noise_std', type=float, default=0.,
                        help="std dev of noise added to regularize sigma_a output, 1e0 recommended")

    # training options
    parser.add_argument('--netdepth', type=int, default=8,
                        help="layers in network")
    parser.add_argument('--netwidth', type=int, default=256,
                        help="channels per layer")
    parser.add_argument('--netdepth_fine', type=int, default=8,
                        help="layers in fine network")
    parser.add_argument('--netwidth_fine', type=int, default=256,
                        help="channels per layer in fine network")

    parser.add_argument('--netchunk', type=int, default=1024*64,
                        help="number of pts sent through network in parallel, decrease if running out of memory")
    parser.add_argument('--chunk', type=int, default=1024*32,
                        help="number of rays processed in parallel, decrease if running out of memory")
    parser.add_argument('--lrate', type=float, default=5e-4,
                        help="learning rate")

    parser.add_argument('--ft_path', type=str, default=None,
                        help="specific weights npy file to reload for coarse network")
    parser.add_argument('--no_reload', action='store_true',
                        help="do not reload weights from saved ckpt")

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
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs = images, savedir=testsavedir, render_factor=args.render_factor)
            #rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs = images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
