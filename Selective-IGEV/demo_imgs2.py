import sys
sys.path.append("core")

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.igev_stereo import IGEVStereo
from core.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
import h5py
import cv2

def resize_image(img_chw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    # img_chw: C x H x W numpy array    
    img_hwc = np.transpose(img_chw, (1, 2, 0))
    resized_hwc = cv2.resize(img_hwc, (target_w, target_h), interpolation=interpolation)
    resized_chw = np.transpose(resized_hwc, (2, 0, 1))
    
    return resized_chw

def resize_batch(batch_nchw, target_h, target_w, interpolation=cv2.INTER_LINEAR):
    return np.stack([resize_image(img, target_h, target_w, interpolation) for img in batch_nchw])

DEVICE = 'cuda'

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo2(args):
    

    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    stereo_params = np.load(args.stereo_params_npz_file, allow_pickle=True)
    P1 = stereo_params['P1']
    #P1[:2] *= args.scale
    f_left = P1[0,0]
    baseline = stereo_params['baseline']


    if args.left_h5_file and args.right_h5_file:
        try:
            with h5py.File(args.left_h5_file, 'r') as f:
                left_all = f['data'][()]   # or np.array(f['left'])
            with h5py.File(args.right_h5_file, 'r') as f:
                right_all = f['data'][()]
        except Exception as e:            
            with h5py.File(args.left_h5_file, 'r') as f:
                left_all = f['left'][()]   # or np.array(f['left'])
            with h5py.File(args.right_h5_file, 'r') as f:
                right_all = f['right'][()]
      
        print(left_all.shape, right_all.shape)
    if left_all.ndim==3:
        left_all = left_all[None]
        right_all = right_all[None]

    N,C,H,W = left_all.shape
    if args.process_only:
        N_stop = args.process_only
    else:
        N_stop = N
    N_max = N_stop
    # aspect ratio for Canon EOS 6D is 3/2. 3648
    # image size of about 1586x2379 works with batch_size of 1, 
    # with resize_factor of 2.3 at 28s/image, up to ~25 images.
    small_dim = min(H,W)
    large_dim = max(H,W)
    resize_factor = max(round(small_dim/1586,1), round(large_dim/2379,1))
    # resize_factor = 1.5
    print(f"Found {N} images,  applying resize_factor {resize_factor} Saving files to {out_dir}.")
    args.max_disp = int(np.ceil(W/resize_factor/4/64/3)*64*3)
    print("args.max_disp", args.max_disp)    

    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt, weights_only=False))

    model = model.module
    model.to(DEVICE)
    model.eval()


    disp_all = []
    depth_all = []

    with torch.no_grad():     
        for i in tqdm(range(0, N, args.batch_size), desc="Processing batches"):  
            img0 = left_all[i:i+args.batch_size]
            img1 = right_all[i:i+args.batch_size]

            if len(img0.shape)==3:
                img0 = img0[None,...]

            if len(img1.shape)==3:
                img1 = img1[None,...]

            img0 = resize_batch(img0, round(H/resize_factor) ,round(W/resize_factor))
            img1 = resize_batch(img1, round(H/resize_factor), round(W/resize_factor))

            img0 = torch.as_tensor(img0).cuda().float()
            img1 = torch.as_tensor(img1).cuda().float()
            print(img0.min(), img0.max(), img0.shape)
            padder = InputPadder(img0.shape, divis_by=32)
            img0, img1 = padder.pad(img0, img1)
            print(img0.shape)
            with torch.amp.autocast("cuda",enabled=True):
                disp = model(img0, img1, iters=args.valid_iters, test_mode=True)
                disp = padder.unpad(disp).cpu().squeeze().numpy()            
            
            depth = f_left * baseline / (disp + 1e-6)
            depth_all.append(depth)        
            disp_all.append(disp)
            if i+args.batch_size >= N_stop:
                N_max = i + img0.shape[0]
                break
    disp_all = np.concatenate(disp_all, axis=0).reshape(N_max,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)
    depth_all = np.concatenate(depth_all, axis=0).reshape(N_max,round(H/resize_factor),round(W/resize_factor)).astype(np.float16)

    with h5py.File(out_dir/'leftview_disp_depth.h5', 'w') as f_out:
        f_out.create_dataset('disp', data=disp_all, compression='gzip')
        f_out.create_dataset('depth', data=depth_all, compression='gzip')   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch size for processing")
    parser.add_argument("--restore_ckpt", help="restore checkpoint", default=None)
    parser.add_argument("--left_h5_file", default="", type=str)
    parser.add_argument("--right_h5_file", default="", type=str)
    parser.add_argument("--stereo_params_npz_file", default = "", type = str)        
    parser.add_argument("--out_dir", default=f'../output/', type=str, help='the directory to save results')
    parser.add_argument("--save_numpy", action="store_true", help="save output as numpy arrays")
    parser.add_argument("--process_only",default=None,type=int)
    # parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default=None)
    # parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default=None)
    # parser.add_argument("--stereo_params_npz_file", help="path to stereo parameters npz file", default=None)
    # parser.add_argument("--output_path", help="path to save output", default=None)
    # parser.add_argument("--output_directory", help="directory to save output", default=None)
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--precision_dtype",default="float16",choices=["float16", "bfloat16", "float32"],help="Choose precision type: float16 or bfloat16 or float32")
    parser.add_argument("--valid_iters",type=int,default=32,help="number of flow-field updates during forward pass")
    # Architecture choices
    parser.add_argument("--hidden_dims",nargs="+",type=int,default=[128] * 3,help="hidden state and context dimensions")
    parser.add_argument("--corr_implementation",choices=["reg", "alt", "reg_cuda", "alt_cuda"],default="reg",help="correlation volume implementation")
    parser.add_argument("--shared_backbone",action="store_true",help="use a single backbone for the context and feature encoders")
    parser.add_argument("--corr_levels",type=int,default=2,help="number of levels in the correlation pyramid")
    parser.add_argument("--corr_radius", type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument("--n_downsample",type=int,default=2,help="resolution of the disparity field (1/2^K)")
    parser.add_argument("--slow_fast_gru",action="store_true",help="iterate the low-res GRUs more frequently")
    parser.add_argument("--n_gru_layers", type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument("--max_disp", type=int, default=192, help="max disp of geometry encoding volume")

    args = parser.parse_args()

    demo2(args)
