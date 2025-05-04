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

DEVICE = "cuda"
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)[..., :3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo2(args):
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        # left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        # right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        # print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        # for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):

        # Load batched images and convert to tensors
        left_np = np.load(left_npy_file)     # (N, H, W, C)
        right_np = np.load(right_npy_file)   # (N, H, W, C)

        # Transpose to (N, C, H, W) and move to GPU
        image0 = torch.from_numpy(left_np).permute(0, 3, 1, 2).float().cuda()
        image1 = torch.from_numpy(right_np).permute(0, 3, 1, 2).float().cuda()
        
        padder = InputPadder(image0.shape, divis_by=32)
        image0, image1 = padder.pad(image0, image1)

        disp = model(image0, image1, iters=args.valid_iters, test_mode=True)
        disp = disp.cpu().numpy()
        disp = padder.unpad(disp.float()).squeeze()


        # file_stem = imfile1.split("/")[-2]
        # filename = os.path.join(output_directory, f"{file_stem}.png")
        plt.figure()
        plt.imshow(disp, cmap="jet")
        plt.show()
        stereo_params = np.load(stereo_params_npz_file, allow_pickle = True)
    
        P1 = stereo_params['P1']
        # P1[:2] *= args.scale
        f_left = P1[0,0]
        baseline = stereo_params['baseline']
        depth = f_left*baseline/(disp+1e-6)
        plt.imshow(depth, cmap="jet")
        plt.show()
        np.save(output_path, depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", default=None)
    parser.add_argument("--save_numpy", action="store_true", help="save output as numpy arrays")
    parser.add_argument("-l", "--left_imgs", help="path to all first (left) frames", default=None)
    parser.add_argument("-r", "--right_imgs", help="path to all second (right) frames", default=None)
    parser.add_argument("--stereo_params_npz_file", help="path to stereo parameters npz file", default=None)
    parser.add_argument("--output_path", help="path to save output", default=None)
    parser.add_argument("--output_directory", help="directory to save output", default=None)
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
