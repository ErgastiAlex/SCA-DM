import argparse, os, sys, glob
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

label_list = ['bg', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_image_and_label(opt, path, img_name, h, w):
    im_path=os.path.join(path,"images",img_name)
    im = Image.open(im_path)
    im = im.resize((h,w))
    im = np.array(im).astype(np.uint8)
    im = np.transpose(im, (2, 0, 1))
    im = (im/127.5 - 1.0).astype(np.float32)

    label_path=os.path.join(path,"labels",img_name)
            #change ext
    label_path=label_path.replace(".jpg",".png")
    label = Image.open(label_path)
    label = label.resize((h,w), Image.NEAREST)
    label = np.array(label).astype(np.uint8)
    label = torch.from_numpy(label).to(torch.int64)
    label = F.one_hot(label, num_classes=opt.num_classes).unsqueeze(0).permute(0, 3, 1, 2).float()
    
    return torch.from_numpy(im[np.newaxis, ...]).cuda(), label.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_folder",
        type=str,
        nargs="?",
        default="/home/datasets/CelebA-HQ/test",
        help="the prompt to render"
    )
    parser.add_argument(
        "--source_img_name",
        type=str,
        nargs="+",
        default="",
    )
    parser.add_argument(
        "--target_img_name",
        type=str,
        nargs="+",
        default="",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/sem-img2img-samples"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=2,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default= 1.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=20
    )

    parser.add_argument(
        "--index_body_part",
        nargs="+",
        type=int,
        default="12",
    )

    opt = parser.parse_args()


    config = OmegaConf.load("configs/latent-diffusion/diffusion-sem.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    # model = load_model_from_config(config, "/home/ergale/projects/LDM-diffusion-sem/logs/2024-01-17T14-33-09_diffusion-sem/checkpoints/epoch=000039.ckpt")  # TODO: check path
    # model = load_model_from_config(config, "/home/ergale/projects/LDM-diffusion-sem/logs/2024-01-22T10-48-09_diffusion-sem/checkpoints/epoch=000040.ckpt")  # TODO: check path
    # model = load_model_from_config(config, "/home/ergale/projects/LDM-diffusion-sem/logs_old/2024-01-17T14-33-09_no_attn/checkpoints/last.ckpt")  # TODO: check path
    model = load_model_from_config(config, "/home/ergale/projects/LDM-diffusion-sem/logs/2024-01-26T09-36-36_no_attn_loss/checkpoints/last.ckpt")  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    source_images=opt.source_img_name
    target_images=opt.target_img_name

    assert len(source_images)==len(target_images)
    assert len(source_images)==1, "Only one source image is supported for now"

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():

            src_img,src_label=load_image_and_label(opt, opt.dataset_folder, source_images[0], opt.H, opt.W)
            target_img, target_label=load_image_and_label(opt, opt.dataset_folder, target_images[0], opt.H, opt.W)

            src_img=src_img.repeat(opt.n_samples, 1, 1, 1)
            src_label=src_label.repeat(opt.n_samples, 1, 1, 1)
            target_img=target_img.repeat(opt.n_samples, 1, 1, 1)
            target_label=target_label.repeat(opt.n_samples, 1, 1, 1)

            print("num samples", opt.n_samples)
            print("src img shape", src_img.shape)
            print("src label shape", src_label.shape)
            print("target img shape", target_img.shape)
            print("target label shape", target_label.shape)


            uc = None
            if opt.scale != 1.0:
                # TODO: Use zero image or pass a zero vector to the model?
                # uc = model.get_learned_conditioning(opt.n_samples * [""])
                uc=torch.zeros(opt.n_samples, 19, 1280).cuda()

                uc={"context":torch.zeros(opt.n_samples, 19, 1280).cuda(), 
                    "y":torch.zeros(opt.n_samples, 20, 256,256).cuda()}

            for n in trange(opt.n_iter, desc="Sampling"):
                src_c =  model.get_learned_conditioning(src_img, src_label)
                target_c =  model.get_learned_conditioning(target_img, target_label)
                # src_c=target_c
                # src_c=torch.randn_like(src_c)
                for i in opt.index_body_part:
                    src_c[:,i,:]=target_c[:,i,:]
                #     src_c[:,i,:]=torch.randn_like(src_c[:,i,:])
                # src_c=target_c
                cond={"context":src_c, "y":src_label}
                shape = [3, opt.H//4, opt.W//4]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=cond,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    name=""
    for i in opt.index_body_part:
        name+=label_list[i]+"_"
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{name}_{opt.source_img_name[0]}_{opt.target_img_name[0]}.png'))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


