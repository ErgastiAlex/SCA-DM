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

label_list = ['bg','skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
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
        default="outputs/fid"
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
        default= 1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=19
    )

    parser.add_argument(
        "--index_body_part",
        nargs="+",
        type=int,
        default="12",
    )

    opt = parser.parse_args()

    config = OmegaConf.load("configs/latent-diffusion/diffusion-sem-with_uc-mask-attn-test-sdm.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "checkpoints/epoch=000235.ckpt") 

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


    datasets = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    datasets.prepare_data()
    datasets.setup()

    base_count = 28000
    with torch.no_grad():
        with model.ema_scope():
            for data in datasets.val_dataloader():
                uc = None
                for sample_n in range(opt.n_samples):

                            
                    x , cond, label = model.get_input(data, "image")

                    cond = model.get_learned_conditioning(cond, label)
                    cond = {"context":cond,"y":label}

                    if opt.scale != 1.0:
                        # TODO: Use zero image or pass a zero vector to the model?
                        # uc = model.get_learned_conditioning(opt.n_samples * [""])
                        uc=torch.zeros(x.shape[0], 19, 1280).cuda()

                        uc={"context":torch.zeros(x.shape[0], 19, 1280).cuda(), 
                            "y":torch.zeros(x.shape[0], 19, 256,256).cuda()}


                    shape = [3, opt.H//4, opt.W//4]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                     conditioning=cond,
                                                     batch_size=x.shape[0],
                                                     shape=shape,
                                                     verbose=False,
                                                     unconditional_guidance_scale=opt.scale,
                                                     unconditional_conditioning=uc,
                                                     eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                    if sample_n>=1:
                        base_count -= x.shape[0] # multiple samples have the same numbers, so it must be resetted
                    for i in range(x.shape[0]):
                        sample = x_samples_ddim[i]
                        sample = rearrange(sample, 'c h w -> h w c')
                        sample = 255. * sample.cpu().numpy()
                        if opt.n_samples>1:
                            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:06d}_sample_{sample_n:02d}.png'))
                        else:
                            Image.fromarray(sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:06d}.png'))
                        base_count += 1

                    # attentions=model.model.diffusion_model.attn
                    # sim=attentions[-1]
                    # #save using matplotlib
                    # b,d,c=sim.shape
                    # h=int(np.sqrt(d))
                    # attn_1=sim.softmax(dim=-1)
                    # attn_1=rearrange(attn_1,"(b1 head) (height width) c-> b1 head c height width",b1=x.shape[0],height=h)
                    # attn_1=attn_1[0]
                    # attn_1=torch.mean(attn_1,dim=0) # c h w
                    # print(attn_1.shape)
                    # import matplotlib.pyplot as plt
                    # for i,attn_c in enumerate(attn_1):
                    #     #save with matplotlib
                    #     plt.imshow(attn_c.cpu().numpy(),cmap='turbo')
                    #     plt.axis('off')
                    #     # save
                    #     plt.savefig(f'attn_{base_count:06d}_{i}.png',bbox_inches='tight', pad_inches=0)
                    
                    # lbl=Image.open("/home/datasets/CelebA-HQ/test/labels/28000.png").resize((32,32), Image.NEAREST)
                    # #to numpy
                    # lbl=np.array(lbl).astype(np.uint8)
                    # lbl=F.one_hot(torch.tensor(lbl).long(),opt.num_classes).permute(2,0,1).float().unsqueeze(0).cuda()
                    # max_neg_value = -torch.finfo(sim.dtype).max
                    # import math
                    # print("Batch size", x.shape[0])
                    # print("sim size", sim.size())
                    # print("lbl size", lbl.size())
                    # sim = rearrange(sim,"(b1 head) (height width) c-> b1 head c height width",b1=x.shape[0],height=h)
                    # attn_2=sim.clone()
                    # attn_2=attn_2[0:1]
                    # sim=sim[0:1]
                    # for i in range(sim.size(2)):
                    #     att_i = sim[:, :, i].unsqueeze(2).clone()
                    #     m_i = torch.repeat_interleave(lbl[:, i].unsqueeze(1), 14, dim=1).unsqueeze(2)
                    #     print((att_i*m_i).squeeze(2).size())
                    #     print(attn_2.size())
                    #     attn_2[:, :, i] = (att_i*m_i).squeeze(2)
                    #     attn_2[attn_2 == 0] = 0
                    
                    # attn_2=rearrange(attn_2,"b1 head c height width -> (b1 head) (height width) c")
                    # attn_3=attn_2.clone()
                    # attn_3[attn_3==0]=max_neg_value

                    # attn_2=attn_2.softmax(dim=-1)
                    # attn_2=rearrange(attn_2,"(b1 head) (height width) c-> b1 head c height width",b1=1,height=h)
                    # attn_2=attn_2[0]
                    # attn_2=torch.mean(attn_2,dim=0) # c h w

                    # attn_3=attn_3.softmax(dim=-1)
                    # attn_3=rearrange(attn_3,"(b1 head) (height width) c-> b1 head c height width",b1=1,height=h)
                    # attn_3=attn_3[0]
                    # attn_3=torch.mean(attn_3,dim=0) # c h w

                    # for i,(attn_c1,attn_c2) in enumerate(zip(attn_2,attn_3)):
                    #     plt.imshow(0.999*attn_c1.cpu().numpy()+0.001*attn_c2.cpu().numpy(),cmap='turbo')
                    #     plt.axis('off')
                    #     # save
                    #     plt.savefig(f'attn_prod_{base_count:06d}_{i}.png',bbox_inches='tight', pad_inches=0)


                    # print("A",flush=True)

    print(f"Your samples are ready and waiting for you here: \n{outpath} \nEnjoy.")


