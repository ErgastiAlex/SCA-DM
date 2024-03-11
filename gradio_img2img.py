import argparse, os, sys, glob
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
import gradio as gr
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# Global variable to store the model
sampler = None

def parse_args():
    parser=argparse.ArgumentParser(description="Gradio Interface for Image to Image")
    parser.add_argument("--dataset", type=str, help="Dataset path", default="/home/datasets/CelebA-HQ/test/")
    return parser.parse_args()

args=parse_args()

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

# Placeholder for your neural network (replace with your actual model)
def load_neural_network():
    config = OmegaConf.load("configs/latent-diffusion/diffusion-sem-with_uc-mask-attn-test.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, "checkpoints/last.ckpt")
    # model = load_model_from_config(config, "checkpoints/last.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)
    return sampler

# Function to initialize the model if not already initialized
def initialize_model():
    global sampler
    if sampler is None:
        sampler = load_neural_network()


def sample(src_img, target_img, src_label, target_label, index_body_part, scale, ddim_steps, ddim_eta=0.0):
    global sampler

    n_samples=1

    model=sampler.model

    src_c =  model.get_learned_conditioning(src_img, src_label)
    target_c =  model.get_learned_conditioning(target_img, target_label)

    for i in index_body_part:
        src_c[:,i,:]=target_c[:,i,:]

    uc = None
    if scale != 1.0:
        # TODO: Use zero image or pass a zero vector to the model?
        # uc = model.get_learned_conditioning(opt.n_samples * [""])

        uc = {"context":torch.zeros(n_samples, 19, 1280).cuda(),
            # "y":src_label
            "y":torch.zeros(n_samples, 19, 256,256).cuda()}        
        # uc = {"context":model.get_learned_conditioning(torch.zeros_like(src_img), src_label),
        #     # "y":src_label
        #     "y":torch.zeros(n_samples, 19, 256,256).cuda()}

    cond={"context":src_c, "y":src_label}
    shape = [3, 256//4, 256//4]
    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                     conditioning=cond,
                                     batch_size=1,
                                     shape=shape,
                                     verbose=False,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=uc,
                                     eta=ddim_eta)

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

    # for x_sample in x_samples_ddim:
    #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
    #     Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
    #     base_count += 1
    # all_samples.append(x_samples_ddim)
    return x_samples_ddim

def transform_img(img):
    img = img.resize((256,256))
    img = np.array(img).astype(np.uint8)
    img = np.transpose(img, (2, 0, 1))
    img = (img/127.5 - 1.0).astype(np.float32)

    return torch.from_numpy(img[np.newaxis, ...]).cuda()

def transform_label(label):
    label = label.resize((256,256), Image.NEAREST)
    label = np.array(label).astype(np.uint8)
    label = torch.from_numpy(label).to(torch.int64)
    label = F.one_hot(label, num_classes=19).unsqueeze(0).permute(0,3,1,2).float()
    return label

# Your image processing and neural network inference logic
def generate_images(src_img_name, target_img_name, body_part, s, ddim_steps, empty_style):
    global args

    src_img_path=os.path.join(args.dataset, "images/", src_img_name+".jpg")
    target_img_path=os.path.join(args.dataset, "images/", target_img_name+".jpg")
    src_img=Image.open(src_img_path)
    target_img=Image.open(target_img_path)

    src_mask_path=os.path.join(args.dataset, "labels/", src_img_name+".png")
    target_mask_path=os.path.join(args.dataset, "labels/", target_img_name+".png")
    src_mask=Image.open(src_mask_path)
    target_mask=Image.open(target_mask_path)


    src_img = transform_img(src_img).cuda()
    target_img = transform_img(target_img).cuda()

    src_mask = transform_label(src_mask).cuda()
    target_mask = transform_label(target_mask).cuda()

    if empty_style==True:
        predictions = sample(torch.zeros(src_img.shape).cuda(), target_img, src_mask, target_mask, body_part, s, ddim_steps)
    else:
        predictions = sample(src_img, target_img, src_mask, target_mask, body_part, s, ddim_steps)
        
    src_img    = torch.clamp((src_img+1.0)/2.0, min=0.0, max=1.0)
    target_img = torch.clamp((target_img+1.0)/2.0, min=0.0, max=1.0)
    return (
            src_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(),
            target_img.detach().cpu().squeeze(0).permute(1,2,0).numpy(),
            predictions.detach().cpu().squeeze(0).permute(1,2,0).numpy()
        )

# Define the Gradio interface
iface = gr.Interface(
    fn=generate_images,
    inputs=[
        gr.Dropdown([str(x) for x in range(27998,30000)]),
        gr.Dropdown([str(x) for x in range(27998,30000)]),
        gr.CheckboxGroup(
            ['bg', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth'],
            type="index",
            label="Select Body Parts"
        ),
        gr.Slider(minimum=1, maximum=7.5, label="s"),
        gr.Slider(minimum=50, maximum=200, label="DDIM Step"),
        gr.Checkbox(label="Empty style"),
    ],
    outputs=[
        gr.Image(type="numpy", label="Input1",width=256),
        gr.Image(type="numpy", label="Input2",width=256),
        gr.Image(type="numpy", label="Processed Images",width=256),
    ],
)

initialize_model()
# Launch the Gradio interface
iface.launch(server_name="0.0.0.0")
