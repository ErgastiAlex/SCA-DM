# Semantic Class-Adaptive Diffusion Model (SCA-DM) 

<p align="center">
<img src=assets/Architecture.svg width=500/><br>
Model architecture. The encoder part of the UNet uses only standard Resnet
Block with SpatialTransformer to guide the diffusion process with the style embedding
obtained from Es. The middle block and the decoder part use SPADEResBlock, as in
SDM, to encapsulate the semantic mask info. The Mask attention mechanism is applied
inside the SpatialTransformer on the Cross Attention Map.
</p>

[**Towards Controllable Face Generation with Semantic Latent Diffusion Models**]


## Results
<p align="center">
<img src=assets/interpolation.svg /><br>
Interpolation of eyes, mouth, hair style and full style going from full target
(left) to full reference (right). Some details are highlighted for a clear observation of
changes.
</p>

<p align="center">
<img src=assets/Style_swap.svg /><br>
Style transfer comparison between different methods and our model. The style
of the reference image is applied to the target image. The overall consistency in style
swap is far better compared to state-of-the-art methods.
</p>



## How to use it
```
conda activate diffusion
python gradio_img2img.py --dataset CELEBA_HQ_TEST_FOLDER
```

### Requirements
A suitable [conda](https://conda.io/) environment named `diffusion` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate diffusion
```

### Checkpoints
To use ```gradio_img2img.py``` download the model from [here](https://univpr-my.sharepoint.com/:f:/g/personal/alex_ergasti_unipr_it/Emfpwm8xK3VPrYHgWzxpxHsB2ENjs5S6u5lPwI8CoO2I2g?e=hCb4Zi) and put it in the `checkpoints` folder.

<p align="center">
<video src="assets/gradio.webm" width="300" />
</p>