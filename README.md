# Semantic Class-Adaptive Diffusion Model (SCA-DM) 

<p align="center">
<img src=assets/Architecture.svg width=500/><br>
Model architecture. The encoder part of the UNet uses only standard Resnet
Block with SpatialTransformer to guide the diffusion process with the style embedding
obtained from Es. The middle block and the decoder part use SPADEResBlock, as in
SDM, to encapsulate the semantic mask info. The Mask attention mechanism is applied
inside the SpatialTransformer on the Cross Attention Map.
</p>

[**Controllable Face Synthesis with Semantic Latent Diffusion Models**]


Accepted at ICPR 2024 FBE workshop

## Results

<p align="center">
<img src=assets/model_capability.svg width=500/><br>
Our model can generate images in three ways: (a) Given a reference image, (b) Given a reference image but with a specific body part with a random style, (c) Fully noise based without any reference.
</p>


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
conda env create - environment.yaml
conda activate diffusion
```

or 

```
pip install -f requirements.txt
```
### Checkpoints
To use ```gradio_img2img.py``` download the model from [here](https://drive.google.com/file/d/1dZ4XQv8i3T2vtHnCuSsKXelz__pvUhdj/view?usp=sharing) and put it in the `checkpoints` folder and download the VQ-F4 (f=4, VQ (Z=8192, d=3), first row in the table) from the [LDM repo](https://github.com/CompVis/latent-diffusion) following their instructions.

<p align="center">
<video src="https://github.com/ErgastiAlex/LDM-Diffusion-sem/assets/20249175/390c24a6-4aee-458c-8028-eaf845174807" />

# Others
We thank CompVis for their opensource [codebase](https://github.com/CompVis/latent-diffusion) on which this project is based on.

# Citation
If you find this repository usefull please cite us:
```tex
@misc{ergasti2024controllablefacesynthesissemantic,
      title={Controllable Face Synthesis with Semantic Latent Diffusion Models}, 
      author={Alex Ergasti and Claudio Ferrari and Tomaso Fontanini and Massimo Bertozzi and Andrea Prati},
      year={2024},
      eprint={2403.12743},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.12743}, 
}
```




</p>
