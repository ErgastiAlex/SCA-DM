import torch
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image
import os

def merge_images_torch(folder1_path, folder2_path, output_path, rows=1):
    # Get list of images from both folders
    files1= os.listdir(folder1_path)
    files2= os.listdir(folder2_path)
    files1.sort(key=lambda x: int(x.split('.')[0]))
    files2.sort(key=lambda x: int(x.split('.')[0]))

    images_folder1 = [f for f in files1 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_folder2 = [f for f in files2 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Ensure both folders have the same number of images
    print(len(images_folder1), len(images_folder2))
    # if len(images_folder1) != len(images_folder2):
        # print("Error: Both folders must have the same number of images.")
        # return


    images_folder1 = images_folder1[:rows]
    images_folder2 = images_folder2[:rows]

    transforms_tt = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load images into PyTorch tensors
    tensor_folder1 = torch.stack([transforms_tt(Image.open(os.path.join(folder1_path, img))) for img in images_folder1])
    tensor_folder2 = torch.stack([transforms_tt(Image.open(os.path.join(folder2_path, img))) for img in images_folder2])

    merged_tensor = torch.cat([tensor_folder1, tensor_folder2], dim=3)

    # Create a grid of images
    merged_grid = make_grid(merged_tensor, nrow=1, padding=5, normalize=True)

    # Convert the PyTorch tensor to a PIL Image
    merged_image = transforms.ToPILImage()(merged_grid)

    # Save the merged image
    merged_image.save(output_path)
    print(f"Merged image saved to {output_path}")

# Example usage
### Merge generated samples image and original images to create a side-by-side comparison
default_path = "/home/datasets/CelebA-HQ/test/images"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_folder",
    type=str,
    nargs="?",
    default="/home/datasets/CelebA-HQ/test",
    help="the prompt to render"
)

parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="img.png"
)

opt= parser.parse_args()



merge_images_torch(default_path, opt.dataset_folder, opt.outdir, rows=10)
