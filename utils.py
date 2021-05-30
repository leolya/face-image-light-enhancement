import numpy as np
from dataloader import img_dataset
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.io import imsave, imread
from glob import glob
import os


def metrics(img1, img2):
    psnr = peak_signal_noise_ratio(img1, img2, data_range=1)
    ssim = structural_similarity(img1, img2, data_range=1, multichannel=True)
    return psnr, ssim


def test(model, path, device):
    dataset = img_dataset(path[0], path[1])
    num_imgs = len(dataset)
    psnr_sum = 0
    ssim_sum = 0
    with torch.no_grad():
        for i in range(num_imgs):
            data = dataset[i]
            inpt = data[0].to(device)
            tagt = data[1].to(device)
            pred = model(inpt.unsqueeze(0))

            tagt = tagt.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy().squeeze()

            tagt = np.moveaxis(tagt, 0, -1)
            pred = np.moveaxis(pred, 0, -1)

            tagt = (tagt + 1) / 2
            pred = (pred + 1) / 2
            pred = np.clip(pred, 0, 1)

            psnr, ssim = metrics(tagt, pred)
            psnr_sum += psnr
            ssim_sum += ssim

    return psnr_sum / num_imgs, ssim_sum / num_imgs


def get_samples(model, device, input_folder, output_folder):
    paths = glob(input_folder + "*.jpg")
    names = os.listdir(input_folder)
    paths.sort()
    names.sort()
    image_num = len(paths)

    for i in range(image_num):
        image = imread(paths[i])
        image = (image.astype(np.float32) * 2 / 255.) - 1
        image = np.moveaxis(image, -1, 0)
        image = np.expand_dims(image, 0)
        image = torch.tensor(image, dtype=torch.float)
        image = image.to(device)
        pred = model(image)
        pred = pred.cpu().detach().numpy().squeeze()
        pred = np.moveaxis(pred, 0, -1)
        pred = (pred + 1) / 2
        pred = np.clip(pred, 0, 1)
        imsave(output_folder + names[i], (pred * 255).astype(np.uint8))
