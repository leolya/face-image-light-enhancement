import torch
import os
from model.model import UNet
from utils import test, get_samples

if __name__ == '__main__':
    model_path = "./weight/unet.pth"
    test_path = ["./data/test_input/",
                 "./data/test_output/"]

    input_folder = "./data/sample_input/"
    output_folder = "./data/sample_output/"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda:0')

    model = UNet()
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # evaluation
    # psnr, ssim = test(model, test_path, device)
    # print("PSNR: ", psnr, " SSIM: ", ssim)

    # generate sample images
    get_samples(model, device, input_folder, output_folder)
