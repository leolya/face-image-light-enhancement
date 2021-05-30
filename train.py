import torch.optim as optim
from torch.utils.data import DataLoader
from model.model import UNet
from tqdm import tqdm
from dataloader import img_dataset
from utils import test
import os
from feature_loss import *


def train_unet(train_path, test_path, save_path,
               lr, batch_size, num_epoch, device):

    model = UNet()
    model.to(device)
    # load pre-trained weight for transfer learning
    # model.load_state_dict(torch.load("path_to_pretrained_weight", map_location="cuda:0"), strict=True)


    global_fe = Face_VGG_16()
    global_fe.to(device)
    global_fe.load_state_dict(torch.load("./vgg/face_vgg.pth"), strict=True)

    local_fe = Image_VGG_16()
    local_fe.to(device)
    local_fe.load_state_dict(torch.load("./vgg/image_vgg.pth"), strict=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = img_dataset(train_path[0], train_path[1])
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)
    criterion = nn.L1Loss(reduction='none')
    model.train()

    # psnr_max = 0
    ssim_max = 0

    for epoch in tqdm(range(num_epoch)):
        epoch_loss = 0
        epoch_loss_l1 = 0
        epoch_loss_f_g = 0
        epoch_loss_f_l = 0
        for i, batch in enumerate(dataloader):
            inpt = batch[0].to(device)
            tagt = batch[1].to(device)
            optimizer.zero_grad()
            otpt = model(inpt)
            l1_loss = criterion(otpt, tagt)
            l1_loss = l1_loss.mean()
            f_loss_g = global_feature_loss(global_fe, otpt, tagt, device)
            f_loss_l = local_feature_loss(local_fe, otpt, tagt, device)

            loss = l1_loss + 0.0002 * f_loss_g + 0.05 * f_loss_l
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_loss_l1 += l1_loss.item()
            epoch_loss_f_g += 0.0002 * f_loss_g.item()
            epoch_loss_f_l += 0.05 * f_loss_l.item()

        # validatioin
        model.eval()
        epoch_psnr, epoch_ssim = test(model, test_path, device)
        print('\nepoch_loss: ', epoch_loss, '  PSNR: ', epoch_psnr, " SSIM: ", epoch_ssim)
        print('\nl1_loss: ', epoch_loss_l1, " f_loss_g: ", epoch_loss_f_g, " f_loss_l: ", epoch_loss_f_l)
        model.train()

        # save the best model
        if epoch_ssim > ssim_max:
            ssim_max = epoch_ssim
            torch.save(model.state_dict(), save_path + "unet_weight_best.pth")
            print("\nmodel saved at: " + str(epoch) + " epoch")

    model.load_state_dict(torch.load(save_path + "unet_weight_best.pth"))
    model.eval()
    final_psnr, final_ssim = test(model, test_path, device)

    print('\nfinal result: PSNR: ', final_psnr, " SSIM: ", final_ssim)


if __name__ == '__main__':
    save_path = "./"
    train_path = ["/path_to/train_input/",
                  "/path_to/train_output/"]
    test_path = ["/path_to/test_input/",
                 "/path_to/test_output/"]

    device = torch.device('cuda:0')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_unet(train_path=train_path, test_path=test_path, save_path=save_path,
               lr=1e-4, batch_size=4, num_epoch=40, device=device)


