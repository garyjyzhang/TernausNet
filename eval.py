import cv2
import argparse
import os

import torch
from unet_models import unet11
from torchvision.transforms import ToTensor, Normalize, Compose

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_model():
    model = unet11(pretrained='carvana')
    model.eval()
    return model.to(device)


def load_image(path, pad=True):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)

    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not pad:
        return img

    height, width, _ = img.shape

    if height % 32 == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = 32 - height % 32
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % 32 == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = 32 - width % 32
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)


def crop_image(img, pads):
    """
    img: numpy array of the shape (height, width)
    pads: (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

    @return padded image
    """
    (x_min_pad, y_min_pad, x_max_pad, y_max_pad) = pads
    height, width = img.shape[:2]

    return img[y_min_pad:height - y_max_pad, x_min_pad:width - x_max_pad]


img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)

    config = parser.parse_args()
    model = get_model()
    for file in os.listdir(config.input_dir):
        if file.endswith('.jpg'):
            print(file)
            img, pads = load_image(os.path.join(config.input_dir, file), pad=True)
            with torch.no_grad():
                input_img = torch.unsqueeze(img_transform(img).to(device), dim=0)
                mask = torch.sigmoid(model(input_img))
                mask_array = mask.data[0].cpu().numpy()[0]
                mask_array = crop_image(mask_array, pads)
                cv2.imwrite(os.path.join(config.output_dir, file.replace('.jpg', '.png')), mask_array)
