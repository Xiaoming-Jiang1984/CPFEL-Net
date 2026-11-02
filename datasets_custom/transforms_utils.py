from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import ImageFilter, ImageOps
import random

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

flip_and_color_jitter = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.4, hue=0.2)],
        p=0.5
    ),
    transforms.RandomGrayscale(p=0.5),
    GaussianBlur(p=0.5),
])