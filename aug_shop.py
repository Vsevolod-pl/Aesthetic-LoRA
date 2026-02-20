from torchvision.transforms import v2

augmentations_set = {
    "RandomResize": v2.RandomResize,
    "RandomCrop": v2.RandomCrop,
    "RandomResizedCrop": v2.RandomResizedCrop,
    "RandomHorizontalFlip": v2.RandomHorizontalFlip,
    "RandomVerticalFlip": v2.RandomVerticalFlip,
    "RandomAffine": v2.RandomAffine,
    "RandomPerspective": v2.RandomPerspective,
    "RandomRotation": v2.RandomRotation,
}
