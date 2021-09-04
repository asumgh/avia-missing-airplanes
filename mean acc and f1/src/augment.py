from src.global_var import IMAGE_SIZE
import albumentations as albu
def train_augmentation(IMAGE_SIZE):
    transform = [
        albu.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
        albu.OneOf(
            [
                albu.HorizontalFlip(p = 0.5),
                albu.VerticalFlip(p = 0.4),
                albu.RandomRotate90(p = 0.1),
            ],
            p = 0.7
        ),
        albu.OneOf(
            [
                albu.HueSaturationValue(p=0.6),
                albu.RandomGamma(p=0.1),
                albu.RandomBrightnessContrast(p=0.3),
            ],
            p = 0.5
        ),
        albu.ShiftScaleRotate(
            shift_limit=0.002, scale_limit=0.2, rotate_limit=180,
            p=0.1
        )
    ]
    return albu.Compose(transform, p = 1)


def valid_augmentation(IMAGE_SIZE):
    transform = [
        albu.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, p=1),
    ]
    return albu.Compose(transform)

