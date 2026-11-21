from dataclasses import dataclass

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TransformBundle:
    train: object
    evaluation: object


def _normalization(enabled: bool) -> list:
    if not enabled:
        return []
    return [transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]


def build_classification_transforms(
    image_size: int = 224,
    normalize: bool = True,
    train_views: int = 1,
) -> TransformBundle:
    if train_views != 1:
        raise ValueError("Only single-view transforms are supported")

    train = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
            p=0.8,
        ),
        transforms.ToTensor(),
        *_normalization(normalize),
    ])

    evaluation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        *_normalization(normalize),
    ])

    return TransformBundle(train=train, evaluation=evaluation)
