import torchvision.transforms as T

def get_transform():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize((256, 192)),
        T.RandomHorizontalFlip(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.ToTensor(),
        T.Resize((256, 192)),
        normalize,
    ])

    return train_transform, valid_transform
