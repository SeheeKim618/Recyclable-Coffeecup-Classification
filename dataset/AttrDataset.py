import torchvision.transforms as T
from config import argument_parser

parser = argument_parser()
args = parser.parse_args()

def get_transform():
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize((height, width)),
        T.RandomHorizontalFlip(),
        normalize,
    ])

    valid_transform = T.Compose([
        T.ToTensor(),
        T.Resize((height, width)),
        normalize,
    ])

    return train_transform, valid_transform
