import albumentations as A
import yaml

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

IMG_WIDTH = config['IMG_WIDTH']
IMG_HEIGHT = config['IMG_HEIGHT']

transform = A.Compose(
    [
        A.Resize(IMG_HEIGHT, IMG_WIDTH, p=1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1)
    ])
