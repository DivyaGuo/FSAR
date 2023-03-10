import os

from torchvision import transforms
from easyfl.datasets import FederatedSkeletonDataset


DB_NAMES = ["pku_part1", "pku_part2", "uestc", "ntu60", "ntu120"]


TRANSFORM_TRAIN_LIST = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_VAL_LIST = transforms.Compose([
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def prepare_train_data(data_dir, db_names=None):
    if db_names is None:
        db_names = DB_NAMES
    client_ids = []
    roots = []
    labels = []
    for db in db_names:
        client_ids.append(db)
        data_path = os.path.join(data_dir, db, 'xsub')
        roots.append(os.path.join(data_path, 'train_position.npy'))
        labels.append(os.path.join(data_path, 'train_label.pkl'))
    data = FederatedSkeletonDataset(roots=roots,
                                    labels=labels,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_TRAIN_LIST,
                                 client_ids=client_ids)
    return data


def prepare_test_data(data_dir, db_names=None):
    if db_names is None:
        db_names = DB_NAMES
    roots = []
    client_ids = []
    labels = []
    for db in db_names:
        client_ids.append(db)
        test_data_path = os.path.join(data_dir, db, 'xsub')
        roots.append(os.path.join(test_data_path, 'val_position.npy'))
        labels.append(os.path.join(test_data_path, 'val_label.pkl'))
    data = FederatedSkeletonDataset(roots=roots,
                                    labels=labels,
                                    simulated=True,
                                    do_simulate=False,
                                    transform=TRANSFORM_VAL_LIST,
                                    client_ids=client_ids)
    return data
