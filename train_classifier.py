import argparse
import os
from pprint import pprint

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning_fabric.utilities.seed import seed_everything
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, AUROC, ROC
from torchmetrics.classification import BinaryAUROC, BinaryROC
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from torchvision.datasets import ImageFolder

# solver settings
OPT = 'adam'  # adam, sgd
WEIGHT_DECAY = 0.0001
MOMENTUM = 0.9  # only when OPT is sgd
BASE_LR = 0.001
LR_SCHEDULER = 'step'  # step, multistep, reduce_on_plateau
LR_DECAY_RATE = 0.1
LR_STEP_SIZE = 10  # only when LR_SCHEDULER is step
LR_STEP_MILESTONES = [10, 15]  # only when LR_SCHEDULER is multistep


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train classifier.')
    parser.add_argument(
        '--dataset', '-d', type=str, required=False, help='Root directory of dataset'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, default='results', help='Output directory'
    )
    parser.add_argument(
        '--model-name', '-m', type=str, default='resnet18', help='Model name (timm)'
    )
    parser.add_argument(
        '--img-size', '-i', type=int, default=112, help='Input size of image'
    )
    parser.add_argument(
        '--epochs', '-e', type=int, default=100, help='Number of training epochs'
    )
    parser.add_argument(
        '--save-interval', '-s', type=int, default=10, help='Save interval (epoch)'
    )
    parser.add_argument('--batch-size', '-b', type=int, default=8, help='Batch size')
    parser.add_argument(
        '--num-workers', '-w', type=int, default=12, help='Number of workers'
    )
    parser.add_argument(
        '--use-image-folder', '-u', action='store_true', help='Use ImageFolder dataset'
    )
    parser.add_argument(
        '--csv-train', type=str, default=None, help='Csv training file'
    )
    parser.add_argument(
        '--csv-val', type=str, default=None, help='Csv validation file'
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--gpu-ids', type=int, default=None, nargs='+', help='GPU IDs to use'
    )
    group.add_argument('--n-gpu', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--seed', type=int, default=44, help='Seed')
    args = parser.parse_args()
    return args


def get_optimizer(parameters) -> torch.optim.Optimizer:
    if OPT == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    elif OPT == 'sgd':
        optimizer = torch.optim.SGD(
            parameters, lr=BASE_LR, weight_decay=WEIGHT_DECAY, momentum=MOMENTUM
        )
    else:
        raise NotImplementedError()

    return optimizer


def get_lr_scheduler_config(optimizer: torch.optim.Optimizer) -> dict:
    if LR_SCHEDULER == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_STEP_SIZE, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_STEP_MILESTONES, gamma=LR_DECAY_RATE
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1,
        }
    elif LR_SCHEDULER == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
        )
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val/loss',
            'interval': 'epoch',
            'frequency': 1,
        }
    else:
        raise NotImplementedError

    return lr_scheduler_config


class ImageTransform:
    def __init__(self, is_train: bool, img_size: int | tuple = 224):
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img)


class DatasetFromDataframe(Dataset):
    def __init__(self, csv_file, class_list, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.class_list = class_list

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        image = Image.open(self.df.file_path[index]).convert('RGB')
        label = self.class_list.index(self.df.state[index])
        # label = self.df.label[index]

        if self.transform:
            image = self.transform(image)
        return image, label


class SimpleData(LightningDataModule):
    def __init__(
            self,
            root_dir: str,
            img_size: int | tuple = 224,
            batch_size: int = 8,
            num_workers: int = 16,
            use_image_folder: bool = True,
            csv_file_train: str = None,
            csv_file_val: str = None,
            class_list: list = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_image_folder = use_image_folder
        self.csv_file_train = csv_file_train
        self.csv_file_val = csv_file_val
        self.class_list = class_list

        if not self.use_image_folder:  # Check if use_image_folder is False
            self.train_dataset = DatasetFromDataframe(
                csv_file=self.csv_file_train,
                class_list=self.class_list,
                transform=ImageTransform(is_train=True, img_size=self.img_size)
            )
            self.val_dataset = DatasetFromDataframe(
                csv_file=self.csv_file_val,
                class_list=self.class_list,
                transform=ImageTransform(is_train=False, img_size=self.img_size)
            )
            self.classes = self.class_list
            self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_list)}
        else:  # use_image_folder is True
            self.train_dataset = ImageFolder(
                root=os.path.join(root_dir, 'train'),
                transform=ImageTransform(is_train=True, img_size=self.img_size),
            )
            self.val_dataset = ImageFolder(
                root=os.path.join(root_dir, 'val'),
                transform=ImageTransform(is_train=False, img_size=self.img_size),
            )
            self.classes = self.train_dataset.classes
            self.class_to_idx = self.train_dataset.class_to_idx

    def train_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.val_dataset,
            # batch_size=self.batch_size,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
        )
        return dataloader


class SimpleModel(LightningModule):
    def __init__(
            self,
            model_name: str = 'resnet18',
            pretrained: bool = False,
            num_classes: int | None = None,
            task: str = 'multiclass',
    ):
        super().__init__()
        self.preds = []
        self.targets = []
        self.rocs = []
        self.save_hyperparameters()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained, num_classes=num_classes
        )
        self.train_loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task, num_classes=num_classes)

        self.val_loss = nn.CrossEntropyLoss()
        self.val_acc = Accuracy(task, num_classes=num_classes)
        self.f1_val = F1Score(task, num_classes=num_classes, average='macro')
        self.cm_val = ConfusionMatrix(task, num_classes=num_classes)
        self.auroc_val = AUROC(task, num_classes=num_classes)
        # self.auroc_val = BinaryAUROC(thresholds=None)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)

        loss = self.train_loss(out, target)
        acc = self.train_acc(pred, target)
        self.log_dict({'train/loss': loss, 'train/acc': acc}, prog_bar=True, on_epoch=True)

        return loss

    # def validation_epoch_start(self):
    #     self.preds = []
    #     self.targets = []

    def validation_step(self, batch, batch_idx):
        x, target = batch

        out = self(x)
        _, pred = out.max(1)
        prob = nn.functional.softmax(out, dim=-1)
        prob, _ = prob.max(1)

        loss = self.val_loss(out, target)
        acc = self.val_acc(pred, target)
        f1 = self.f1_val(pred, target)
        # cm = self.cm_val(pred, target)
        # auroc = self.auroc_val(out, target)
        self.log_dict({'val/loss': loss, 'val/acc': acc, 'val/f1': f1,
                       # 'val/f1': f1, 'val/cm': cm, 'val/auroc': auroc
                       })

        # self.cm_val(pred, target)
        # self.auroc_val(out, target)
        self.preds.append(pred)
        self.targets.append(target)
        self.rocs.append(prob)
        # return {'preds': pred, 'target': target}

    def on_validation_epoch_end(self):
        preds = torch.cat([self.preds[i] for i in range(len(self.preds))])
        targets = torch.cat([self.targets[i] for i in range(len(self.targets))])
        outs = torch.cat([self.rocs[i] for i in range(len(self.rocs))])
        # targets = torch.tensor([0, 0, 1, 1]).cuda()
        # print(outs)
        # print(targets)

        # Confusion matrix plot
        confusion_matrix = self.cm_val(preds, targets)
        df_cm = pd.DataFrame(confusion_matrix.cpu().data.numpy().astype(int), index=range(2), columns=range(2))
        plt.figure(figsize=(10, 7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral', fmt='d').get_figure()
        plt.close(fig_)

        self.logger.experiment.add_figure("Confusion matrix", fig_, self.current_epoch)

        # ROC Curve plot
        # aucroc = self.auroc_val(outs.cpu(), targets.cpu())

        # print(aucroc)

        self.preds.clear()
        self.targets.clear()
        self.rocs.clear()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.parameters())
        lr_scheduler_config = get_lr_scheduler_config(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


def get_basic_callbacks() -> list:
    lr_callback = LearningRateMonitor(logging_interval='epoch')
    ckpt_callback = ModelCheckpoint(
        filename='epoch{epoch:03d}',
        auto_insert_metric_name=False,
        save_top_k=1,
        # every_n_epochs=checkpoint_interval,
    )
    last_ckpt_callback = ModelCheckpoint(
        filename='last_model_{epoch:03d}-{val/loss:.4f}-{val/acc:02.0f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor=None,
    )
    best_ckpt_calllback = ModelCheckpoint(
        filename='best_model_{epoch:03d}-{val/loss:.4f}-{val/acc:.2f}',
        auto_insert_metric_name=False,
        save_top_k=1,
        monitor='val/loss',
        mode='min',
        verbose=True
    )
    early_stopping_callback = EarlyStopping(
        monitor='val/loss',  # Metric to monitor for improvement
        mode='min',  # Choose 'min' or 'max' depending on the metric (e.g., 'min' for loss, 'max' for accuracy)
        patience=10,  # Number of epochs with no improvement before stopping
    )
    return [last_ckpt_callback, best_ckpt_calllback, lr_callback, early_stopping_callback]

    # lr_callback = LearningRateMonitor(logging_interval='epoch')
    # ckpt_callback = ModelCheckpoint(
    #     monitor='val_loss',  # Choose the metric to monitor for best model
    #     mode='min',  # Choose 'min' for metrics like loss, 'max' for accuracy
    #     filename='best_model_{epoch:03d}_{val_loss:.4f}',  # Set the filename pattern for the best model checkpoint
    #     save_top_k=1,  # Save only the best model
    #     verbose=True,
    #     save_weights_only=False
    # )
    # return [ckpt_callback, lr_callback]


def get_gpu_settings(
        gpu_ids: list[int], n_gpu: int) -> tuple[str, int | list[int] | None, str | None]:
    """Get gpu settings for pytorch-lightning trainer:
    https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-flags

    Args:
        gpu_ids (list[int])
        n_gpu (int)

    Returns:
        tuple[str, int, str]: accelerator, devices, strategy
    """
    if not torch.cuda.is_available():
        return "cpu", None, None

    if gpu_ids is not None:
        devices = gpu_ids
        strategy = "ddp" if len(gpu_ids) > 1 else None
    elif n_gpu is not None:
        # int
        devices = n_gpu
        strategy = "ddp" if n_gpu > 1 else None
    else:
        devices = 1
        strategy = 'auto'

    return "gpu", devices, strategy


def get_trainer(args: argparse.Namespace) -> Trainer:
    callbacks = get_basic_callbacks()
    accelerator, devices, strategy = get_gpu_settings(args.gpu_ids, args.n_gpu)

    logs_dir = args.outdir
    tb_logger = TensorBoardLogger(os.path.join(logs_dir, 'tb_logs'))

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=logs_dir,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=[tb_logger],
        deterministic=True,
    )
    return trainer


if __name__ == '__main__':
    args = get_args()
    seed_everything(args.seed, workers=True)

    data = SimpleData(
        root_dir=args.dataset,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_image_folder=args.use_image_folder,
        csv_file_train = args.csv_train,
        csv_file_val=args.csv_val,
        class_list=['defective', 'normal'],
    )
    model = SimpleModel(
        model_name=args.model_name, pretrained=False,
        num_classes=len(data.classes), task='binary',
        # num_classes=2, task='binary',
        # model_name=args.model_name, pretrained=True, num_classes=6
    )

    trainer = get_trainer(args)
    print(data.img_size)
    print('Args:')
    pprint(args.__dict__)
    print('Training classes:')
    pprint(data.class_to_idx)
    trainer.fit(model, data)
