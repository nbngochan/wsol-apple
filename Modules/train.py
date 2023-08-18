import torch
from torchvision.utils import make_grid, draw_bounding_boxes, draw_segmentation_masks
import torch.nn.functional as F
from Detector import WrappingDetector
from ultis import *
from pytorch_lightning import LightningModule
<<<<<<< HEAD
from Dataloader import DataModule
from neptune.types import File
=======
from torchmetrics.detection.mean_ap import MeanAveragePrecision
>>>>>>> 67f00d1755434582e30a3ffdc0a91b563ba55a1c


seed_everything(44)
# torch.backends.cudnn.benchmark = True # keep True if all the input have same size.
torch.set_float32_matmul_precision('medium')

<<<<<<< HEAD

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y=None):
        pass

class ModelBase(LightningModule):
=======
    
class Model(LightningModule):
>>>>>>> 67f00d1755434582e30a3ffdc0a91b563ba55a1c
    def __init__(self, PARAMS):
        super().__init__()
        self.save_hyperparameters()

        self.architect_settings = PARAMS['architect_settings']
        self.train_settings = PARAMS['training_settings']
        self.dataset_settings = PARAMS['dataset_settings']
        self.task = PARAMS['task']
        # Model selection
        self.model = DummyModel()
        # Metric selection
        self.metrics_name = self.train_settings['metric']
        self.train_metrics = get_metric(self.metrics_name, self.dataset_settings['n_cls'])
        self.valid_metrics = get_metric(self.metrics_name, self.dataset_settings['n_cls'])

    def setup(self, stage: str):
        if stage == "fit":
            # Loss selection
            self.loss = get_loss_function(self.train_settings['loss'])
            self.train_step_outputs = []
            self.validation_step_outputs = []
        elif stage == "test":
            self.loss = get_loss_function(self.train_settings['loss'])
            self.test_step_outputs = []
        elif stage == "predict":
            self.pred_step_outputs = []

    def forward(self, x, y=None):
        return self.model(x, y)
    
    def one_step(self, x, y):
        pass

    def one_step_classification(self, x, y):
        y = y.long()
        y_hat = self(x)
        self.train_metrics.update(y_hat.cpu(), y.cpu())
        return self.loss(y_hat, y)
    
    def one_step_detection(self, x, y):
        loss_dict = self(x, y)
        loss = sum(loss for loss in loss_dict.values())
        return loss

    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        
        loss = self.one_step(x, y)
        self.log("metrics/batch/train_loss", loss, prog_bar=False)

        return loss

    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        if(self.task == 'detection'):
            metrics = metrics['map']
        self.log(f"metrics/epoch/train_{self.metrics_name}", metrics)
        self.train_metrics.reset()

    def _shared_eval_step(self, batch, batch_idx):
        x, y, images = batch
        y_hat = self(x)
     
        y_pred = [{k: v.cpu() for k, v in t.items()} for t in y_hat]
        targets = [{k: v.cpu() for k, v in t.items()} for t in y]
        images = [(t * 255).to(torch.uint8).cpu() for t in images]

        return images, y_pred, targets, -1
      
    def validation_step(self, batch, batch_idx):
    
        images, y_pred, targets, loss = self._shared_eval_step(batch, batch_idx)
        self.valid_metrics.update(y_pred, targets)
        self.validation_step_outputs.append({"image": images, "predictions": y_pred, "targets": targets, "loss": loss})
        self.log('metrics/batch/val_loss', loss)

    def on_validation_epoch_end(self):
        loss =[outputs['loss'] for outputs in self.validation_step_outputs]
        self.log('metrics/epoch/val_loss', sum(loss) / len(loss))
        self.log(f"metrics/epoch/val_{self.metrics_name}", self.valid_metrics.compute()['map'])
        
        outputs = self.validation_step_outputs[0]
        images, predictions, targets = outputs["image"], outputs["predictions"], outputs["targets"]
       
        if("maskrcnn" in self.architect_settings['backbone']['name']):
            boolean_masks = [out['masks'][out['scores']  > .75] > 0.5 for out in predictions]
            reconstructions = [draw_segmentation_masks(image, mask.squeeze(1), alpha=0.9) 
                                for image, mask in zip(images, boolean_masks)]
        else:
            boxes = [out['boxes'][out['scores'] > .8] for out in predictions]
            reconstructions = [draw_bounding_boxes(image, box, width=10, colors='red')
                                    for image, box in zip(images, boxes)]
        reconstructions = torch.stack([F.interpolate(img.unsqueeze(0), size=(128, 128))
                                        for img in reconstructions]).squeeze(1)
          
        reconstructions = make_grid(reconstructions, nrow= int(self.train_settings['n_batch'] ** 0.5))
        reconstructions = reconstructions.numpy().transpose(1, 2, 0) / 255
        self.logger.experiment["val/reconstructions"].append(File.as_image(reconstructions))

        self.validation_step_outputs.clear()
        self.valid_metrics.reset()


    def test_step(self, batch, batch_idx):

        images, y_pred, targets, loss = self._shared_eval_step(batch, batch_idx)
        self.valid_metrics.update(y_pred, targets) # can reuse valid metrics 
        self.test_step_outputs.append({"image": images, "predictions": y_pred, "targets": targets, "loss": loss})
    
    def on_test_epoch_end(self):
        loss =[outputs['loss'] for outputs in self.test_step_outputs]
        self.log('metrics/epoch/test_loss', sum(loss) / len(loss))
        self.log(f"metrics/epoch/test_{self.metrics_name}", self.valid_metrics.compute()['map'])
         
        self.test_step_outputs.clear()
        self.valid_metrics.reset()
            
    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.train_settings)
        lr_scheduler_config = get_lr_scheduler_config(optimizer, self.train_settings)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
class ClassifierModel(ModelBase):
    def __init__(self, PARAMS):
        super().__init__(PARAMS)
        self.model = WrappingClassifier(PARAMS['architect_settings'])

    def one_step(self, x, y):
        y = y.long()
        y_hat = self(x)
        self.train_metrics.update(y_hat.cpu(), y.cpu())
        return self.loss(y_hat, y)

class DetectorModel(ModelBase):
    def __init__(self, PARAMS):
        super().__init__(PARAMS)
        self.model = WrappingDetector(PARAMS['architect_settings'])
    
    def one_step(self, x, y):
        loss_dict = self(x, y)
        loss = sum(loss for loss in loss_dict.values())
        return loss
    
if __name__ == "__main__":
    
    import yaml
    from pytorch_lightning.loggers import NeptuneLogger
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file', '-c', type=str, required=True, help='Config file'
    )
    args = parser.parse_args()

    with open(args.config_file, 'r') as stream:
        PARAMS = yaml.safe_load(stream)
        print(PARAMS)

    neptune_logger = NeptuneLogger(
            project=PARAMS['logger']['project'],
            # with_id="AIS-113",
            # api_key=PARAMS['logger']['api_key'],
            tags=PARAMS['logger']['tags'],
            log_model_checkpoints=False
        )
    neptune_logger.log_hyperparams(params=PARAMS)
    
    #load data
    data = DataModule(PARAMS['dataset_settings'], PARAMS['training_settings'])

    # create model
    model = Model(PARAMS=PARAMS)
    trainer = get_trainer(PARAMS['training_settings'], neptune_logger)
    # train
    trainer.fit(model, data)
    # test
    trainer.test(model, data)
    