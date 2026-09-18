# TCRFlow
Flow matching based TCR sequence generation
## Framework
![workflow](https://github.com/gaol00034/TCRFlow/blob/main/Figures/model.png)
***
## Data description
All the TCR-pMHC data are download from [VDJdb](https://vdjdb.cdr3.net/overview), [IEDB](https://www.iedb.org/), [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR) and [MIRA](https://clients.adaptivebiotech.com/pub/covid-2020); 

***
## Training
Run the above script example:
```
import torch
import numpy as np
import random
from models.TCRFlow_network import TCRFlow
from Scripts.train import Trainer
import logging
from Dataprocessing.batch_loader import BatchLoader

if __name__ == "__main__":
    #load configs
    config = Model_config()
    train_config = Train_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TCRFlow(config).to(device)
    logging.basicConfig(
        filename='.../logs/log.log',
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )
    train_loader = BatchLoader(".../TrainingData/train_batches", shuffle=True)
    valid_loader = BatchLoader(".../TrainingData/valid_batches", shuffle=True)
    trainer = Trainer(model, logging.getLogger(__name__), train_config, save_dir='.../ckpt/', save_model_name='finalmodel.pt', device=device)
    trainer.train(train_loader=train_loader, val_loader=valid_loader)
```
