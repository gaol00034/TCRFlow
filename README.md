# TCRFlow
Flow matching based TCR sequence generation
## Framework
![workflow](https://github.com/gaol00034/TCRFlow/blob/main/Figures/model.png)

## Data description
All the TCR-pMHC data are download from [VDJdb](https://vdjdb.cdr3.net/overview), [IEDB](https://www.iedb.org/), [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR) and [MIRA](https://clients.adaptivebiotech.com/pub/covid-2020); 


## Training
Run the example below:
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

if __name__ == "__main__":
    main()
```
## Inference
Run the example below:
```
def main():
    sampling_config = Sampling_config()#load_config(args.config)
    model_config = Model_config()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    from Model.TCRFlow_network import TCRFlow
    model = TCRFlow(model_config).to(device)
    ckpt = torch.load('.../ckpt/best_model.pt', weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    sample_id = 0
    eval_loader = BatchLoader(".../TrainingData/eval_batches")
    n_seed = sampling_config.n_seed
    seed_start = random.randrange(42, 2026)
    flowed_results = []

    sampler = TCRFlowInference(
        model=model,
        config=sampling_config,
        device=device,
    )

    for batch in eval_loader:
        eval_val_batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        val_epitopes = eval_val_batch['Epitopes']#pMHC
        tar_len = eval_val_batch['target_length']
        cond_emb = eval_val_batch["cond"]
        cond_seq_mask = eval_val_batch["cond_seq_mask"]
        #target_attn = eval_val_batch.get("attention_mask", None)

        seed_end = seed_start + n_seed

        pred_ids = sampler.sample_with_seed_range(batch_size=cond_emb.shape[0],
                                                  tar_len=tar_len,
                                                  cond=cond_emb,
                                                  cond_mask=cond_seq_mask,
                                                  seed_start = seed_start,
                                                  seed_end = seed_end,
                                                  seq_length=model_config.max_length,
                                                  latent_dim=model_config.target_encoder_dim,
                                                  )
        pred_cdr3b = id_to_aa(pred_ids)
        val_epitopes = val_epitopes * n_seed
        flowed_results.extend([r for r in zip(pred_cdr3b, val_epitopes)])
        seed_start = seed_end
    flowed_results_df = pd.DataFrame(flowed_results, columns=['flowed_cdr3b', 'Epitope'])
    flowed_results_df.to_csv('.../predicted.csv', index=False)

if __name__ == "__main__":
    main()
```
