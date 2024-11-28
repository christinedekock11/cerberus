from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.pl_trainer import DataModulePL, MatFactPL
from configs import pl_config as train_config

if __name__ == '__main__':
    train_config['name'] = datetime.now().strftime('%H%M%d%m%y')
    logger = TensorBoardLogger("logs/", name=f"matfact_{train_config['name']}")

    # DataModule for both A and C sources
    dm = DataModulePL(train_path_C=train_config['C_path'],
                      train_path_A=train_config['A_path'],
                      batch_size=train_config['batch_size'])
    dm.prepare_data()
    dm.setup()

    # Model
    model = MatFactPL(N=dm.shape_A[0], M=dm.shape_A[1],
                      D=dm.shape_C[1], T=dm.T,
                      **train_config)

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',  dirpath='models/', filename='{train_config["name"]}-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1, mode='min', save_weights_only=True, every_n_epochs=100)
    es = pl.callbacks.EarlyStopping(monitor='val_loss', patience=train_config['early_stopping_p'],
                                    min_delta=train_config['early_stopping_tol'])
    progbar = pl.callbacks.RichProgressBar(leave=True)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=train_config['num_epochs'],
        devices=train_config['devices'],
        accelerator=train_config['accelerator'],
        enable_progress_bar=True,
        callbacks=[es, checkpoint_callback, progbar],
        logger=logger)

    # Fit the model
    trainer.fit(model, datamodule=dm)