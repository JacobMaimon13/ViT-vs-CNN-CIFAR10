import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.cnn import SimpleCNN
from src.models.vit import ViTLightningModule
from src.utils.data_loader import get_cifar10_loaders

def main(args):
    pl.seed_everything(42)
    
    # Data
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)

    # Model
    if args.model == 'cnn':
        model = SimpleCNN(lr=args.lr)
        project_name = "CNN-CIFAR10"
    else:
        model = ViTLightningModule(lr=args.lr)
        project_name = "ViT-CIFAR10"

    # Logger & Callbacks
    wandb_logger = WandbLogger(project=project_name, log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1
    )

    # Train
    print(f"Starting training for {args.model.upper()}...")
    trainer.fit(model, train_loader, val_loader)

    # Test
    print("Evaluating on test set...")
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='vit', choices=['cnn', 'vit'], help='Model type')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    args = parser.parse_args()
    main(args)
