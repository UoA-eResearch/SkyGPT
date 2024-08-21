import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import ddp

from vqvae import VQVAE
from data import VideoData
import torch

def main():
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VQVAE.add_model_specific_args(parser)
    parser.add_argument('--data_path', type=str, default = '/mnt/SkyGPT/codes/video_prediction/SkyGPT/script/GPT_benchmark.hdf5') #default='/scratch/groups/abrandt/GAN_project/models/VideoGPT/GPT_full_2min.hdf5')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    data = VideoData(args)
    # pre-make relevant cached files if necessary
    data.train_dataloader()
    data.test_dataloader()
    model = VQVAE(args)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/recon_loss', mode='min'))

    kwargs = dict()

    #set some args
    args.accelerator = 'cuda'
    args.devices = torch.cuda.device_count()

    if args.devices < 1:
        print("ERROR: No CUDA GPUs found")
    else:
        print(f"CUDA GPUs: {args.devices}")

    args.gpus = -1
    # Some params later will fail, ignore the warning during running
    args.strategy = ddp.DDPStrategy(find_unused_parameters=True)

    print(args)
    print(kwargs)

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, max_steps = 2000, max_epochs = 2, **kwargs)
    trainer.fit(model, data)


if __name__ == '__main__':
    main()

