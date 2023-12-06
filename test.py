from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl


@hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    model = NeRFModel.load_from_checkpoint('model.ckpt')
    datamodule = hydra.utils.instantiate(opt.dataset, train=False)
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)
    result = trainer.test(model, datamodule=datamodule)[0]


if __name__ == "__main__":
    main()
