from animatableGaussian.model.nerf_model import NeRFModel
import hydra
import pytorch_lightning as pl


@hydra.main(config_path="./confs", config_name="gala", version_base="1.1")
def main(opt):
    pl.seed_everything(0)

    model = NeRFModel(opt)
    datamodule = hydra.utils.instantiate(opt.dataset)
    trainer = pl.Trainer(accelerator='gpu',
                         **opt.trainer_args)

    trainer.fit(model, datamodule=datamodule)
    result = trainer.test(model, datamodule=datamodule)[0]
    trainer.save_checkpoint('model.ckpt')


if __name__ == "__main__":
    main()
