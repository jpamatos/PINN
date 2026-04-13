import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from configs import Config
from data.protocol import DataLoaderProtocol
from pinn.protocol import PINNProtocol
from training.trainer import Trainer

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(config_path=None, config_name="base_config", version_base=None)
def main(cfg: Config) -> None:
    data_loader: DataLoaderProtocol = instantiate(cfg.data, _recursive_=False)
    pinn: PINNProtocol = instantiate(cfg.model, _recursive_=False)

    trainer: Trainer = instantiate(cfg.trainer)
    model = trainer.train(model=pinn, data_loader=data_loader)


if __name__ == "__main__":
    main()
