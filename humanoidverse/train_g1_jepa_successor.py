"""Train the BeJEPA G1 policy and latent world model."""

import json

from humanoidverse.distributed_launch import isolate_local_rank_visible_device


def build_train_config(cfg=None):
    from humanoidverse.train import TrainConfig
    from humanoidverse.train_g1_jepa_successor_config import G1JepaSuccessorConfig

    options = cfg or G1JepaSuccessorConfig()
    return TrainConfig.model_validate_json(json.dumps(options.to_dict()))


def main():
    isolate_local_rank_visible_device()
    import tyro

    from humanoidverse.train_g1_jepa_successor_config import G1JepaSuccessorConfig

    options = tyro.cli(G1JepaSuccessorConfig)
    if options.print_config:
        print(json.dumps(options.to_dict(), indent=2))
        return
    options.validate_data()
    cfg = build_train_config(options)
    cfg.build().train()


if __name__ == "__main__":
    main()
