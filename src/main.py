from pathlib import Path

from .enhancer import Enhancer
from schema.gnn_build import GNNConfig


def main():
    config = GNNConfig(
        input_size=...,
        output_size=...,
        
        activation=...,
        activation_args={},

        conv_operator=...,
        conv_args={},

        encoder_scheme=[],
        predictor_scheme=[],
    )

    test = Enhancer(config, None, Path("./enhancer_cache"))
    test.run_compare()


if __name__ == "__main__":
    main()
