from typing import NamedTuple

import numpy as np


# TODO: input config validation
CONFIG_FILE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "structure": {
                "type": "object",
                "properties": {
                    "encoder": {
                        "type": "array",
                        "items": {"type": "number"}
                    },
                    "estimator": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                }
            },
            "convolution": {
                "type": "string",
                "enum": [],
            },
            "edges": {
                "type": "string",
                "enum": [],
            },
            "data": { "type": "string" },
        }
    }
}

EnhancerData = NamedTuple("EnhancerData", [
    ("features", np.ndarray),
    ("target", np.ndarray),
    ("spatial", np.ndarray),
])
