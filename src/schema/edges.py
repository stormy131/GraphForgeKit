from typing import NamedTuple

from encoders import EdgeCreator
from schema.data import EnhancerData


EdgeStrategy = NamedTuple("EdgeStrategy", [
    ("builder", EdgeCreator),
    ("spatial", EnhancerData),
])
