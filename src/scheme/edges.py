from typing import NamedTuple

from encoders import EdgeCreator
from scheme.data import EnhancerData


EdgeStrategy = NamedTuple("EdgeStrategy", [
    ("builder", EdgeCreator),
    ("spatial", EnhancerData),
])
