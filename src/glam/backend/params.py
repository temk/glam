from dacite import DaciteError
from dacite import from_dict as dacite_from_dict
from typing import TypeVar

from glam.common.errors import GlamError

T = TypeVar("T")


def parse_params(params: dict, data_class: type[T], error_cls: type[GlamError], label: str) -> T:
    """Deserialize a service's `params` mapping into a backend's typed config dataclass."""
    try:
        return dacite_from_dict(data_class=data_class, data=params)
    except DaciteError as e:
        raise error_cls(f"invalid params for {label} service: {e}") from e
