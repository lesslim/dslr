import operator
from typing import (
    Any, Callable, Collection, Dict, Iterable, List, Mapping, Optional, Tuple,
    Union
)

from .i_hate_42 import max_, min_


class Series(Collection):
    _default_values: Dict[type, Any] = {float: float("NaN")}

    def __init__(self,
                 data: Optional[Iterable] = None,
                 dtype: Optional[type] = None,
                 use_default_values: bool = True,
                 try_convert_strings: bool = False):
        """
        Create a new Series object with the data and dtype provided.
        If dtype is not provided, it is deduced to be the type of the first
        element of the data. Default dtype is float.
        """
        self._use_default_values = use_default_values

        self._data: List[Any] = []
        self._dtype: type = float if dtype is None else dtype

        if data is not None:
            self._data, self._dtype = self._convert_dtype_iter(
                data, dtype, try_convert_strings)

        if self._dtype not in {bool, float, int, str}:
            raise NotImplementedError(f"Unsupported dtype: {self._dtype}")

    #
    #  Properties =============================================================
    #

    @property
    def dtype(self) -> type:
        return self._dtype

    @property
    def size(self) -> int:
        return len(self)

    @property
    def values(self) -> "Series":
        return self

    #
    #  Public API =============================================================
    #

    def copy(self) -> "Series":
        return Series._from_data(self._data.copy(), self._dtype)

    def item(self) -> Any:
        """Returns the only item as a python object"""
        if len(self) != 1:
            raise ValueError("item() can only be used with Series of length 1")
        return self._data[0]

    def dropna(self) -> "Series":
        """
        Returns a new Series with NaN values dropped.
        If the dtype is not float, returns self.
        """
        if self._dtype == float:
            return self[self == self]
        return self

    def max(self, **kwargs) -> Any:
        """kwargs to keep compatibility with pandas"""
        return max_(self._data)

    def mean(self) -> Any:
        if self._dtype not in {int, float}:
            raise ValueError(f"Not defined for Series of dtype {self._dtype}")
        if not len(self):
            raise ValueError("Not defined for empty Series")
        return sum(self._data) / len(self)

    def min(self, **kwargs) -> Any:
        """kwargs to keep compatibility with pandas"""
        return min_(self._data)

    def map(self,
            arg: Union[Callable, Mapping, "Series"],
            na_action: Optional[str] = None
            ) -> "Series":
        """
        Map values of Series according to input correspondence. Used for
        substituting each value in a Series with another value, that may be
        derived from a function, a dict or a Series.
        """
        skipna = False
        if na_action is not None:
            if na_action not in {"ignore"}:
                raise ValueError(f"Invalid value of na_action={na_action}")
            skipna = (self._dtype == float)
        if isinstance(arg, Series):
            if len(arg) != len(self):
                raise ValueError(f"Size mismatch, got {len(arg)}, {len(self)}")
            return arg.copy() if not skipna else arg[self == self]
        if callable(arg):
            return Series([arg(elem) for elem in self._data
                           if not skipna or elem == elem])
        if isinstance(arg, Mapping):
            return Series([arg[elem] for elem in self._data
                           if not skipna or elem == elem])
        raise ValueError(f"Cannot map with {type(arg)}")

    def percentile(self, rank: float) -> Any:
        if not len(self):
            raise ValueError("Not defined for empty Series")
        if not 0 < rank < 1:
            raise ValueError("Rank must be in range [0, 1]")
        sorted_self = self.sort()
        return sorted_self[int(rank * len(sorted_self))].item()

    def sort(self, in_place: bool = False) -> "Series":
        if in_place:
            self._data.sort()
            return self
        return Series._from_data(sorted(self._data), self._dtype)

    def std(self) -> Any:
        mean = self.mean()
        return (sum((e - mean) ** 2 for e in self) / len(self)) ** 0.5

    #
    #  Magic methods (except operators) =======================================
    #

    def __contains__(self, value: Any) -> bool:
        return value in self._data

    def __getitem__(self, key: Union[int, slice, "Series"]) -> Any:
        index = self._get_index(key)
        data = [self._data[i] for i in index]
        return Series._from_data(data, self._dtype)

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return (f"bears.Series, len={len(self)}, dtype={self._dtype}:"
                f"\n{self._data}")

    def __setitem__(self,
                    key: Union[int, slice, "Series"],
                    value: Any) -> None:
        index = self._get_index(key)

        if isinstance(value, Iterable) and not isinstance(value, str):
            data, _ = self._convert_dtype_iter(value, self._dtype)
            if len(data) != len(index):
                raise ValueError(f"Expected exactly {len(self)} elements,"
                                 f" received {len(data)}")
            for i, v in zip(index, data):
                self._data[i] = v

        else:
            if not isinstance(value, self._dtype):
                value = self._convert_dtype(value, self._dtype)
            for i in index:
                self._data[i] = value

    def __str__(self) -> str:
        return (f"{self._data}")

    #
    #  Operators ==============================================================
    #

    def __add__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.add)

    def __and__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.and_)

    def __divmod__(self, other: Any) -> Tuple["Series", "Series"]:
        return self.__floordiv__(other), self.__mod__(other)

    def __eq__(self, other: Any) -> "Series":  # type: ignore # mypy wants bool
        return self._binary_op(other, operator.eq)

    def __floordiv__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.floordiv)

    def __ge__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.ge)

    def __gt__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.gt)

    def __index__(self) -> int:
        if self._dtype != int:
            raise ValueError(f"Series has dtype={self._dtype}, expected int")
        return self.item()

    def __invert__(self) -> "Series":
        return Series((e.__invert__() for e in self._data), self._dtype)

    def __le__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.le)

    def __lt__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.lt)

    def __mod__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.mod)

    def __mul__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.mul)

    def __ne__(self, other: Any) -> "Series":  # type: ignore # mypy wants bool
        return self._binary_op(other, operator.ne)

    def __or__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.or_)

    def __pow__(self, other: Any, modulo: Any = 1) -> "Series":
        return self._binary_op(other, operator.pow)

    def __sub__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.sub)

    def __truediv__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.truediv)

    def __xor__(self, other: Any) -> "Series":
        return self._binary_op(other, operator.xor)

    #
    #  Private methods ========================================================
    #

    def _binary_op(self,
                   other: Any,
                   operator: Callable[[Any, Any], Any]) -> "Series":
        """
        Is a higher-order polymorphic function, so no proper type hints...
        """
        if isinstance(other, Series):
            if len(self) != len(other):
                raise ValueError(f"Size mismatch: {len(self)} vs {len(other)}")
            data = [operator(lhs, rhs) for lhs, rhs in zip(self, other)]
        else:
            data = [operator(lhs, other) for lhs in self]
        return Series._from_data(data, Series._deduce_dtype(data))

    def _convert_dtype_iter(self,
                            data: Iterable,
                            dtype: Optional[type] = None,
                            try_convert_strings: bool = False
                            ) -> Tuple[List, type]:
        out = []
        for elem in data:
            if dtype is None:
                dtype = type(elem)
                if dtype == str and try_convert_strings:
                    dtype = Series._dtype_from_str(elem)
            if isinstance(elem, dtype):
                out.append(elem)
            else:
                out.append(self._convert_dtype(elem, dtype))
        if dtype is None:
            dtype = float
        return out, dtype

    def _convert_dtype(self, item: Any, dtype: type) -> Any:
        try:
            return dtype(item)
        except ValueError:
            if (self._use_default_values
                    and dtype in Series._default_values):
                return Series._default_values[dtype]
            else:
                raise TypeError(f"Couldn't convert {item} to {dtype}.")

    @staticmethod
    def _deduce_dtype(data: Iterable) -> type:
        if not data:
            return float
        return type(next(iter(data)))

    @staticmethod
    def _from_data(data: List, dtype: type) -> "Series":
        """
        Creates a new Series without copying (and checking) data.
        """
        out = Series(dtype=dtype)
        out._data = data
        return out

    @staticmethod
    def _dtype_from_str(item: str) -> type:
        try:
            out: type = type(int(item))
        except ValueError:
            try:
                out = type(float(item))
            except ValueError:
                return str
        return out

    def _get_index(self,
                   key: Union[int, slice, "Series"]) -> Collection[int]:
        if isinstance(key, int):
            return (key,)
        if isinstance(key, slice):
            return list(range(len(self))[key])
        if isinstance(key, Series):
            if not len(key):
                return []
            if key.dtype == bool:
                if len(key) != len(self):
                    raise IndexError(f"Expected Series of size {len(self)}")
                return [i for i, value in enumerate(key) if value]
            if key.dtype == int:
                return key._data  # type: ignore
            raise IndexError(f"Can't index with Series of dtype {key.dtype}")
        raise IndexError(f"Can't index with {type(key)}")
