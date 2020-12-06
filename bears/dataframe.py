from collections import defaultdict
import csv
import os
from typing import Any, Collection, Dict, List, Optional, Tuple, Union

from .series import Series
from .i_hate_42 import min_


class DataFrame():
    """
    A DataFrame is an aggregation of Series objects
    """
    def __init__(self,
                 cols: List[Series],
                 names: Optional[Collection[str]] = None):
        self._cols = cols
        self._nrows = 0 if len(cols) == 0 else len(cols[0])
        self._names = ({} if names is None
                       else {name: i for i, name in enumerate(names)})
        self._id2names = ["" for _ in range(len(self._names))]
        for name, i in self._names.items():
            self._id2names[i] = name

    @property
    def columns(self):
        if self._names:
            return Series(self._id2names, str)
        return Series(range(len(self)), int)

    def append(self, df: "DataFrame"):
        if (self._names != df._names
                or self._id2names != df._id2names
                or not all(c1.dtype == c2.dtype
                           for c1, c2 in zip(self._cols, df._cols))):
            raise ValueError("Incompatible DataFrames")
        for c1, c2 in zip(self._cols, df._cols):
            c1._data.extend(c2._data)
        self._nrows += df._nrows

    def describe(self):
        features = self._get_numeric_data()
        cols = [col.dropna() for col in features._cols]
        if features._names:
            names = features._id2names
        else:
            names = [f"f{i}" for i in len(features)]
        to_retrieve = [
            ("Count", "__len__", ()),
            ("Mean", "mean", ()),
            ("Std", "std", ()),
            ("Min", "min", ()),
            ("25%", "percentile", (0.25,)),
            ("50%", "percentile", (0.50,)),
            ("75%", "percentile", (0.75,)),
            ("Max", "max", ())
        ]
        data = ["", *names]
        field_str = "{:>{w}.{w}}"
        value_str = "{:>{w}.1f}"
        out = ["|" + (field_str + "|") * (len(cols) + 1) + "\n"]
        for row_name, func, args in to_retrieve:
            data.append(row_name)
            out.append("|" + field_str + "|")
            data.extend(float(col.__getattribute__(func)(*args))
                        for col in cols)
            out.append((value_str + "|") * len(cols) + "\n")
        try:
            _, w = os.popen('stty size', 'r').read().split()
        except ValueError:
            w = 140
        print("".join(out).format(*data, w=int(w) // (len(cols) + 3)))

    def dropna(self):
        """
        Returns a new DataFrame where all the rows containing NaN are dropped.
        """
        index = None
        for col in self._cols:
            if col.dtype != float:
                continue
            mask = (col == col)
            if index is None:
                index = mask
            else:
                index = (index & mask)
        return self[index]

    def isna(self):
        cols = [(col == col) for col in self._cols]
        names = self._id2names if self._names else None
        return DataFrame(cols, names)

    @staticmethod
    def read_csv(filepath: str,
                 sep: str = ',',
                 header: bool = True,
                 names: Optional[Collection[str]] = None,
                 dtype: Union[type, Dict[Union[int, str], type]] = None,
                 skiprows: int = 0,
                 nrows: int = None,
                 encoding: str = "utf8") -> "DataFrame":
        """
        Guess what, this method reads a csv into a DataFrame.

        Args:
            filepath: the path to the csv file
            sep: the separator used in said file
            header: whether the csv file has a header
            names: what names should be used for columns. Must not be set if
                header is True.
            dtype: column-type mapping
            skiprows: the number of rows to skip BEFORE the header (if any)
            nrows: the maximum number of rows to retrieve.
            encoding: the encoding used in the csv file
        """
        if names is not None and header is True:
            raise ValueError("'names' and 'header' are incompatible")

        with open(filepath, mode="r", encoding=encoding) as f:
            csvdata = csv.reader(f, delimiter=sep)

            for i in range(skiprows):
                next(csvdata)

            if header:
                names = next(csvdata)

            cols: List[List[str]] = []
            for row_i, row in enumerate(csvdata):
                if row_i == 0:
                    cols.extend([] for _ in row)
                if nrows is not None and row_i == nrows:
                    break
                if len(row) != len(cols):
                    raise ValueError(f"Invalid number of columns at row "
                                     f"{skiprows + row_i + int(header)}")
                for i in range(len(row)):
                    cols[i].append(row[i])

            if isinstance(dtype, type):
                dtype_map = defaultdict(lambda: dtype)  # type: ignore
            elif isinstance(dtype, Dict):
                dtype_map = defaultdict(lambda: None, dtype)
                if names is not None:
                    dtype_map.update({i: dtype_map[name]
                                      for i, name in enumerate(names)
                                      if i not in dtype_map})
            else:
                dtype_map = defaultdict(lambda: None)

            processed_cols = []
            for col_i, col in enumerate(cols):
                processed_cols.append(Series(col,
                                             dtype_map[col_i],  # type: ignore
                                             try_convert_strings=True))

            return DataFrame(processed_cols, names=names)

    def __getitem__(self, key: Any) -> Union[Series, "DataFrame"]:
        from_cols, index = self._get_index(key)
        if from_cols:
            cols = [self._cols[i] for i in index]
            if not self._names:
                names: Optional[List[str]] = None
            else:
                names = [self._id2names[i] for i in index]
            if len(cols) == 1:
                return cols[0]
            return DataFrame(cols, names)
        else:
            index_series = Series._from_data(index, int)
            cols = [col[index_series] for col in self._cols]
            return DataFrame(cols, self._id2names if self._names else None)

    def __invert__(self) -> "DataFrame":
        cols = [col.__invert__() for col in self._cols]
        names = self._id2names if self._names else None
        return DataFrame(cols, names)

    def __len__(self) -> int:
        return len(self._cols)

    def __repr__(self) -> str:
        out = [f"bears.DataFrame, ncols={len(self)}, nrows={self._nrows}:\n"
               f"{'no col names' if not self._names else self._id2names}"]
        for i in range(min_(self._nrows, 10)):
            out.append(str(self._get_row(i)))
        if len(out) - 1 < self._nrows:
            out.append(". . .")
            out.append(str(self._get_row(-1)))
        return "\n".join(out)

    def __setitem__(self, idx: Union[int, str], value: Series):
        """ Only changing one column at a time is supported for now """
        if isinstance(idx, int):
            numeric_id = idx
        elif isinstance(idx, str):
            numeric_id = self._names[idx]
        else:
            raise ValueError(f"Unrecognized argument type: {type(idx)}")
        if not isinstance(value, Series):
            raise ValueError("Expected a Series")
        if len(value) != self._nrows:
            raise ValueError(f"Expected a Series of size {self._nrows}")
        self._cols[numeric_id] = value.copy()

    def _get_index(self, key: Any) -> Tuple[bool, List[int]]:
        """
        Returns (from_cols, indices), where if from_cols is True, the indices
        are column indices (row indices otherwise)
        """
        if isinstance(key, int):
            return True, [key]
        if isinstance(key, slice):
            return True, list(range(len(self))[key])
        if isinstance(key, str):
            return True, [self._names[key]]
        if isinstance(key, Series):
            if key.dtype == bool:
                if len(key) != self._nrows:
                    raise IndexError(f"Expected Series of size {len(self)}")
                return False, [i for i, value in enumerate(key) if value]
            if key.dtype == int:
                return False, key._data
            if key.dtype == str:
                return True, [self._names[k] for k in key]
            raise IndexError(f"Can't index with Series of dtype {key.dtype}")
        if isinstance(key, Collection):
            if len(key) == 0:
                return True, []
            if isinstance(key[0], Series):  # type: ignore
                key = [k.item() for k in key]
            if isinstance(key[0], int):
                return True, list(key)
            if isinstance(key[0], str):
                return True, [self._names[k] for k in key]
        raise IndexError(f"Can't index with {type(key)}")

    def _get_numeric_data(self) -> Union[Series, "DataFrame"]:
        """
        Because using undocumented private API is so much FUN, isn't it,
        matplotlib developers?
        """
        cols = [i for i in range(len(self))
                if self._cols[i].dtype in {int, float}]
        return self[cols]

    def _get_row(self,
                 row_id: int) -> List:
        return [col[row_id].item() for col in self._cols]
