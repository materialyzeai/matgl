---
layout: default
title: Benchmarking M3GNet Predictions of Cubic Lattice Parameters.md
nav_exclude: true
---

# Introduction

This notebook is written to demonstrate the use of M3GNet as a structure relaxer as well as to provide more comprehensive benchmarks for cubic crystals based on exp data on Wikipedia and MP DFT data. This benchmark is limited to cubic crystals for ease of comparison since there is only one lattice parameter.

If you are running this notebook from Google Colab, uncomment the next code box to install matgl first.



```python
# !pip install matgl
```


```python
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import matgl
from matgl.ext.ase import Relaxer

warnings.filterwarnings("ignore")
```

The next cell just compiles data from Wikipedia.



```python
data = pd.read_html("http://en.wikipedia.org/wiki/Lattice_constant")[0]
struct_types = [
    "Hexagonal",
    "Wurtzite",
    "Wurtzite (HCP)",
    "Orthorhombic",
    "Tetragonal perovskite",
    "Orthorhombic perovskite",
]
data = data[~data["Crystal structure"].isin(struct_types)]
data = data.rename(columns={"Lattice constant (Å)": "a (Å)"})
data = data.drop(columns=["Ref."])
data["a (Å)"] = data["a (Å)"].map(float)
data = data[["Material", "Crystal structure", "a (Å)"]]


additional_fcc = """10 Ne 4.43 54 Xe 6.20
13 Al 4.05 58 Ce 5.16
18 Ar 5.26 70 Yb 5.49
20 Ca 5.58 77 Ir 3.84
28 Ni 3.52 78 Pt 3.92
29 Cu 3.61 79 Au 4.08
36 Kr 5.72 82 Pb 4.95
38 Sr 6.08 47 Ag 4.09
45 Rh 3.80 89 Ac 5.31
46 Pd 3.89 90 Th 5.08"""

additional_bcc = """3 Li 3.49 42 Mo 3.15
11 Na 4.23 55 Cs 6.05
19 K 5.23 56 Ba 5.02
23 V 3.02 63 Eu 4.61
24 Cr 2.88 73 Ta 3.31
26 Fe 2.87 74 W 3.16
37 Rb 5.59 41 Nb 3.30"""


def add_new(str_, structure_type, df):
    tokens = str_.split()
    new_crystals = []
    for i in range(int(len(tokens) / 3)):
        el = tokens[3 * i + 1].strip()
        if el not in df["Material"].to_numpy():
            new_crystals.append([tokens[3 * i + 1], structure_type, float(tokens[3 * i + 2])])
    df2 = pd.DataFrame(new_crystals, columns=data.columns)
    return pd.concat([df, df2])


data = add_new(additional_fcc, "FCC", data)
data = add_new(additional_bcc, "BCC", data)
data = data[data["Material"] != "NC0.99"]
data = data[data["Material"] != "Xe"]
data = data[data["Material"] != "Kr"]
data = data[data["Material"] != "Rb"]
data = data.set_index("Material")
print(data[61:80])
```


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    Cell In[3], line 1
    ----> 1 data = pd.read_html("http://en.wikipedia.org/wiki/Lattice_constant")[0]
          2 struct_types = [
          3     "Hexagonal",
          4     "Wurtzite",


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/html.py:1226, in read_html(io, match, flavor, header, index_col, skiprows, attrs, parse_dates, thousands, encoding, decimal, converters, na_values, keep_default_na, displayed_only, extract_links, dtype_backend, storage_options)
       1222 check_dtype_backend(dtype_backend)
       1224 io = stringify_path(io)
    -> 1226 return _parse(
       1227     flavor=flavor,
       1228     io=io,
       1229     match=match,
       1230     header=header,
       1231     index_col=index_col,
       1232     skiprows=skiprows,
       1233     parse_dates=parse_dates,
       1234     thousands=thousands,
       1235     attrs=attrs,
       1236     encoding=encoding,
       1237     decimal=decimal,
       1238     converters=converters,
       1239     na_values=na_values,
       1240     keep_default_na=keep_default_na,
       1241     displayed_only=displayed_only,
       1242     extract_links=extract_links,
       1243     dtype_backend=dtype_backend,
       1244     storage_options=storage_options,
       1245 )


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/html.py:979, in _parse(flavor, io, match, attrs, encoding, displayed_only, extract_links, storage_options, **kwargs)
        968 p = parser(
        969     io,
        970     compiled_match,
       (...)    975     storage_options,
        976 )
        978 try:
    --> 979     tables = p.parse_tables()
        980 except ValueError as caught:
        981     # if `io` is an io-like object, check if it's seekable
        982     # and try to rewind it before trying the next parser
        983     if hasattr(io, "seekable") and io.seekable():


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/html.py:237, in _HtmlFrameParser.parse_tables(self)
        229 def parse_tables(self):
        230     """
        231     Parse and return all tables from the DOM.
        232
       (...)    235     list of parsed (header, body, footer) tuples from tables.
        236     """
    --> 237     tables = self._parse_tables(self._build_doc(), self.match, self.attrs)
        238     return (self._parse_thead_tbody_tfoot(table) for table in tables)


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/html.py:789, in _LxmlFrameParser._build_doc(self)
        786 parser = HTMLParser(recover=True, encoding=self.encoding)
        788 if is_url(self.io):
    --> 789     with get_handle(self.io, "r", storage_options=self.storage_options) as f:
        790         r = parse(f.handle, parser=parser)
        791 else:
        792     # try to parse the input in the simplest way


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/common.py:776, in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        773     codecs.lookup_error(errors)
        775 # open URLs
    --> 776 ioargs = _get_filepath_or_buffer(
        777     path_or_buf,
        778     encoding=encoding,
        779     compression=compression,
        780     mode=mode,
        781     storage_options=storage_options,
        782 )
        784 handle = ioargs.filepath_or_buffer
        785 handles: list[BaseBuffer]


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/common.py:405, in _get_filepath_or_buffer(filepath_or_buffer, encoding, compression, mode, storage_options)
        403 # assuming storage_options is to be interpreted as headers
        404 req_info = urllib.request.Request(filepath_or_buffer, headers=storage_options)
    --> 405 with urlopen(req_info) as req:
        406     content_encoding = req.headers.get("Content-Encoding", None)
        407     if content_encoding == "gzip":
        408         # Override compression based on Content-Encoding header


    File ~/repos/matgl/.venv/lib/python3.12/site-packages/pandas/io/common.py:282, in urlopen(*args, **kwargs)
        276 """
        277 Lazy-import wrapper for stdlib urlopen, as that imports a big chunk of
        278 the stdlib.
        279 """
        280 import urllib.request
    --> 282 return urllib.request.urlopen(*args, **kwargs)


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:215, in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        213 else:
        214     opener = _opener
    --> 215 return opener.open(url, data, timeout)


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:521, in OpenerDirector.open(self, fullurl, data, timeout)
        519 for processor in self.process_response.get(protocol, []):
        520     meth = getattr(processor, meth_name)
    --> 521     response = meth(req, response)
        523 return response


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:630, in HTTPErrorProcessor.http_response(self, request, response)
        627 # According to RFC 2616, "2xx" code indicates that the client's
        628 # request was successfully received, understood, and accepted.
        629 if not (200 <= code < 300):
    --> 630     response = self.parent.error(
        631         'http', request, response, code, msg, hdrs)
        633 return response


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:553, in OpenerDirector.error(self, proto, *args)
        551     http_err = 0
        552 args = (dict, proto, meth_name) + args
    --> 553 result = self._call_chain(*args)
        554 if result:
        555     return result


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:492, in OpenerDirector._call_chain(self, chain, kind, meth_name, *args)
        490 for handler in handlers:
        491     func = getattr(handler, meth_name)
    --> 492     result = func(*args)
        493     if result is not None:
        494         return result


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:745, in HTTPRedirectHandler.http_error_302(self, req, fp, code, msg, headers)
        742 fp.read()
        743 fp.close()
    --> 745 return self.parent.open(new, timeout=req.timeout)


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:521, in OpenerDirector.open(self, fullurl, data, timeout)
        519 for processor in self.process_response.get(protocol, []):
        520     meth = getattr(processor, meth_name)
    --> 521     response = meth(req, response)
        523 return response


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:630, in HTTPErrorProcessor.http_response(self, request, response)
        627 # According to RFC 2616, "2xx" code indicates that the client's
        628 # request was successfully received, understood, and accepted.
        629 if not (200 <= code < 300):
    --> 630     response = self.parent.error(
        631         'http', request, response, code, msg, hdrs)
        633 return response


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:559, in OpenerDirector.error(self, proto, *args)
        557 if http_err:
        558     args = (dict, 'default', 'http_error_default') + orig_args
    --> 559     return self._call_chain(*args)


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:492, in OpenerDirector._call_chain(self, chain, kind, meth_name, *args)
        490 for handler in handlers:
        491     func = getattr(handler, meth_name)
    --> 492     result = func(*args)
        493     if result is not None:
        494         return result


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/urllib/request.py:639, in HTTPDefaultErrorHandler.http_error_default(self, req, fp, code, msg, hdrs)
        638 def http_error_default(self, req, fp, code, msg, hdrs):
    --> 639     raise HTTPError(req.full_url, code, msg, hdrs, fp)


    HTTPError: HTTP Error 403: Forbidden


In the next cell, we generate an initial structure for all the phases. The cubic constant is set to an arbitrary value of 5 angstroms for all structures. It does not matter too much what you set it to, but it cannot be too large or it will result in isolated atoms due to the cutoffs used in m3gnet to determine bonds. We then call the Relaxer, which is the M3GNet universal IAP pre-trained on the Materials Project.



```python
predicted = []
mp = []
os.environ["MPRESTER_MUTE_PROGRESS_BARS"] = "true"
mpr = MPRester("YOUR_API_KEY")

# Load the pre-trained M3GNet Potential
pot = matgl.load_model("M3GNet-MP-2021.2.8-PES")
# create the M3GNet Relaxer
relaxer = Relaxer(potential=pot)
for formula, v in data.iterrows():
    formula = formula.split()[0]
    c = Composition(formula)
    els = sorted(c.elements)
    cs = v["Crystal structure"]

    # We initialize all the crystals with an arbitrary lattice constant of 5 angstroms.
    if "Zinc blende" in cs:
        s = Structure.from_spacegroup("F-43m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.25, 0.25, 0.75]])
    elif "Halite" in cs:
        s = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.5, 0, 0]])
    elif "Caesium chloride" in cs:
        s = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), [els[0], els[1]], [[0, 0, 0], [0.5, 0.5, 0.5]])
    elif "Cubic perovskite" in cs:
        s = Structure(
            Lattice.cubic(5),
            [els[0], els[1], els[2], els[2], els[2]],
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]],
        )
    elif "Diamond" in cs:
        s = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5), [els[0]], [[0.25, 0.75, 0.25]])
    elif "BCC" in cs:
        s = Structure(Lattice.cubic(4.5), [els[0]] * 2, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    elif "FCC" in cs:
        s = Structure(
            Lattice.cubic(4.5), [els[0]] * 4, [[0.0, 0.0, 0.0], [0.5, 0.5, 0], [0.0, 0.5, 0.5], [0.5, 0, 0.5]]
        )
    else:
        predicted.append(0)
        mp.append(0)
        continue
    # print(s.composition.reduced_formula)
    relax_results = relaxer.relax(s, fmax=0.01)

    final_structure = relax_results["final_structure"]

    predicted.append(final_structure.lattice.a)

    try:
        mids = mpr.get_material_ids(s.composition.reduced_formula)
        for i in mids:
            try:
                structure = mpr.get_structure_by_material_id(i)
                sga = SpacegroupAnalyzer(structure)
                sga2 = SpacegroupAnalyzer(final_structure)
                if sga.get_space_group_number() == sga2.get_space_group_number():
                    conv = sga.get_conventional_standard_structure()
                    mp.append(conv.lattice.a)
                    break
            except Exception:
                pass
        else:
            raise RuntimeError
    except Exception:
        mp.append(0)

data["MP a (Å)"] = mp
data["Predicted a (Å)"] = predicted
```


```python
data["% error vs Expt"] = (data["Predicted a (Å)"] - data["a (Å)"]) / data["a (Å)"]
data["% error vs MP"] = (data["Predicted a (Å)"] - data["MP a (Å)"]) / data["MP a (Å)"]
```


```python
data.sort_index().style.format({"% error vs Expt": "{:,.2%}", "% error vs MP": "{:,.2%}"}).background_gradient()
```


```python
data["% error vs MP"].replace([np.inf, -np.inf], np.nan).dropna().hist(bins=20)
```


```python
# This generates a pretty markdown table output.

# df = data.sort_values("% error vs MP", key=abs).replace([np.inf, -np.inf], np.nan).dropna()
# df["% error vs MP"] = [f"{v*100:.3f}%" for v in df["% error vs MP"]]
# df["% error vs Expt"] = [f"{v*100:.3f}%" for v in df["% error vs Expt"]]
# print(df.to_markdown())
```