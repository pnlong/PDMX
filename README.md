[![GitHub license](https://img.shields.io/github/license/pnlong/PDMX)](https://github.com/pnlong/PDMX/blob/master/LICENSE)

# PDMX: A Large-Scale *P*ublic *D*omain *M*usic*X*ML Dataset for Symbolic Music Processing

Recent [copyright infringement lawsuits against leading music generation companies](https://www.riaa.com/record-companies-bring-landmark-cases-for-responsible-ai-againstsuno-and-udio-in-boston-and-new-york-federal-courts-respectively) have sent shockwaves throughout the AI-Music community, highlighting the need for copyright-free training data. Meanwhile, the most prevalent format for symbolic music processing, MIDI, is well-suited for modeling sequences of notes but omits an abundance of extra musical information present in sheet music, which the MusicXML format addresses. To mitigate these gaps, we present **[PDMX]()**: a large-scale open-source dataset of over 250K public domain MusicXML scores. We also introduce `MusicRender`, an extension of the Python library [MusPy](https://hermandong.com/muspy/doc/muspy.html)'s universal `Music` object, designed specifically to handle MusicXML.

---

## Installation

To access the functionalities that we introduce, please clone the latest version of this [repository](https://github.com/pnlong/PDMX). Then, install relevant dependencies to the Conda environment `my_env` with `conda env update -n my_env --file environment.yml`.

### TL;DR

```
git clone https://github.com/pnlong/PDMX.git
conda env update -n my_env --file PDMX/environment.yml
conda activate my_env
```



## Important Methods

We present a few important contributions to interact with both the PDMX dataset and MusicXML-like files.

### `MusicRender`

We introduce `MusicRender`, an extension of [MusPy](https://hermandong.com/muspy/doc/muspy.html)'s universal `Music` object, that can hold musical performance directives through its `annotations` field.

```python
from pdmx import MusicRender
```

Let's say `music` is a `MusicRender` object. We can write `music` to various output formats, where the output filetype is inferred from the filetype of `path` (`.wav` is audio, `.midi` is symbolic).

```python
music.write(path = path)
```

We can also save `music` to a JSON file at the location `path`.

```python
music.save_json(path = path)
```

### `load_json()`

We store PDMX as JSONified `MusicRender` objects (see the `save_json()` method above). We can reinstate these objects into Python by reading them with the `load_json()` function, which returns a `MusicRender` object given the path to the JSON file.

```python
from pdmx import load_json
music = load_json(path = path)
```

### `read_musescore()`

PDMX was created by scraping the public domain content of [MuseScore](https://musescore.com), a score-sharing online platform on which users can upload their own sheet music arrangements in a MusicXML-like format. MusPy alone lacked the ability to fully parse these files. Our `read_musescore()` function can, and returns a `MusicRender` object given the path to the MuseScore file.

```python
from pdmx import read_musescore
music = read_musescore(path = path)
```



## Citing & Authors

If you find this repository helpful, feel free to cite our publication [PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing]():

```bibtex
@inproceedings{long2024pdmx,
    title = "PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing",
    author = "Long, Phillip and Novack, Zachary and Berg-Kirkpatrick, Taylor and McAuley, Julian",
    booktitle = "",
    month = "9",
    year = "2024",
    publisher = "",
    url = "",
}
```

