[![arXiv](https://img.shields.io/badge/cs.SD-2409%3A10831-b31b1b?style=flat&logo=arxiv&logoColor=b31b1b&logoSize=auto)](https://arxiv.org/abs/2409.10831/)
[![Zenodo](https://img.shields.io/badge/Dataset-10.5281%2Fzenodo.14984509-blue?style=flat&logo=zenodo&logoColor=blue&logoSize=auto)](https://zenodo.org/records/14984509)
[![GitHub license](https://img.shields.io/github/license/pnlong/PDMX?style=flat)](https://github.com/pnlong/PDMX/blob/master/LICENSE)

# PDMX: A Large-Scale *P*ublic *D*omain *M*usic*X*ML Dataset for Symbolic Music Processing

![Public Domain MusicXML](./PDMX.png "PDMX")

Recent [copyright infringement lawsuits against leading music generation companies](https://www.riaa.com/record-companies-bring-landmark-cases-for-responsible-ai-againstsuno-and-udio-in-boston-and-new-york-federal-courts-respectively) have sent shockwaves throughout the AI-Music community, highlighting the need for copyright-free training data. Meanwhile, the most prevalent format for symbolic music processing, MIDI, is well-suited for modeling sequences of notes but omits an abundance of extra musical information present in sheet music, which the MusicXML format addresses. To mitigate these gaps, we present **[PDMX](https://arxiv.org/abs/2409.10831/)**: a large-scale open-source dataset of over 250K public domain MusicXML scores. We also introduce `MusicRender`, an extension of the Python library [MusPy](https://hermandong.com/muspy/doc/muspy.html)'s universal `Music` object, designed specifically to handle MusicXML. The dataset, and further specifics, can be downloaded on [Zenodo](https://zenodo.org/records/14984509).

---

## Updates :star:

- **Copyright License Discrepancy**: Upon further use of the PDMX dataset, we discovered a discrepancy between the public-facing copyright metadata on the [MuseScore website](https://musescore.com/) and the internal copyright data of the MuseScore files themselves, which affected 12.29% of (31,221) songs. We have decided to proceed with the former given its public visibility on Musescore (i.e. this is what the MuseScore website presents its users with). We have noted files with conflicting internal licenses in the `license_conflict` column of PDMX on [Zenodo](https://zenodo.org/records/14984509). We recommend using the `no_license_conflict` subset of PDMX moving forward.

- **New MXL, PDF, and MID Files**: For each song in PDMX, we not only provide the `MusicRender` and metadata JSON files, but we also try to include associated compressed MusicXML (MXL), sheet music (PDF), and MIDI (MID) files when available. Due to the corruption of 42 of the original MuseScore files, these songs lack those associated files (since they could not be converted to those formats) and only include the `MusicRender` and metadata JSON files. The `all_valid` subset of PDMX describes the songs where all associated files are valid.

---

## Installation

To access the functionalities that we introduce, please clone the latest version of this repository. Then, install relevant dependencies to the Conda environment `my_env` with `conda env update -n my_env --file environment.yml`. Alternatively, you can use `pip` to create a virtual environment with `pip install -r requirements.txt`.

### TL;DR

With *Conda*:

```
git clone https://github.com/pnlong/PDMX.git
cd PDMX
conda create -n my_env python=3.10
conda env update -n my_env --file environment.yml
conda activate my_env
```

With `pip`:

```
git clone https://github.com/pnlong/PDMX.git
cd PDMX
python3 -m my_env .my_env
source .my_env/bin/activate
pip install -r requirements.txt
```

## Important Methods

We present a few important contributions to interact with both the PDMX dataset and MusicXML-like files.

### `MusicRender`

We introduce `MusicRender`, an extension of [MusPy](https://hermandong.com/muspy/doc/muspy.html)'s universal `Music` object, that can hold musical performance directives through its `annotations` field.

```python
from pdmx import MusicRender
```

Let's say `music` is a `MusicRender` object. We can save `music` to a JSON or YAML file at the location `path`:

```python
music.save(path = path)
```

However, we could just as easily use `write()`, where `path` ends with `.json` or `.yaml`. The benefit of this method is that we can write `music` to various other output formats, where the output filetype is inferred from the filetype of `path` (`.wav` is audio, `.midi` is symbolic).

```python
music.write(path = path)
```

When writing to audio or symbolic formats, performance directive (e.g. dynamics, tempo markings) are realized to their fullest extent. This functionality should not be confused with the `music.realize_expressive_features()` method, which realizes the directives inside a `MusicRender` object. This method should not be used explicitly before writing, as it is implicitly called during that process and any directives will be doubly applied.

### `load()`

We store PDMX as JSONified `MusicRender` objects (see the `write()` or `save()` methods above). We can reinstate these objects into Python by reading them with the `load()` function, which returns a `MusicRender` object given the path to a JSON or YAML file.

```python
from pdmx import load
music = load(path = path)
```

### `read_musescore()`

PDMX was created by scraping the public domain content of [MuseScore](https://musescore.com), a score-sharing online platform on which users can upload their own sheet music arrangements in a MusicXML-like format. MusPy alone lacked the ability to fully parse these files. Our `read_musescore()` function can, and returns a `MusicRender` object given the path to the MuseScore file.

```python
from pdmx import read_musescore
music = read_musescore(path = path)
```



## Citing & Authors

If you find this repository or our dataset helpful, feel free to cite our publication [PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing](https://arxiv.org/abs/2409.10831/):

```bibtex
@inproceedings{long2024pdmx,
  author={Long, Phillip and Novack, Zachary and Berg-Kirkpatrick, Taylor and McAuley, Julian},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={PDMX: A Large-Scale Public Domain MusicXML Dataset for Symbolic Music Processing}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Filtering;Data integrity;Companies;Signal processing;Filtering algorithms;Data collection;Explosions;Multiple signal classification;Speech processing;Capacity planning;symbolic music datasets;symbolic music generation;music copyright},
  doi={10.1109/ICASSP49660.2025.10890217}
}
```

Additionally, if you use the PDMX dataset, please also cite the publication [Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset](https://arxiv.org/abs/2410.02084):

```bibtex
@article{xu2024generating,
    title={Generating Symbolic Music from Natural Language Prompts using an LLM-Enhanced Dataset},
    author={Xu, Weihan and McAuley, Julian and Berg-Kirkpatrick, Taylor and Dubnov, Shlomo and Dong, Hao-Wen},
    journal={arXiv preprint arXiv:2410.02084},
    year={2024}
}
```



## Acknowledgements

We thank Hao-Wen Dong for his efforts in scraping [MuseScore](https://musescore.com/) and compiling the data used to create PDMX.
