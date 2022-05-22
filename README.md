# Music Semantic Dataset Preprocessor

Dataset preprocessor for current music semantic dataset

<p align = "center">
<img src = "https://i.imgur.com/LZNLhV0.png">
</p>

### Why we made this repo

There are now too many datasets and too many data splits. Because of this, if you are using multi-datasets, creating a loader will cost you a lot of time. To solve this, we propose a preprocessor for making `KV Style (key-values) annotation` file, `track split` file, and `resampler`. This will help the re-implementation of the research.


### Dataset

|   Datasets   |   Size  | hour | # of Tag | Avg.Tag | Genre | Style | Inst. | Vocal | Mood | Theme | Culture |
|:------------:|:-------:|:----:|:--------:|:-------:|:-----:|:-----:|:-----:|:-----:|:----:|:-----:|:-------:|
|  MSD-Lastfm  | 241,889 |  |    50    |     |   ✅   |       |   ✅   |   ✅   |   ✅  |       |         |
| MSD-AllMusic | 503,806 |      |   1402   |         |   ✅   |   ✅   |       |       |   ✅  |   ✅   |         |
|    MSD-500   |         |      |    500   |         |   ✅   |   ✅   |   ✅   |   ✅   |   ✅  |   ✅   |    ✅    |
|      FMA     | 104,186 |  868 |    161   |   3.34  |   ✅   |       |       |       |      |       |    ✅    |
|      MTG     |  55,525 |  463 |    183   |   4.15  |   ✅   |       |   ✅   |       |   ✅  |   ✅   |         |
|   Deezer-MT  |  48,120 |  401 |    15    |   7.32  |       |       |       |       |   ✅  |   ✅   |         |
|     MTAT     |  21,108 |  176 |    188   |   4.20  |   ✅   |   ✅   |   ✅   |   ✅   |   ✅  |       |    ✅    |
|    OpenMIC   |  20,000 |  56  |    20    |   2.08  |       |       |   ✅   |       |      |       |         |
|      KVT     |  6,787  |  19  |    42    |  22.78  |       |   ✅   |       |   ✅   |      |       |    ✅    |


### Reference
will be updated

- MSD: 
    - standard split: https://github.com/jongpillee/music_dataset_split
    - allmusic split: https://github.com/tuwien-musicir/msd
    - msd500 split: https://drive.google.com/drive/folders/1Y_XY4vdiVvqvEoe3HQcDx3wawBEgsl0o
- FMA: https://github.com/mdeff/fma
- OPENMIC: https://zenodo.org/record/1432913/files/openmic-2018-v1.0.0.tgz
- JAMENDO: https://github.com/MTG/mtg-jamendo-dataset
- MTAT: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
- DEEZER: https://github.com/SeungHeonDoh/deezer_contextual_tag
- KVT: https://khlukekim.github.io/kvtdataset/
- GIANTSTEP:
    - https://repositori.upf.edu/bitstream/handle/10230/45236/knees_ismir2015_two.pdf?sequence=1&isAllowed=y
    - https://arxiv.org/pdf/2107.05677.pdf
- GTZAN:
    - Genre: http://opihi.cs.uvic.ca/sound/genres.tar.gz
    - BenchMark: https://arxiv.org/pdf/1903.10839.pdf
    - Key: https://github.com/alexanderlerch/gtzan_key
    - tempo: https://ieeexplore.ieee.org/abstract/document/6879451 
    - tempo: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat/
    - tempo : https://arxiv.org/pdf/2109.01607.pdf
