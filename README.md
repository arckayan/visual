![banner](/docs/banner.jpg)

This repository is our implementation of various "Visual" tasks in computer vision and pattern recognition. 

|                Paper          |                    PDF                        |
|-------------------------------|-----------------------------------------------|
| Show, Ask, Attend, and Answer | [arxiv](https://arxiv.org/pdf/1704.03162.pdf) |

## Datasets `--download`

The repository comes with inbuilt utility for downloading and processing datasets, and the commands are following:

To download the dataset use `--download` option and `--split` to sepecify the split to be downloaded. This will download the dataset in the default path (`./data/vqa`)
```bash
python . --split [valid | train | test] --download 
```

The dataset can also be downloaded in the custom specified path with  `--path` option.
```bash
python . --split [valid | train | test] --path <path> --download 
```

## Extracting Visual Features `--process-vf`

The visual features are extracted and stored in a `h5` file (specified by `--vf-file`, that defaults to `data/vqa/visual_features.h5`), so that we don't recompute these every time

```bash
python . --vf-file [data/vqa/visual_features.h5 | custom part] --process-vf 
```

There may be case when the processing of an image might have failed, therfore after computing the features run the following to verify the features:

```bash
python . --vf-file [data/vqa/visual_features.h5 | custom part] --verify-vf
```

