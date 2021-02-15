![banner](/docs/banner.jpg)

This repository is the collection multiple computer vision database, models and utilites.

<!-- ### Introduction -->

<!-- ### Installation 

you will need wget install in your system to download the datasets, run:
- Mac - `brew install wget`
- Linux - `apt-get install wget`
- Windows - `winget install -e --id GnuWin32.Wget` -->

<!-- ### Usage -->

## Download Dataset

Download the dataset with inbuilt utility, run the following command and specify the split that you wish to download.

```bash
python . --split [valid|train|test] --download 
```

The dataset can also be downloaded in the custom specified path with  `--path` option
```bash
python . --split [valid|train|test] --path <path> --download 
```
