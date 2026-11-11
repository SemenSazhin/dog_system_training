# Datasets for human cleanup-after-dog detection

Exact public datasets of people picking up dog feces are essentially nonexistent.
This folder stores the closest open proxies and manifests.

## Downloaded / prepared here
- `charades/` — Charades annotations + cleanup-like video id list
- `shitspotter/sample_images/` — small image sample (before/after dog feces)
- `stipra/` — metadata + instructions for full 21.7GB trash-disposal videos
- `cleanup_proxy/` — synthetic pose sequences for local pipeline tests
- `manifests/` — catalogs and filtered lists

## Download
```bash
py -3 datasets/download_datasets.py --all
py -3 datasets/download_datasets.py --stipra-full   # optional, ~22GB
```
