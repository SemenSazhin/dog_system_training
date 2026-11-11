# Stipra (TeknTrash)

Full archive is ~21.7 GB (`videos.zip`, 1273 mp4).

```bash
pip install huggingface_hub
huggingface-cli download TeknTrash/Stipra --repo-type dataset --include videos.zip --local-dir datasets/stipra
```

Then unzip and treat clips as **positive cleanup-like** examples.
