---
license: osl-3.0
tags:
- barcode
- product
- recycling
- trash
- rubbish
- disposal
pretty_name: Stipra
size_categories:
- 10M<n<100M
---
# Stipra Dataset

The **Stipra Dataset** consists of 1273 videos of regular people disposing household products into the garbage. 

Stipra was a product of TeknTrash, a UK research company which provides robotic-based solutions to trash handling. Stipra was designed to provide companies with real time 
consumption data so they could understand their consumers better while rewarding consumers for proper disposal practices. Thus, it was effectively a loyalty system for disposal.

All videos are up to 60 seconds  long and were anonymized. They were obtained using an app designed for Android and IOS which would capture barcodes while making a video. 
The app would then send the video for the backend, which would confirm each one of the barcodes identified in order to prevent scanning the same product more than once. The app has also been made available at https://github.com/tekntrash/stipra

The dataset is provided in a single zip file of around 20 Gb size. It is open source, making it freely available for use in any project, for any purpose.

## Dataset Overview

- **Number of Videos**: 1273
- **Video Format**: MP4
- **Video Duration**: Up to 60 seconds per video
- **Content**: Regular people disposing of household items in the garbage
- **Anonymization**: All individuals in the videos have been anonymized to protect privacy
- **License**: Open Source (Free to use for any purpose)

## Dataset Usage

This dataset is ideal for training models on tasks such as:
- Object detection (e.g., recognizing household items)
- Action recognition (e.g., identifying the action of disposing items)
- Gesture or movement detection

Since the dataset is fully open source, you can use it for any purpose, whether personal or commercial.

## Downloading the Dataset

To access and download the dataset, use the following command:

```bash
# Install the Hugging Face datasets library
pip install --upgrade datasets

# Download the Stipra dataset
from datasets import load_dataset
dataset = load_dataset("TeknTrash/Stipra")