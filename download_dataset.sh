# !/usr/bin/bash
# This script downloads the Flick8k dataset, which contains images and their correspond captions.

# images
echo "Downloading Flick8k images ... "
wget -O ./images/Flickr8k_Dataset.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip

echo "Unzipping the images ... "
unzip ./images/Flickr8k_Dataset.zip -d ./images

rm -f ./images/Flickr8k_Dataset.zip

# captions
echo "Downloading Flick8k captions ... "
wget -O ./data/Flickr8k_text.zip https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

echo "Unzipping the captions ... "
unzip ./data/Flickr8k_text.zip -d ./data

rm -f ./data/Flickr8k_text.zip
