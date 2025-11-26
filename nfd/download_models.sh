#!/bin/bash
# Download models (ddpm and decoder) for cars class of ShapeNet

cd models

# DDPM checkpoints
wget https://imt-public-datasets.s3.amazonaws.com/ddpm_ckpts/cars/ddpm_cars_405k.zip && unzip ddpm_cars_405k.zip -d cars && rm ddpm_cars_405k.zip

# Decoder checkpoints
wget https://imt-public-datasets.s3.amazonaws.com/decoder_ckpts/car_decoder.pt -P cars/

# Statistics
wget https://imt-public-datasets.s3.amazonaws.com/triplane_statistics/cars_triplanes_stats.zip && unzip cars_triplanes_stats.zip -d cars/statistics && rm cars_triplanes_stats.zip

cd ..