# Point Cloud Generation - README

## Installation

First, install the required dependencies:

```bash
pip install trimesh open3d numpy scipy
```

## Generation

### Option 1: Generate Everything at Once (Recommended)

Generate triplanes, meshes, AND point clouds in one command:

```bash
python gen_samples.py \
    --ddpm_ckpt cars/ddpm_cars_ckpts/model405000.pt \
    --decoder_ckpt cars/car_decoder.pt \
    --stats_dir cars/statistics/cars_triplanes_stats \
    --num_samples 5 \
    --generate_pointclouds
```

### Option 2: Generate Only Meshes First

If you want to generate meshes without point clouds:

```bash
python gen_samples.py \
    --ddpm_ckpt cars/ddpm_cars_ckpts/model405000.pt \
    --decoder_ckpt cars/car_decoder.pt \
    --stats_dir cars/statistics/cars_triplanes_stats \
    --num_samples 5
```

Then convert meshes to point clouds later:

```bash
python mesh_to_pointcloud.py \
    --input samples/objects \
    --output samples/pointclouds \
    --num_points 2048 \
    --num_variants 5
```

### Option 3: Custom Point Cloud Settings

Generate high-resolution point clouds with more variants:

```bash
python gen_samples.py \
    --ddpm_ckpt cars/ddpm_cars_ckpts/model405000.pt \
    --decoder_ckpt cars/car_decoder.pt \
    --stats_dir cars/statistics/cars_triplanes_stats \
    --num_samples 5 \
    --generate_pointclouds \
    --pc_num_points 4096 \
    --pc_variants 10
```

## Command-Line Arguments

### gen_samples.py (New Arguments)

- `--generate_pointclouds`: Enable point cloud generation (flag, default: disabled)
- `--pc_num_points`: Number of points per cloud (default: 2048)
- `--pc_variants`: Number of variants per mesh (default: 5)

### mesh_to_pointcloud.py (Standalone)

- `--input`: Input mesh file or directory (required)
- `--output`: Output directory (required)
- `--num_points`: Points to sample (default: 2048)
- `--num_variants`: Variants to generate (default: 5)
- `--no_noise`: Disable noise addition (flag)
- `--no_partial`: Disable partial view generation (flag)
