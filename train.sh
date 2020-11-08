
# export DEBUG=19
export L5KIT_DATA_FOLDER='./data'
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS='ignore'
export NUM_WORKERS=16
export TRAIN_TRAJ_SIZES='./notebooks/train_zarr_sizes.csv'

# PYTHONPATH=. python3 src/experiments/baseline_resnet_fast_rasterizer/1_exp_baseline.py
PYTHONPATH=. python3 src/experiments/baseline_resnet_fast_rasterizer/1_exp_val_and_submit.py

# PYTHONPATH=. python3 src/experiments/resnet18_biger_images_confidence_continue4.py
