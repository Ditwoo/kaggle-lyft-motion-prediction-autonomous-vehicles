
# export DEBUG=19
export L5KIT_DATA_FOLDER='./data'
export PYTHONWARNINGS='ignore'
export NUM_WORKERS=12
export TRAIN_TRAJ_SIZES='./notebooks/train_zarr_sizes.csv'

# PYTHONPATH=. python3 src/experiments/test_create_val_subset.py
PYTHONPATH=. python3 src/experiments/resnet18_bi_cat_hist_confidence.py
# PYTHONPATH=. python3 src/experiments/resnet18_biger_images_confidence_continue4.py
