
# export DEBUG=19
export L5KIT_DATA_FOLDER='./data'
export PYTHONWARNINGS="ignore"
export NUM_WORKERS=20

# PYTHONPATH=. python3 src/experiments/test_create_val_subset.py
PYTHONPATH=. python3 src/experiments/test_resnet18_longer_history_confidence.py
