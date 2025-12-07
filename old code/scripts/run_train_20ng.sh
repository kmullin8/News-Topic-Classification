#mark as executable: chmod +x scripts/run_train_20ng.sh
#from root of the project, run: ./scripts/run_train_20ng.sh

#!/usr/bin/env bash
set -e  # Exit on error

# Ensure the project root is in PYTHONPATH
export PYTHONPATH=.

# Run the training module
python -m src.training.train_20ng