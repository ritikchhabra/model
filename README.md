# YOLOP

This is the workspace for the YOLOP (You Only Look Once for Panoptic driving Perception) project.

## Directory Structure
- `train.py`: Script used to train the model.
- `validate.py`: Script used to validate the model performance.
- `inference.py`: Script for running model inference.
- `configs/`: Contains configuration files for training and inference.
- `datasets/`: Directory for storing training and validation datasets.
- `models/`: Directory containing model architecture definitions.
- `scripts/`: Helper and utility scripts.
- `checkpoints/`: Directory where trained model checkpoints are saved.
- `requirements.txt`: Python package dependencies for the project.

## Environment Setup

1. **Activate the virtual environment**:
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To train the model, run the training script:
```bash
python train.py
```

### Validation
To validate the model, run:
```bash
python validate.py
```
