# Models Directory

This directory stores trained machine learning models.

## Structure:
- `individual/` - Single algorithm models 
- `ensemble/` - Combined models

## Files:
- `.pkl` files - Saved scikit-learn models
- `.json` files - Model configurations and performance metrics

## Usage:
Models are automatically saved when you run the training scripts.

```bash
python src/ensemble_model.py
```

The trained models can then be used for predictions in the dashboard or command-line interface.
