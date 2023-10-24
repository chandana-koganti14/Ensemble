# Ensemble

This repository contains code and instructions for executing the Ensemble project.

## Project Structure

- `Ensemble/`
  - `training/`
    - `train/` - Training images.
    - `train_a/` - Training masks.
  - `output/` - Predicted images.
  - `testing/`
    - `test/` - Test images.
    - `test_a/` - Test masks.

## Files

- `ensem.py` - Code for ensemble inference.
- `trainn.py` - Code for training the model.
- `testing.py` - Code for making predictions.
- `Unet.py` - Code for the UNet model.
- `best_resunet_weights.pth` - Pre-trained weights for the ResUNet model.
- `best_attentionunet_weights.pth` - Pre-trained weights for the Attention UNet model.
- `best_vanillaunet_weights.pth` - Pre-trained weights for the Vanilla UNet model.
- `ensemble_model_weights.pth` - Pre-trained ensemble model.

## Execution

To execute the code, follow these steps:
## Command Prompt Execution

To run the code in the JoinUNet project using a command prompt, follow these steps:

### 1. Navigate to the Project Folder

Open your command prompt and navigate to the JoinUNet project folder using the `cd` (change directory) command:
```bash
cd path/to/Ensemble
```

### 2. Model Training

To train both the ResUNet and Attention UNet models, you can use the following command:
```bash
python trainn.py --model resunet/attentionunet/vanillaunet
```

### 3. Ensemble Inference

Once the training is complete, you can perform ensemble inference and save the ensemble_model_weights.pth file:
```bash
python ensem.py
```

### 4. Generating Predictions

Now, to generate predictions on test images using the ensemble model, use the following command:
```bash
python testing.py --model path-to-the-model\ensemble_model_weights.pth --test-folder path-to-testimages-folder\images --output-folder path-to-save-predicted-masks\output
```
Make sure to replace path-to-the-model, path-to-testimages-folder, and path-to-save-predicted-masks with the actual paths on your system.

By following these steps, you can execute the Ensemble project via the command prompt, including model training, ensemble inference, and generating predictions on test images.
