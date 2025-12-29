# GitHub Actions Workflows Guide

SUB ai uses GitHub Actions to automatically train and test the model on GitHub's servers. No local GPU or setup required!

## Available Workflows

### 1. Train SUB ai Model
**File**: `.github/workflows/train-model.yml`

Automatically trains the number detection model on GitHub's cloud infrastructure.

#### How to Run

1. Go to your repository on GitHub
2. Click on the **Actions** tab
3. Select **"Train SUB ai Model"** from the left sidebar
4. Click **"Run workflow"** button (top right)
5. (Optional) Customize parameters:
   - **Epochs**: Number of training epochs (default: 15)
   - **Batch Size**: Batch size for training (default: 128)
6. Click **"Run workflow"**

#### What It Does

- ☑️ Sets up Python 3.10 environment
- ☑️ Installs all dependencies from `requirements.txt`
- ☑️ Downloads MNIST dataset automatically
- ☑️ Trains the CNN model
- ☑️ Evaluates model accuracy (target: 98-99%)
- ☑️ Saves trained model as `.h5` file
- ☑️ Generates training history plots
- ☑️ Uploads model and plots as artifacts
- ☑️ Commits trained model back to repository

#### Training Time

- **Duration**: ~5-8 minutes on GitHub's runners
- **Cost**: FREE (GitHub provides free minutes for public repos)

#### Artifacts

After training completes, you can download:
- **trained-model-XXX**: Contains the `.h5` model files
- **training-logs-XXX**: Contains training history plots

Artifacts are kept for 30 days.

---

### 2. Test SUB ai Model
**File**: `.github/workflows/test-model.yml`

Automatically tests the model detector functionality.

#### When It Runs

- Automatically on every **push** to main branch
- Automatically on every **pull request**
- Manually via **workflow_dispatch**

#### What It Does

- ☑️ Sets up Python environment
- ☑️ Installs dependencies
- ☑️ Runs `test_detector.py`
- ☑️ Uploads test results as artifacts

---

## Viewing Workflow Results

### Check Workflow Status

1. Go to **Actions** tab in your repository
2. Click on the workflow run you want to check
3. View the logs and results

### Download Trained Model

1. Go to completed workflow run
2. Scroll down to **Artifacts** section
3. Click to download:
   - `trained-model-XXX.zip` - Contains your trained model
   - `training-logs-XXX.zip` - Contains training plots

### View Training Results

After the workflow completes, the trained model is automatically committed to your repository:
- `models/sub_ai_model_latest.h5` - Latest trained model
- `models/sub_ai_model_YYYYMMDD_HHMMSS.h5` - Timestamped model
- `models/sub_ai_model_training_history.png` - Training visualization

---

## Workflow Configuration

### Customizing Training Parameters

Edit `.github/workflows/train-model.yml` to change defaults:

```yaml
inputs:
  epochs:
    description: 'Number of training epochs'
    default: '15'  # Change this
  batch_size:
    description: 'Batch size for training'
    default: '128'  # Change this
```

### Auto-trigger on Code Changes

Uncomment these lines in `train-model.yml` to auto-train when code changes:

```yaml
push:
  branches: [ main ]
  paths:
    - 'train.py'
    - 'number_detector.py'
```

---

## Benefits of GitHub Actions Training

✅ **No Local Setup**: Train without installing anything locally
✅ **Free Compute**: GitHub provides free CI/CD minutes
✅ **Reproducible**: Same environment every time
✅ **Automated**: Set it and forget it
✅ **Version Control**: Models are committed with full history
✅ **Artifact Storage**: Download models from any workflow run

---

## Troubleshooting

### Workflow Fails

1. Check the logs in the Actions tab
2. Common issues:
   - Dependency installation failure: Update `requirements.txt`
   - Out of memory: Reduce batch size
   - Timeout: Reduce epochs

### Model Not Committed

- Check if the workflow has write permissions
- Go to Settings → Actions → General → Workflow permissions
- Enable "Read and write permissions"

### Download Artifacts

If you can't see artifacts:
- Wait for the workflow to complete
- Artifacts are available for 30 days (configurable)
- Failed workflows may not produce artifacts

---

## Example: Training Output

```
============================================================
SUB ai - Number Detection Model Training
============================================================

Training Configuration:
  Epochs: 15
  Batch Size: 128

Loading MNIST dataset...
Training samples: 60000
Testing samples: 10000
Image shape: (28, 28, 1)
Data loaded successfully!

Building CNN model...
Model built successfully!

Starting training for 15 epochs...

Epoch 1/15
422/422 [██████████] - 45s - loss: 0.2514 - accuracy: 0.9234
Epoch 2/15
422/422 [██████████] - 42s - loss: 0.0876 - accuracy: 0.9738
...

Training completed!

Evaluating model on test data...
Test Loss: 0.0312
Test Accuracy: 98.94%

============================================================
Training completed successfully!
Final Test Accuracy: 98.94%
Model saved to: models/sub_ai_model_latest.h5
============================================================
```

---

## Next Steps

After training:

1. **Download the model** from artifacts
2. **Test locally**: `python test_detector.py`
3. **Use in your app**: Import `NumberDetector` class
4. **Deploy**: Use the trained model in production

---

**Happy Training! 🚀**
