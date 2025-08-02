# HiDiNet: Hierarchical Diffusion Network for Healthcare Time Series Modeling

HiDiNet is a state-of-the-art deep learning framework for modeling healthcare time series data, combining Variational Autoencoders (VAE), Stochastic Differential Equations (SDE), Transformer architectures, and Memory Networks to predict health trajectories and survival probabilities.

## Overview

HiDiNet integrates multiple advanced deep learning components:
- **Variational Autoencoders (VAE)** with Normalizing Flows for latent representation learning
- **Stochastic Differential Equations (SDE)** for modeling disease dynamics and temporal evolution
- **Transformer Architecture** for capturing long-range temporal dependencies
- **Memory Networks** for maintaining patient-specific context
- **Survival Analysis** components for mortality risk prediction

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HiDiNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

### ELSA Dataset
This project uses the English Longitudinal Study of Ageing (ELSA) dataset. To obtain the data:
1. Visit the [ELSA website](https://www.elsa-project.ac.uk/)
2. Register for data access
3. Download the required ELSA data files
4. Place the data files in the appropriate folders as described below

## Model Training

### Training the Original HiDiNet Model

1. **Create Required Folders:**
   ```bash
   mkdir Data
   mkdir Data/ELSA
   mkdir Parameters
   ```

2. **Prepare Data:**
   - Place ELSA data files in the `Data/ELSA/` folder
   - Run the data parser to generate processed files:
   ```bash
   cd Data_Parser
   ./run_parser.sh
   ```
   All generated files will be saved in the `Data/` folder.

3. **Train the Model:**
   ```bash
   python train.py --job_id <unique_id> --batch_size 800 --niters 1000
   ```

4. **Generate Predictions:**
   ```bash
   python predict.py --job_id <unique_id>
   ```

5. **Create Visualizations:**
   ```bash
   cd Plotting_code
   python <plotting_script>.py
   ```

### Training the Transformer Model

1. **Navigate to Transformer Directory:**
   ```bash
   cd transformer
   ```

2. **Create Required Folders:**
   ```bash
   mkdir Analysis_Data_elsa
   mkdir Output_elsa
   mkdir Parameters_elsa
   ```

3. **Train the Transformer Model:**
   ```bash
   python train.py --job_id <unique_id> --batch_size 800 --niters 1000
   ```

4. **Generate Predictions:**
   ```bash
   python predict.py --job_id <unique_id>
   ```

5. **Create Visualizations:**
   ```bash
   cd Plotting
   python <plotting_script>.py
   ```

## Key Features

- **Mixed Precision Training**: Optimized for faster training and reduced memory usage
- **KL Divergence Scheduling**: Gradual increase of KL terms for training stability
- **Monte Carlo Simulations**: For uncertainty quantification in predictions
- **Autoregressive Decoding**: Sequential prediction in transformer decoder
- **Memory-Efficient Processing**: Optimized for large healthcare datasets
- **Comprehensive Evaluation**: C-index, Brier scores, and longitudinal RMSE metrics

## Model Architecture

HiDiNet combines several advanced components:

1. **VAE with Normalizing Flows**: Handles missing data and learns latent representations
2. **SDE Dynamics**: Models temporal evolution of health variables
3. **Transformer Encoder**: Captures complex temporal dependencies
4. **Transformer Decoder**: Generates survival predictions autoregressively
5. **Memory Networks**: Maintains patient-specific context
6. **Survival Analysis**: Predicts mortality risk and survival probabilities

## Training Parameters

### Key Hyperparameters:
- `batch_size`: Training batch size (default: 800)
- `learning_rate`: Learning rate for optimization
- `niters`: Number of training iterations
- `corruption`: Data corruption rate for robustness
- `gamma_size`: Size of gamma parameters
- `z_size`: Latent space dimensionality
- `decoder_size`: Decoder network size
- `Nflows`: Number of normalizing flow layers

### Advanced Features:
- Gradient clipping for training stability
- Learning rate scheduling with ReduceLROnPlateau
- Weights & Biases integration for experiment tracking
- Mixed precision training with automatic mixed precision (AMP)

## Evaluation Metrics

The model is evaluated using multiple metrics:
- **C-index**: Concordance index for survival prediction accuracy
- **Brier Score**: Calibration measure for survival predictions
- **Longitudinal RMSE**: Root mean square error for trajectory predictions
- **Survival Probability**: Time-varying survival estimates

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## Contact

For questions or support, please contact [adhanani@trinity.edu]
