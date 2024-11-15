# Gold Price Prediction Using LSTM

This project implements a Long Short-Term Memory (LSTM) model to predict gold prices based on historical data. It builds upon and enhances an existing Kaggle notebook by introducing additional features, hyperparameter tuning, and improving the overall performance of the prediction model.

## Introduction

This repository provides an advanced approach to predicting gold prices using an LSTM model. The steps include:

1. **Data Preprocessing:**
   - The dataset is preprocessed to clean and normalize the input data.
   - Additional features, such as the 'Vol.' column, are included to enrich the model's input.

2. **Feature Engineering:**
   - The 'Vol.' column, representing the volume of trading activity, is integrated to provide additional context for price fluctuations.

3. **Model Architecture:**
   - The model uses LSTM layers to capture sequential dependencies in the data.
   - Dense layers are added for fine-tuning predictions.

4. **Hyperparameter Tuning:**
   - The model incorporates `keras_tuner` to optimize hyperparameters like the number of LSTM units, dropout rates, and learning rates.

5. **Model Training and Evaluation:**
   - The model is trained using the enriched dataset, and its performance is evaluated using metrics like Test Loss, MAPE (Mean Absolute Percentage Error), and Accuracy.

## Dataset

The dataset used for this project is a historical gold price dataset from [Kaggle](https://www.kaggle.com/datasets/farzadnekouei/gold-price-10-years-20132023). Ensure you have the dataset saved in the working directory as `Gold Price (2013-2023).csv`.

## Improvements Over Original Code

The original notebook, available [here](https://www.kaggle.com/code/farzadnekouei/gold-price-prediction-lstm-96-accuracy/notebook), did not include the 'Vol.' column in the model. Additionally, it lacked hyperparameter tuning. The key improvements in this repository include:

- **Feature Addition:** Inclusion of the 'Vol.' column to enhance the predictive capability of the model.
- **Hyperparameter Tuning:** Used `keras_tuner` to optimize the model configuration for better performance.
- **Improved Model Accuracy:** Achieved better predictive accuracy by refining the model architecture.

### Performance Comparison

| Metric           | Original Code              | This Code                  | Improvement         |
|-------------------|----------------------------|----------------------------|---------------------|
| Test Loss        | 0.0009619238553568721      | 0.0004864981456194073      | Reduced by ~49.4%   |
| Test MAPE        | 0.032585574742406954       | 0.022679463173407313       | Reduced by ~30.4%   |
| Test Accuracy    | 96.74%                     | 97.73%                     | Increased by ~1%    |

## Results

The enhanced model successfully predicts gold prices with high accuracy, demonstrating the importance of feature engineering and hyperparameter optimization in time-series forecasting tasks.

## Acknowledgments

This project is based on the original work by [Farzad Nekouei](https://www.kaggle.com/code/farzadnekouei/gold-price-prediction-lstm-96-accuracy/notebook). Special thanks to the Kaggle community for providing the dataset and initial implementation.
