
# Parkinson Disease Diagnosis Prediction

## Project Description
This project uses a Support Vector Machine (SVM) to predict whether a patient has Parkinson's disease based on features such as tremor amplitude, voice pitch, motor coordination, and reaction time.

## Files
- `parkinson_diagnosis_data.csv`: Synthetic dataset for training and testing.
- `parkinson_diagnosis.py`: Python script to train the SVM model and evaluate its performance.
- `parkinson_predictions.csv`: Output file with actual and predicted labels.

## Requirements
- Python 3.x
- pandas
- scikit-learn

## Usage
1. Install the required libraries:
   ```bash
   pip install pandas scikit-learn
   ```
2. Run the Python script:
   ```bash
   python parkinson_diagnosis.py
   ```
3. Check the `parkinson_predictions.csv` file for predictions and evaluation results.

## Future Improvements
- Include more features such as brain scan data or additional biomarkers.
- Experiment with advanced machine learning models or ensemble methods.
