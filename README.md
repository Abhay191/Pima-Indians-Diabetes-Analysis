# Pima Indians Diabetes Analysis and PCA Project

## Overview

This project involves the analysis of the Pima Indians Diabetes Database, focusing on data preprocessing, normalization, standardization, and Principal Component Analysis (PCA). The objective is to predict whether a patient has diabetes based on diagnostic measurements. The project also includes generating synthetic data, performing PCA, and reconstructing data samples using eigenvectors.

## Data Description

The dataset contains 768 samples with 9 attributes:
1. `pregs`: Number of times pregnant
2. `plas`: Plasma glucose concentration 2 hours in an oral glucose tolerance test
3. `pres`: Diastolic blood pressure (mm Hg)
4. `skin`: Triceps skin fold thickness (mm)
5. `test`: 2-Hour serum insulin (mu U/mL)
6. `BMI`: Body mass index (weight in kg/(height in m)^2)
7. `pedi`: Diabetes pedigree function
8. `Age`: Age (years)
9. `class`: Class variable (0 or 1, indicating diabetes status)

## Project Steps

### 1. Data Preprocessing

- **Outlier Detection and Correction**: Outliers in each attribute were replaced with the median of the respective attribute.
- **Normalization**: Min-Max normalization was applied to scale the attribute values in the range 5 to 12.
- **Standardization**: Each attribute was standardized using the formula \( \hat{x}_n = \frac{x_n - \mu}{\sigma} \), where \( \mu \) is the mean and \( \sigma \) is the standard deviation of the attribute.

### 2. Synthetic Data Generation

- **Data Generation**: Generated 1000 samples of 2-dimensional synthetic data with a bi-variate Gaussian distribution using specified mean and covariance matrix.
- **Scatter Plot**: Created a scatter plot of the synthetic data samples.
- **Eigenvalue and Eigenvector Computation**: Computed the eigenvalues and eigenvectors of the covariance matrix and plotted the eigen directions on the scatter plot.
- **Data Projection and Reconstruction**: Projected the data onto the first and second eigen directions and reconstructed the data using both eigenvectors. Estimated reconstruction error using Euclidean distance.

### 3. Principal Component Analysis (PCA)

- **Dimensionality Reduction**: Performed PCA on the outlier-corrected standardized data to reduce the data from 8 dimensions to 2.
- **Variance and Eigenvalues**: Compared the variance of the projected data along the two principal components with the corresponding eigenvalues.
- **Scatter Plot**: Created a scatter plot of the reduced dimensional data.
- **Eigenvalue Plot**: Plotted all the eigenvalues in descending order.
- **Reconstruction Errors**: Plotted reconstruction errors for different values of principal components (l = 1, 2, ..., 8). Printed covariance matrices for each dimensional representation and compared them with the original 8-dimensional data.

## Results and Observations

- **Normalization and Standardization**: Normalization scales the data to a specific range, making it suitable for algorithms that require bounded input. Standardization centers the data around the mean with unit variance, making it suitable for algorithms that assume normally distributed data.
- **Dimension Reduction**: PCA reduces data dimensionality while retaining most of the variance, simplifying the dataset and improving computational efficiency.
- **Principal Components**: The direction of principal components indicates the directions of maximum variance in the data.
- **Reconstructed Data**: Observations from the reconstructed synthetic data showed how well the original data could be approximated using the principal components.
- **Variance and Reconstruction Errors**: Analysis of variance and reconstruction errors provided insights into the trade-offs between dimensionality reduction and data fidelity.

## Usage

To run the analysis, ensure you have the following dependencies installed:
- pandas
- numpy
- matplotlib
- scikit-learn

Clone the repository and execute the main script to perform the analysis and generate the plots.

```bash
git clone https://github.com/yourusername/pima-indians-diabetes-analysis.git
cd pima-indians-diabetes-analysis
python main.py
```

## Conclusion

This project demonstrates the importance of data preprocessing, normalization, standardization, and dimensionality reduction techniques in predictive modeling. The PCA results provide valuable insights into the data structure and the effectiveness of dimensionality reduction in preserving data variance.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or suggestions, please feel free to contact me at abhay.gupta@example.com.

---
