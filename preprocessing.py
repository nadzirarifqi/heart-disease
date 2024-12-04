import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the Heart Disease dataset
dataset = 'https://storage.googleapis.com/dqlab-dataset/heart_disease.csv'
heart_data = pd.read_csv(dataset)

# Preprocessing and cleaning
# Handle missing values
heart_data = heart_data.dropna()

# Standardize the features
scaler = StandardScaler()

# Fit and transform using scaler
heart_data_scaled = scaler.fit_transform(heart_data.drop('target', axis=1))

# Exploratory Data Analysis
# Calculate descriptive statistics
descriptive_stats = heart_data.describe()

# Calculate correlation matrix
correlation_matrix = heart_data.corr()

# Perform PCA for dimensionality reduction
feature_number = len(heart_data_scaled[0])
pca = PCA(n_components=feature_number)

# Fit PCA with dataset
pca.fit(heart_data_scaled)

# Get variance information
variance_ratio = pca.explained_variance_ratio_

# Calculate cummulative
cumulative_variance = np.cumsum(variance_ratio)

# Create Scree Plot
plt.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker='o')
plt.xlabel('Komponen Utama ke-')
plt.ylabel('Varians (Nilai Eigen)')
plt.title('Scree Plot')
plt.show()

# Determine the number of features that can represent the entire dataset based on the image that has been created (n_components = 9)
pca = PCA(n_components=9)
# Fit and transform PCA with dataset
heart_data_reduced = pca.fit_transform(heart_data_scaled)

feature_names = heart_data.drop('target', axis=1).columns.to_list()
component_names = [f"PC{i+1}" for i in range(pca.n_components_)]

for component, component_name in zip(pca.components_, component_names):
    feature_indices = component.argsort()[::-1]
    retained_features = [feature_names[idx] for idx in feature_indices[:pca.n_components_]]
    print(f"{component_name}: {retained_features}")