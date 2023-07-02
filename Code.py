#Descriptive Statistics:

import pandas as pd
# Read data from CSV
data = pd.read_csv('descriptive_data.csv')
# Calculate descriptive statistics
mean = data['Value'].mean()
median = data['Value'].median()
variance = data['Value'].var()
standard_deviation = data['Value'].std()
range_value = data['Value'].max() - data['Value'].min()
print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", standard_deviation)
print("Range:", range_value)

#Hypothesis Testing:

import pandas as pd
from scipy.stats import ttest_ind
# Read data from CSV
data = pd.read_csv('hypothesis_data.csv')
# Perform t-test
t_statistic, p_value = ttest_ind(data['Group1'], data['Group2'])
print("T-statistic:", t_statistic)
print("P-value:", p_value)

#Correlation Analysis:

import pandas as pd
# Read data from CSV
data = pd.read_csv('correlation_data.csv')
# Calculate correlation coefficient
correlation = data['X'].corr(data['Y'])
print("Correlation coefficient:", correlation)

#Regression Analysis:

import pandas as pd
import statsmodels.api as sm

# Read data from CSV
data = pd.read_csv('regression_data.csv')

# Perform linear regression
X = sm.add_constant(data['X'])
model = sm.OLS(data['Y'], X)
results = model.fit()

print(results.summary())


#ANOVA (Analysis of Variance):

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
# Read data from CSV
data = pd.read_csv('anova_data.csv')
# Perform ANOVA for Group1
model_formula1 = 'Group1 ~ 1'
model1 = ols(model_formula1, data=data).fit()
anova_table1 = sm.stats.anova_lm(model1, typ=2)
print("ANOVA for Group1")
print(anova_table1)
# Perform ANOVA for Group2
model_formula2 = 'Group2 ~ 1'
model2 = ols(model_formula2, data=data).fit()
anova_table2 = sm.stats.anova_lm(model2, typ=2)
print("ANOVA for Group2")
print(anova_table2)
# Perform ANOVA for Group3
model_formula3 = 'Group3 ~ 1'
model3 = ols(model_formula3, data=data).fit()
anova_table3 = sm.stats.anova_lm(model3, typ=2)
print("ANOVA for Group3")
print(anova_table3)

#Principal Component Analysis (PCA):

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# Read data from CSV
data = pd.read_csv('pca_data.csv')
# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
print("Explained variance ratio:", pca.explained_variance_ratio_)

#Cluster Analysis:

import pandas as pd
from sklearn.cluster import KMeans
# Read data from CSV
data = pd.read_csv('cluster_data.csv')
# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(data)
print("Cluster labels:", clusters)







#Principal Component Analysis (PCA):
import pandas as pd
from sklearn.decomposition import PCA
# Read data from CSV
data = pd.read_csv('pca_data.csv')
# Perform PCA
pca = PCA()
pca.fit(data)
# Access the principal components and explained variance ratio
components = pca.components_
explained_variance_ratio = pca.explained_variance_ratio_
print("Principal Components:")
print(components)
print("Explained Variance Ratio:")
print(explained_variance_ratio)

#Hierarchical Clustering:

import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
# Read data from CSV
data = pd.read_csv('hierarchical_clustering_data.csv')
# Perform hierarchical clustering
linked = linkage(data, 'ward')
# Plot dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked)
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

#Survival Analysis:

import pandas as pd
import lifelines

# Read data from CSV
data = pd.read_csv('survival_analysis_data.csv')

# Perform survival analysis
kmf = lifelines.KaplanMeierFitter()
kmf.fit(data['Time'], event_observed=data['Event'])

# Plot survival curve
kmf.plot()
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.title('Survival Analysis')
plt.show()

#Structural Equation Modeling (SEM):

import semopy
# Read data from CSV
data = pd.read_csv('sem_data.csv')
# Define the SEM model
model = """
    X1 ~ X2 + X3
    X2 ~ X3
"""
# Fit the SEM model
fit = semopy.Model(model)
fit.fit(data)
# Access the estimated parameters
params = fit.parameters
print("Estimated Parameters:")
print(params)

#Multilevel Modeling:

import pandas as pd
import statsmodels.api as sm
# Read data from CSV
data = pd.read_csv('multilevel_data.csv')
# Perform multilevel modeling
model_formula = 'Score ~ 1 + Group'
model = sm.MixedLM.from_formula(model_formula, groups=data['Individual'], data=data)
results = model.fit()

print(results.summary())


