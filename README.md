# Fraud_Detection
Detecting fraudulent transactions with machine learning techniques

# Abstract
Supervised learning and unsupervised learning models were tasked to detect fraud observations.  

For supervised learning, CatBoost performs the best:  
Evaluating CatBoost at threshold = 0.34  
Recall: 0.855  
Type II Error (missed fraud): 0.145  
Type I Error (false alarms): 0.148  
ROC AUC: 0.932  

For unsupervised learning, kmeans performs the best:  
KMeans ROC-AUC: 0.663


# Background
This project simulates a Work Pass Fraud Detection system inspired by regulatory operations.
In many employment-permit workflows, agencies screen large volumes of applications submitted by employers. Fraudulent or suspicious cases often involve under-reported salaries, quota abuse, or repeated re-applications by the same worker within a short period.

Because real work-pass data is confidential, we use synthetic data that captures the same behavioral patterns and distributions found in real operations. This allows model experimentation without revealing sensitive information.

# Synthetic Data Generation
The dataset is randomly generated with realistic field relationships:

Employers and workers are assigned IDs.  
Each record represents a work-pass application with attributes such as:  
* declared_salary (monthly wage)  
* skill_tier (1–3, roughly reflecting worker type)  
* sector (e.g., Manufacturing, F&B, Marine, Logistics)  
* nationality, timestamp, and an estimated monthly workload (est_monthly_load)  

Fraud-like patterns are injected based on domain rules that mimic how regulatory teams flag anomalies:  
1. Underpayment pattern — applications where the declared salary is far below expected for the skill tier.  
2. Quota abuse pattern — employers with a sudden spike in the number of applications (est_monthly_load) far above their historical norm.  
3. Worker hopping pattern — the same worker applying to different employers within only a few days, suggesting fake transfers or pass recycling.  

Rows that match these behavioral patterns are labeled as suspicious_flag = 1, forming the positive (fraud) class.  
The remaining rows are labeled 0 (normal applications).

Finally, the dataset is downsampled to achieve a realistic fraud rate of about 3 %, similar to the rarity of actual enforcement cases.

# Data Engineering New Columns
Several engineered features were added to make the synthetic dataset more realistic and useful for machine-learning experiments:

1️. salary_deviation  
Measures how far an applicant’s declared salary deviates from the typical pay within the same skill tier and sector.  
salary_deviation = declared_salary - mean(salary for that tier & sector)  
A large negative deviation suggests possible underpayment or salary manipulation — a common red flag in work-pass fraud.

2️. days_since_last_application  
For each worker, this feature measures the number of days between their current and previous application.  
Short intervals (e.g., < 4 days) may indicate worker hopping, where the same worker reapplies quickly under different employers — typical of pass recycling or shell-company schemes.

3️. load_zscore  
Employers’ application activity often follows predictable volumes.  
To detect unusual spikes, the z-score of each employer’s current monthly load is computed relative to their historical mean:  
load_zscore = (current_load - mean(load)) / std(load)  
High values flag potential quota abuse, where an employer suddenly submits far more applications than normal.

# Models
## Supervised Learning
Every supervised learning models were ran with stratified k-fold to ensure similar class ratio in each fold. All supervised learning models have in-built class balance arguments that do not necessitate resampling to counter the imbalance class ratio. Instead, higher weights are assigned to the much smaller suspicious_flag==1 class.
### Logistic Regression
A general logistic regression was ran to determine significant features.  
The following features were found to be significant:  
"declared_salary","est_monthly_load","skill_tier","salary_deviation","tier_salary_ratio","days_since_last_application"

With these features, another logistic regression model is fitted with stratified k-fold tested at different threshold.  
Logistic regression appears to be the poorest classifier with AUC: 0.657.

Evaluating at threshold = 0.5  
Recall: 0.837  
Type II Error (missed fraud): 0.163  
Type I Error (false alarms): 0.634  
ROC AUC: 0.657  

### Decision Tree
We see a significant improvement in decision tree with relatively acceptable type 1 and type 2 error.

Evaluating Decision Tree at threshold = 0.26  
Recall: 0.898  
Type II Error (missed fraud): 0.102  
Type I Error (false alarms): 0.285  
ROC AUC: 0.91  

### Random Forest
Random forest improves upon decision tree.

Evaluating Random Forest at threshold = 0.32  
Recall: 0.858  
Type II Error (missed fraud): 0.142  
Type I Error (false alarms): 0.182  
ROC AUC: 0.931  

### CatBoost
CatBoost boasts the highest AUC value with relatively low type 1 and type 2 error.

Evaluating CatBoost at threshold = 0.34  
Recall: 0.855  
Type II Error (missed fraud): 0.145  
Type I Error (false alarms): 0.148  
ROC AUC: 0.932  

## Unsupervised Learning
Unsupervised learning requires resampling method to address class imbalance, SMOTEENN was utilised.  
IsolationForest, Kmeans, OneClassSVM, AutoEncoder and other unsupervised were tested simply.  
Kmeans performed the best.  
KMeans ROC-AUC: 0.663
