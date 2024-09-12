Resume Classification: Approach, Findings, and Challenges
Introduction

This report details the process and results of developing a machine learning model to classify resumes into various job categories. The objective was to build a model capable of automatically categorizing resumes based on their content, which could significantly streamline recruitment processes.

Methodology

Data Preprocessing
1. Text Cleaning: Removed special characters and converted text to lowercase.
2. Tokenization: Split the text into individual words.
3. Lemmatization: Reduced words to their base form to standardize the text.

Feature Extraction
Utilized Term Frequency-Inverse Document Frequency (TF-IDF) vectorization to convert the text data into numerical features. This method captures the importance of words in each document relative to the entire corpus.

Model Selection and Training
Chose Logistic Regression as the classification algorithm due to its effectiveness in text classification tasks and interpretability. The data was split into training (80%) and testing (20%) sets.

Results

MODEL PERFORMANCE
The model achieved an impressive accuracy of 99.48% on the test set. 

Classification Report Highlights:
- Most categories achieved perfect precision, recall, and F1-scores (1.00).
- Slight imperfections were observed in:
  - DevOps Engineer: Recall of 0.93
  - PMO: Precision of 0.88

 

 Feature Importance
 
The most important features (words) for classification closely align with specific job roles and skills, demonstrating the model's ability to identify key terms associated with different resume categories.
 
Challenges and Limitations

1. Potential Overfitting: The extremely high accuracy (99.48%) raises concerns about overfitting. While it suggests that the resumes in different categories are highly distinct, it may limit the model's ability to generalize to new, unseen resumes.

2. Limited Error Analysis: With very few misclassifications, it was challenging to perform comprehensive error analysis to improve the model further.

3. Class Imbalance: Some categories had significantly fewer samples, which could lead to biased predictions in a real-world scenario with a more diverse set of resumes.

Conclusion

The developed model demonstrates exceptional performance in classifying resumes into various job categories. Its high accuracy suggests strong potential for automating initial resume screening processes in recruitment. However, the near-perfect performance also warrants caution and further testing on diverse, real-world data to ensure generalizability.

Future Work

1. Cross-validation: Implement k-fold cross-validation to ensure consistent performance across different data splits.
2. Alternative Models: Experiment with other algorithms like Random Forests or Support Vector Machines for comparison.
3. Feature Engineering: Explore domain-specific feature engineering to potentially improve model interpretability and performance.
4. Expanded Dataset: Test the model on a larger, more diverse set of resumes to validate its real-world applicability.

---

This report provides a concise overview of your approach, findings, and challenges. To further enhance .

BY…
Vaibhav Ayush Raj
