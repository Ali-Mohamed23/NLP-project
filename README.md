# mini-project-V

The repo with instructions for Mini Project V

Welcome to your final mini-project of this bootcamp. We hope you will enjoy it. 

### Description

We will combine the skills we developed in the previous modules to identify duplicate questions in a dataset provided by Quora. This dataset  was labeled by human experts which is an expensive process. The model you will build will need to automatically identify and label duplicate questions.

We are going to need to build a classifier model to achieve this result..


### Data

The labeled dataset can be downloaded from [here](https://drive.google.com/file/d/19iWVGLBi7edqybybam56bt2Zy7vpf1Xc/view?usp=sharing).

### In order to use this notebook

1) Download the dataset from the link above & keep in same directory as the notebook.

### Overview of project steps:

Download Data: The project starts by downloading the dataset containing question pairs from Quora. The dataset is stored in the train.csv file.

Exploration: Explore the dataset to gain insights into its structure and distribution. Analyze the distribution of duplicate and non-duplicate questions to understand the class balance.

Cleaning: Perform data cleaning tasks to prepare the dataset for modeling. Handle missing values and remove any unnecessary columns.

Feature Engineering: Create relevant features from the existing data that can be used to train the classifier model. This step involves tokenization, stopword removal, punctuation removal, normalization, and stemming of the questions.

Modeling: Build a classifier model using the cleaned and engineered features. Train the model on a portion of the dataset and evaluate its performance using the remaining portion as a validation set. Use appropriate evaluation metrics to assess the model's accuracy and effectiveness in identifying duplicate questions.

Model Evaluation: Assess the performance of the trained model by analyzing evaluation metrics and comparing them to the project's objectives. Fine-tune the model if necessary to achieve better results.

Final Testing: Set aside a separate portion of the dataset as the final testing data to assess the model's performance on unseen data. Evaluate the model on this test set and report its performance.

Presentation: Prepare a presentation that summarizes the entire project, including the data exploration, cleaning, feature engineering, model development, evaluation, and final testing results. Clearly explain the steps taken, the choices made, and the insights gained throughout the project. Present the model's performance and discuss its potential impact on Quora's user experience.


### Each step taken:

1.	Loaded the data file containing the question pairs.

2.	Explored the dataset to understand its structure and contents.


3.	Preprocessed the text data by removing punctuation, converting to lowercase, and removing stopwords.

4.	Tokenized the text data to obtain individual words or tokens.


5.	Performed stemming on the tokens to reduce them to their base or root form.

6.	Calculated various features to capture the characteristics of the question pairs, including:
•	Number of words in each question (q1_words, q2_words).
•	Number of similar words between q1 and q2 (q_1_2_similar).
•	Whether the last word of q1 and q2 is the same (last_word_same).
•	Word overlap percentage between q1 and q2 (word_overlap_percentage).

7.	Handled missing values by either dropping the corresponding rows or filling them with the mode value.

8.	Encoded the target variable 'is_duplicate' to binary labels (0 for not duplicate, 1 for duplicate).


9.	Utilized TF-IDF vectorization to convert the preprocessed text data into numerical feature vectors (tfidf_q1, tfidf_q2).

10.	Employed word2vec to obtain word embeddings for each word in the questions (q1_word2vec, q2_word2vec).


11.	Combined all the features (tfidf_q1, tfidf_q2, q1_words, q2_words, q_1_2_similar, last_word_same, word_overlap_percentage) to create the feature matrix X.

12.	Split the dataset into training and testing sets using train_test_split.

13.	Created a logisitic regression model. Results:

Accuracy: 0.7586931876739152
Precision: 0.6966242875931609
Recall: 0.6184254420104526
F1-score: 0.6551998350661208



14.	Created an XGBoost model using the XGBoost library and trained it on the training set.

Accuracy: 0.7666907142120993
Precision: 0.7056319782863488
Recall: 0.6359946625152897
F1-score: 0.669006053162558

15.	Made predictions on the testing set and evaluated the model's performance using various metrics such as accuracy, precision, recall, and F1-score.

16.	Explored the relationship between the feature 'q_1_2_similar' and 'is_duplicate' by creating a visual representation (pie chart) to visualize the distribution.


17.	Created a new feature 'last_word_same' that indicates whether the last word of q1 and q2 is the same.

18.	Visualized the relationship between 'last_word_same' and 'is_duplicate' using a bar chart.


19.	Performed grid search to find the best combination of hyperparameters for the XGBoost model.

20.	Created a new XGBoost model with the best parameters obtained from grid search and evaluated its performance on the testing set.

