# %%
import pandas as pd
from warnings import filterwarnings
filterwarnings('ignore')

# %% [markdown]
# # Data Preparation

# %%
df = pd.read_csv("../data/Spam Email raw text for NLP.csv")
df

# %%
df.info()

# %%
type_mapping = {
    'MESSAGE': 'string',
    'FILE_NAME': 'string'
}
df = df.astype(type_mapping)
df.info()

# %%
df['CATEGORY'].value_counts()

# %%
import nltk

nltk.download('wordnet')
nltk.download('stopwords')

# %%
tokenizer = nltk.RegexpTokenizer(r"\w+")

sentences = "HEY <LADIES>, <drop> it down. Just want to see you touch the ground. Don't be shy girl, go <BONANZA>. Shake your body like a belly dancer"

tokenized_sentences = tokenizer.tokenize(sentences)
sentences, tokenized_sentences

# %%
sentences_lower_cased = [t.lower() for t in tokenized_sentences]
sentences_lower_cased

# %%
from nltk import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_sentences = [wordnet_lemmatizer.lemmatize(token) for token in sentences_lower_cased]
lemmatized_sentences

# %%
from nltk.corpus import stopwords
stopwords_list = stopwords.words('english')
useful_tokens = [stopword for stopword in lemmatized_sentences if stopword not in stopwords_list]
useful_tokens

# %%
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer

def message_tokenizer(message):
    reg_tokenizer = RegexpTokenizer(r'\w+')
    tokenized_message = reg_tokenizer.tokenize(message)
    lower_cased = [t.lower() for t in tokenized_message]
    stop_words = stopwords.words('english')

    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(lower_case) for lower_case in lower_cased]

    required_tokens = [token for token in lemmatized_tokens if token not in stop_words]
    return required_tokens

# %%
message_tokenizer(sentences)

# %% [markdown]
# # Train/Test split

# %%
df = df.sample(frac=1, random_state=1)
df = df.reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df, test_df = df[:split_index], df[split_index:]

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df, test_df

# %%
train_df['CATEGORY'].value_counts()

# %%
test_df['CATEGORY'].value_counts()

# %% [markdown]
# # Data preprocessing

# %%
token_count = {}

for message in train_df['MESSAGE']:
    tokenized_message = message_tokenizer(message)

    for token in tokenized_message:
        if token in token_count:
            token_count[token] += 1
        else:
            token_count[token] = 1

token_count, len(token_count)

# %%
def keep_token(processed_token, threshold=10000):
    if processed_token not in token_count:
        return False
    else:
        return token_count[processed_token] > threshold

# %%
keep_token('quick', 100)

# %%
features = set()

for token in token_count:
    if keep_token(token, 9981) == True:
        features.add(token)

features = list(features)
features

# %%
token_to_index_mapping = {t:i for t, i in zip(features, range(len(features)))}
token_to_index_mapping

# %%
message_tokenizer('3d b <br> .com bad font font com randoms')

# %% [markdown]
# ## Bag of word approach

# %% [markdown]
# **"Bag of Words" (count vector)**
# 
# **-> T_s = [http  tr  size  3d  font  br  com  td   p   b]**
# 
# **-> I_s = [0      1    2    3    4    5    6   7   8   9]**
# 
# **-> V_s = [0,   0,   0,   1,  2,   1,   2,   0,  0,  1]**
# 
# *Res*: `[0.,  0.,  0.,   1., 2.,  1., 2.,  0., 0., 1.]`

# %%
import numpy as np

def message_to_count_vector(message):
    count_vector = np.zeros(len(features))
    useful_tokens = message_tokenizer(message)
    for token in useful_tokens:
        if token not in features:
            continue
        token_index = token_to_index_mapping[token]
        count_vector[token_index] += 1
    return count_vector.astype('int64')

# %%
message_to_count_vector(train_df['MESSAGE'].iloc[3])

# %%
message_to_count_vector(train_df['MESSAGE'].iloc[10])

# %%
def df_to_X_Y(df: pd.DataFrame):
    y = df['CATEGORY'].to_numpy().astype('int64')
    X = []
    for message in df['MESSAGE']:
        count_vector = message_to_count_vector(message=message)
        X.append(count_vector)
    return np.array(X).astype(int), np.array(y).astype(int)

# %%
X_train, y_train = df_to_X_Y(df=train_df)
X_test, y_test = df_to_X_Y(df=test_df)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# %% [markdown]
# ## Scaling the `X_train` and `X_test`

# %%
from sklearn.preprocessing import StandardScaler

s_scaler = StandardScaler()
X_train_scaled, X_test_scaled = s_scaler.fit_transform(X_train), s_scaler.fit_transform(X_test)
X_train_scaled, X_test_scaled

# %% [markdown]
# # Training classification models

# %%
from sklearn.svm import SVC
from sklearn.metrics import classification_report

svc = SVC(kernel='linear', gamma='scale', class_weight='balanced')
svc.fit(X=X_train_scaled, y=y_train)
predictions_svc = svc.predict(X=X_test_scaled)
print(classification_report(y_pred=predictions_svc, y_true=y_test))

# %%
svc = SVC(kernel='rbf', gamma='scale', class_weight='balanced')
svc.fit(X=X_train_scaled, y=y_train)
predictions_svc = svc.predict(X=X_test_scaled)
print(classification_report(y_pred=predictions_svc, y_true=y_test))

# %%
from sklearn.linear_model import LogisticRegression

l_regression = LogisticRegression(penalty='l2', random_state=1, solver='saga', class_weight='balanced')
pred_logistic = l_regression.fit(X=X_train_scaled, y=y_train).predict(X_test_scaled)
print(classification_report(y_pred=pred_logistic, y_true=y_test))

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_jobs=300,
    criterion='log_loss',
    class_weight='balanced',
    random_state=1,
    max_features=len(features),
    warm_start=True,
).fit(X_train, y_train)
print(classification_report(y_true=y_test, y_pred=rf.predict(X_test)))

# %% [markdown]
# If you see it in all of those, you can see the class imbalance. Therefore, there has to be augmented data for non-spam (1) category.

# %% [markdown]
# ## Balanced data model training
#
# We'll create synthetic data by generating new samples based on the existing data points in the minority class. The **SMOTE (Synthetic Minority Over-sampling Technique)** algorithm is popular for generating synthetic data by interpolating between existing minority class samples.

# %%
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=1)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
X_test_balanced, y_test_balanced = smote.fit_resample(X_test_scaled, y_test)

l_regression = LogisticRegression(penalty='l2', random_state=1, solver='saga', class_weight='balanced')
pred_logistic_smote = l_regression.fit(X=X_train_balanced, y=y_train_balanced).predict(X_test_balanced)

print(classification_report(y_pred=pred_logistic_smote, y_true=y_test_balanced))

# %% [markdown]
# It looks like use of SMOTE to balance the classes had a noticeable impact on model's performance. Here’s what the output tells us:
# 
# ### Observations:
# - **Precision and Recall Trade-off**: Your recall for class `0` (majority class) is very high (0.99), indicating that the model is very good at identifying `0` instances. However, the precision for class `0` has decreased to 0.67, which means that when the model predicts `0`, it's correct about 67% of the time.
# - **Class `1` (minority class) Performance**: The precision for class `1` has improved to 0.99, which is excellent, but the recall is lower at 0.52. This indicates that while the model is very good at identifying when it predicts class `1`, it misses half of the actual class `1` instances.
# - **Overall Accuracy**: Your model's accuracy has dropped to 0.76, which may be a result of the imbalance correction affecting how well it generalizes to both classes.
# 
# ### Key Takeaways:
# 1. **SMOTE has helped with balancing** the dataset by creating synthetic instances, improving the model's ability to identify the minority class (`1`) when it's predicted.
# 2. **Precision and recall** are not in perfect balance; improving one reduces the other, so consider the specific use case and which metric matters more for your objectives.
# 3. **Overfitting Potential**: You may be at risk of overfitting due to the synthetic data generated by SMOTE. This is especially true if the number of synthetic samples is high.
# 
# ### Recommendations for Improvement:
# - **Try Different Sampling Techniques**: You could experiment with **SMOTE-NC** (SMOTE for mixed-type data) or **ADASYN** for more adaptive oversampling.
# - **Use Ensemble Models**: Consider using ensemble methods like **Random Forest** or **Gradient Boosting** (e.g., `XGBoost` or `LightGBM`) with **balanced class weights**.
# - **Tune Hyperparameters**: Adjust hyperparameters for `LogisticRegression` (e.g., `C` value for regularization) and try techniques like **grid search** or **random search** for optimal results.
# - **Evaluate with Different Metrics**: If you're dealing with an imbalanced problem, focus more on metrics like **precision-recall AUC**, **F1-score**, and **confusion matrix** rather than just accuracy.
# 
# ### Next Steps:
# 1. **Plot Precision-Recall Curve**: Visualize how precision and recall change with different threshold values.
# 2. **Cross-Validation**: Ensure you're validating your model using cross-validation to get a more robust estimate of performance.
# 3. **Try Class Weights**: Combining SMOTE with `class_weight='balanced'` may help optimize results without losing generality.

# %%
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_jobs=300,
    criterion='entropy',
    class_weight='balanced',
    random_state=1,
    max_features=len(features),
    warm_start=True,
)\
    .fit(X_train_balanced, y_train_balanced)
print(classification_report(y_true=y_test_balanced, y_pred=rf.predict(X_test_balanced)))

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(rf, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
print(f'Cross-Validation F1 Score: {scores.mean()}')

# %%
import xgboost as xgb

xgb_classifier = xgb.XGBClassifier(
    n_estimators=1000,
    n_jobs=200,
    grow_policy='lossguide',
    learning_rate=0.001,
    booster='gbtree',
    random_state=1,
    tree_method='exact',
    eval_metric='logloss'
).fit(X=X_train_balanced, y=y_train_balanced)

print(classification_report(y_true=y_test_balanced, y_pred=xgb_classifier.predict(X_test_balanced)))

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb_classifier, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
print(f'Cross-Validation F1 Score: {scores.mean()}')

# %% [markdown]
# # Ensembling based model training
# 
# As of now, our best models with balanced dataset of being trained and tested are that of `l_regression` and `xgb_classifier`. We'll combine them up for the best of the results

# %% [markdown]
# ## VotingClassifier
# 
# A simple way to combine models by averaging their predictions. This method combines the predictions from multiple models and selects the most frequent class (hard voting) or the average of probabilities (soft voting) as the final prediction. Here's how you can do it:

# %%
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[
        ('log_reg', l_regression),
        ('xgb', xgb_classifier)
    ],
    voting='soft'
)

ensemble_model.fit(X_train_balanced, y_train_balanced)

print(classification_report(y_true=y_test_balanced, y_pred=ensemble_model.predict(X_test_balanced)))

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(ensemble_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
print(f'Cross-Validation F1 Score: {scores.mean()}')

# %% [markdown]
# ## StackingClassifier
# 
# Use the predictions of one set of models as input features for another model to learn how to combine them optimally. A stacking classifier combines the predictions of base models and uses a meta-model to find the optimal combination of them. This is helpful when you want a model to learn how to best combine the predictions from LogisticRegression and XGBoost.

# %%
from sklearn.ensemble import StackingClassifier

base_models = [('l_regression', l_regression), ('xgb_classifier', xgb_classifier)]

stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=rf
)
stacked_model.fit(X_train_balanced, y_train_balanced)
print(classification_report(y_true=y_test_balanced, y_pred=stacked_model.predict(X_test_balanced)))

# %%
scores = cross_val_score(stacked_model, X_train_balanced, y_train_balanced, cv=5, scoring='f1')
print(f'Cross-Validation F1 Score: {scores.mean()}')

# %% [markdown]
# # End notes
# While doing this project, I did the following mistakes:
#  * **Not identifying the patterns of the spam and non-spam emails**: I should've first identified the pattern of token(s)' occurrences with both spam and non-spam kind of emails.
#  * **Simulating randomness**: Instead of randomness by going for `df.sample`, I should've gone for suitable data augmentation technique.
#  * **A good dataset**: This dataset is way too small for email classification, for spam classification, there're also emoji, phishing techniques involved as well, and other vice versa.

# %% [markdown]
# Here are a few additional points where improvements could be made or explored further:
# 
# ---
# 
# ### **1. Lack of Data Preprocessing Evaluation**
#    - **Possible Issue**: Preprocessing choices like tokenization, lemmatization, and threshold-based inclusion may not align optimally with the nature of the dataset.
#    - **Suggested Improvement**: Evaluate whether preprocessing steps:
#      - Retain meaningful information.
#      - Don't inadvertently remove critical spam indicators (e.g., numbers, special characters, or URLs).
#      - Include domain-specific stopwords (e.g., "click", "free", "offer" for spam classification).
#    - Experiment with different preprocessing pipelines and validate their impact on model performance.
# 
# ---
# 
# ### **2. Overlooking Model Interpretability**
#    - **Possible Issue**: The project could benefit from analyzing which features or tokens contribute most to predictions. Without this, we're working in a "black box" mode.
#    - **Suggested Improvement**:
#      - Use **SHAP (SHapley Additive exPlanations)** or **LIME (Local Interpretable Model-agnostic Explanations)** to identify the importance of specific tokens.
#      - Analyze token contributions for spam vs. non-spam predictions to uncover new insights into patterns and biases.
# 
# ---
# 
# ### **3. Focusing Solely on SMOTE Oversampling**
#    - **Possible Issue**: While SMOTE balances the dataset, it creates synthetic samples that might not accurately represent the data distribution, especially for high-dimensional data like text.
#    - **Suggested Improvement**:
#      - Compare SMOTE results with **undersampling**, **class-weight adjustments**, or **other oversampling techniques** like ADASYN.
#      - Incorporate **data augmentation** methods to increase diversity without relying solely on resampling algorithms.
# 
# ---
# 
# ### **4. Model-Specific Overfitting Risk**
#    - **Possible Issue**: With ensemble models like XGBoost and RandomForest, there's a chance of overfitting to the balanced dataset if hyperparameters aren’t carefully tuned.
#    - **Suggested Improvement**:
#      - Use a **validation set** alongside cross-validation to detect overfitting.
#      - Regularize the model using appropriate parameters (e.g., `gamma` for XGBoost, `min_samples_split` for RandomForest).
# 
# ---
# 
# ### **5. Overlooking Ensemble Diversity**
#    - **Possible Issue**: While the ensemble approach with logistic regression and XGBoost is promising, both models might share similar biases.
#    - **Suggested Improvement**:
#      - Increase diversity in your ensemble by including fundamentally different algorithms (e.g., Naïve Bayes or SVM with custom kernels).
#      - Use a meta-model (e.g., stacking with a Logistic Regression or a LightGBM as a meta-learner) to combine predictions more effectively.
# 
# ---
# 
# ### **6. Overreliance on Accuracy and F1 Scores**
#    - **Possible Issue**: Metrics like accuracy and F1-score might not fully capture the model's ability to differentiate between spam and non-spam emails.
#    - **Suggested Improvement**:
#      - Incorporate other evaluation metrics like:
#        - **Precision-Recall AUC**: Especially important for imbalanced datasets.
#        - **False Positive Rate (FPR)**: To check if the spam filter mistakenly classifies valid emails as spam.
#
# ---
#
# ### **7. Dataset Size vs. Complexity Trade-off**
#    - **Possible Issue**: Applying highly complex models like XGBoost on a small dataset might not fully leverage their capabilities.
#    - **Suggested Improvement**:
#      - Simplify the model if scaling up the dataset isn’t an option (e.g., using simpler classifiers like Logistic Regression or Decision Trees).
#      - Alternatively, explore transfer learning to reduce dependency on dataset size.
#
# ---
#
# ### **8. Limited Real-World Email Features**
#    - **Possible Issue**: Spam classification often involves metadata beyond email text, such as:
#      - Sender reputation.
#      - Email headers (e.g., "From", "Reply-To").
#      - Attached links and domains.
#    - **Suggested Improvement**:
#      - Enrich the dataset with such features if possible.
#      - Use a multi-modal approach where text data and metadata are both included as inputs.
#
# ---
#
# ### **9. Lack of Robustness Testing**
#    - **Possible Issue**: Without robustness testing, the model might perform poorly when applied to unseen, real-world datasets.
#    - **Suggested Improvement**:
#      - Test the model on an **external dataset** or simulate real-world scenarios, such as:
#        - Emails with heavy use of emojis, URLs, or phishing techniques.
#        - Emails in different languages or with mixed character sets.
#
# ---
#
# ### **10. Not Exploring Sequential Patterns**
#    - **Possible Issue**: Emails have natural sequences of words and phrases that models like Bag-of-Words or TF-IDF might ignore.
#    - **Suggested Improvement**:
#      - Experiment with sequential models like **LSTMs**, **GRUs**, or **Transformers** to capture temporal dependencies.
#      - Alternatively, explore **n-gram features** for a middle ground.
#
# ---
#
# By addressing these points, you can further refine your project and make it robust, scalable, and insightful.

# %% [markdown]
# | Platform   | Link                                           |
# |------------|------------------------------------------------|
# | GitHub     | [JackTheProgrammer](https://github.com/JackTheProgrammer) |
# | LinkedIn   | [Fawad Awan](https://www.linkedin.com/in/fawad-awan-893a58171/) |