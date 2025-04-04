import pandas as pd
import numpy as np
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.svm import LinearSVC
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from scipy import stats

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

NLTK_stop_words_list = stopwords.words('english')
custom_stop_words_list = ['...']
final_stop_words_list = NLTK_stop_words_list + custom_stop_words_list

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in final_stop_words_list])

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
REPEAT = 30

for project in projects:
    path = f'{project}.csv'
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)
    pd_all['Title+Body'] = pd_all.apply(
        lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'],
        axis=1
    )
    pd_tplusb = pd_all.rename(columns={
        "Unnamed: 0": "id",
        "class": "sentiment",
        "Title+Body": "text"
    })

    data = pd_tplusb.copy().fillna('')
    text_col = 'text'
    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    params = {'C': [0.1, 1, 10]}
    precisions = []
    recalls = []
    f1_scores = []

    for repeated_time in range(REPEAT):
        indices = np.arange(data.shape[0])
        train_index, test_index = train_test_split(
            indices, test_size=0.3, random_state=repeated_time
        )
        train_text = data[text_col].iloc[train_index]
        test_text = data[text_col].iloc[test_index]
        y_train = data['sentiment'].iloc[train_index]
        y_test = data['sentiment'].iloc[test_index]

        tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000
        )
        X_train = tfidf.fit_transform(train_text)
        X_test = tfidf.transform(test_text)

        clf = LinearSVC(
            class_weight='balanced',
            max_iter=10000
        )

        grid = GridSearchCV(
            clf,
            params,
            cv=5,
            scoring='roc_auc'
        )
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_

        y_pred = best_clf.predict(X_test)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precisions.append(prec)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recalls.append(rec)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_scores.append(f1)

    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1_scores)

    print("=== SVM + TF-IDF Experiment Results ===")
    print(f"Project:                {project}")
    print(f"Average Precision:      {final_precision:.4f}")
    print(f"Average Recall:         {final_recall:.4f}")
    print(f"Average F1 score:       {final_f1:.4f}")

    baseline_file = f'baseline_results_{project}.csv'
    baseline_data = pd.read_csv(baseline_file)
    baseline_precisions = baseline_data['precision'].values
    baseline_recalls    = baseline_data['recall'].values
    baseline_f1_scores  = baseline_data['f1'].values

    t_stat_prec, p_val_prec = stats.ttest_rel(precisions, baseline_precisions)
    t_stat_rec, p_val_rec = stats.ttest_rel(recalls, baseline_recalls)
    t_stat_f1, p_val_f1 = stats.ttest_rel(f1_scores, baseline_f1_scores)

    print("\n--- Statistical Test Results ---")
    print(f"Precision: t-statistic = {t_stat_prec:.4f}, p-value = {p_val_prec:.4f}")
    print(f"Recall:    t-statistic = {t_stat_rec:.4f}, p-value = {p_val_rec:.4f}")
    print(f"F1 Score:  t-statistic = {t_stat_f1:.4f}, p-value = {p_val_f1:.4f}")
