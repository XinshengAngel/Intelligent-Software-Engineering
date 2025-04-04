import pandas as pd
import numpy as np
import re
import torch
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from transformers import AutoTokenizer, AutoModel
import time
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

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
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

model_name = "huawei-noah/TinyBERT_General_4L_312D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_tinyBERT_embeddings(texts, batch_size=16, max_length=128):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            summed = torch.sum(last_hidden * mask, 1)
            summed_mask = torch.clamp(mask.sum(1), min=1e-9)
            batch_embeddings = (summed / summed_mask).cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

projects = ['pytorch', 'tensorflow', 'keras', 'incubator-mxnet', 'caffe']
REPEAT = 30

for project in projects:
    start_time = time.time()
    path = f'{project}.csv'
    pd_all = pd.read_csv(path)
    pd_all = pd_all.sample(frac=1, random_state=999)
    pd_all['Title+Body'] = pd_all.apply(lambda row: row['Title'] + '. ' + row['Body'] if pd.notna(row['Body']) else row['Title'], axis=1)
    pd_tplusb = pd_all.rename(columns={"Unnamed: 0": "id", "class": "sentiment", "Title+Body": "text"})
    data = pd_tplusb.copy().fillna('')
    text_col = 'text'

    data[text_col] = data[text_col].apply(remove_html)
    data[text_col] = data[text_col].apply(remove_emoji)
    data[text_col] = data[text_col].apply(remove_stopwords)
    data[text_col] = data[text_col].apply(clean_str)

    texts = data[text_col].tolist()
    sentiments = data['sentiment'].tolist()

    X_all = get_tinyBERT_embeddings(texts, batch_size=16, max_length=128)
    y_all = np.array(sentiments)

    precisions = []
    recalls = []
    f1_scores = []

    for repeated_time in range(REPEAT):
        indices = np.arange(X_all.shape[0])
        train_index, test_index = train_test_split(indices, test_size=0.3, random_state=repeated_time)
        X_train = X_all[train_index]
        X_test = X_all[test_index]
        y_train = y_all[train_index]
        y_test = y_all[test_index]

        mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', alpha=1e-4,
                            learning_rate_init=0.001, max_iter=200, random_state=42)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

        precisions.append(prec)
        recalls.append(rec)
        f1_scores.append(f1)

    final_precision = np.mean(precisions)
    final_recall = np.mean(recalls)
    final_f1 = np.mean(f1_scores)

    elapsed = time.time() - start_time
    print("=== MLP + TinyBERT Experiment Results ===")
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
