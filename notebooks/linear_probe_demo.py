import sys
sys.path.extend(["/home/user/wangtao/prov-gigapath/TITAN"])
import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel
from titan.eval_linear_probe import train_and_evaluate_logistic_regression_with_val
from titan.utils import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
os.environ["OMP_NUM_THREADS"] = "8"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model from huggingface
model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
model = model.to(device)

# load task configs
with open('../datasets/config_tcga-ot.yaml', 'r') as file:
    task_config = yaml.load(file, Loader=yaml.FullLoader)
target = task_config['target']
label_dict = task_config['label_dict']

# load pre-extracted TITAN slide embeddings for TCGA
import pickle
from huggingface_hub import hf_hub_download
# slide_feature_path = hf_hub_download(
#     "MahmoodLab/TITAN",
#     filename="TCGA_TITAN_features.pkl",
# )
slide_feature_path = "/home/user/sngp/TCGA-OT/Patch512/TITAN/h5_files_slide/tcga_nsclc_titan_slide_embedding.pkl"
with open(slide_feature_path, 'rb') as file:
    data = pickle.load(file)
embeddings_df = pd.DataFrame({'slide_id': data['filenames'], 'embeddings': list(data['embeddings'][:])})
# embeddings_df["embeddings"] = embeddings_df["embeddings"].apply(lambda x: x / np.linalg.norm(x))

# load splits
train_split = pd.read_csv('../datasets/tcga-ot_train.csv')
train_df = pd.merge(embeddings_df, train_split, on='slide_id')
val_split = pd.read_csv('../datasets/tcga-ot_val.csv')
val_df = pd.merge(embeddings_df, val_split, on='slide_id')
test_split = pd.read_csv('../datasets/tcga-ot_test.csv')
test_df = pd.merge(embeddings_df, test_split, on='slide_id')

print(f"len(train_df): {len(train_df)}, len(val_df): {len(val_df)}, len(test_df): {len(test_df)}")

train_data = np.stack(train_df.embeddings.values)
train_labels = train_df[target].apply(lambda x: label_dict[x]).values
val_data = np.stack(val_df.embeddings.values)
val_labels = val_df[target].apply(lambda x: label_dict[x]).values
test_data = np.stack(test_df.embeddings.values)
test_labels = test_df[target].apply(lambda x: label_dict[x]).values

# print("train df", pd.DataFrame(train_labels).value_counts())
# print("val df", pd.DataFrame(val_labels).value_counts())
# print("test df", pd.DataFrame(test_labels).value_counts())

# log_spaced_values = np.logspace(np.log10(10e-2), np.log10(10e2), num=1)
log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45)
test_slide_id, test_patient_id = None, None
### get label_dict

prefix_path = Path("/home/user/wangtao/prov-gigapath/TITAN")

with open(prefix_path / 'datasets/config_tcga-ot.yaml', 'r') as file:
    task_config = yaml.load(file, Loader=yaml.FullLoader)
target = task_config['target']
label_dict = task_config['label_dict']
num_classes = 46
results, outputs = train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels, test_slide_id, test_patient_id, log_spaced_values=log_spaced_values)

invert_label_dict = {v: k for k, v in label_dict.items()}
per_class_correct = np.zeros(num_classes)
per_class_total = np.zeros(num_classes)
predictions = outputs['probs'].argmax(axis=1)

for pred, label in zip(predictions, outputs['targets']):
    per_class_correct[label] += (pred == label)
    per_class_total[label] += 1

per_class_acc = per_class_correct / np.maximum(per_class_total, 1)

print("\nPer-class accuracies:")
for i in range(num_classes):
    if per_class_total[i] > 0:
        print(
            f"Class {invert_label_dict[i]}: {per_class_acc[i]:.3f} ({int(per_class_correct[i])}/{int(per_class_total[i])})")
print(per_class_acc)
# to use the default setting from our paper use the default value for searching C (log_spaced_values = np.logspace(np.log10(10e-6), np.log10(10e5), num=45))
# results = train_and_evaluate_logistic_regression_with_val(train_data, train_labels, val_data, val_labels, test_data, test_labels)
for key, value in results.items():
    print(f"{key.split('/')[-1]: <12}: {value:.4f}")