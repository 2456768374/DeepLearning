import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from paddle.io import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import paddle.nn as nn
import paddle
from tqdm import tqdm
import copy


class ClickDataset(Dataset):
    def __init__(self, data):
        super(ClickDataset, self).__init__()
        self.label = data['isClick'].values.astype('int64')
        self.features = data.drop(columns=['id', 'date', 'isClick']).values.astype('int64')

    def __getitem__(self, idx):
        label = self.label[idx]
        features = self.features[idx]
        return features, label

    def __len__(self):
        return len(self.label)


class ResidualUnit(nn.Layer):
    def __init__(self, input_size):
        super(ResidualUnit, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.fc1(x)
        output = self.relu(output)
        output = self.fc2(output)
        output = output + x
        return output


class DeepCrossing(nn.Layer):
    def __init__(self, embedding_sizes):
        super(DeepCrossing, self).__init__()
        self.embeddings = nn.LayerList(
            [nn.Embedding(num_embeddings=categories + 1, embedding_dim=size) for categories, size in embedding_sizes]
        )
        n_emb = sum(e[1] for e in embedding_sizes)
        self.fc1 = nn.Linear(n_emb, 32)
        self.res1 = ResidualUnit(32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, features):
        features = [embedding(features[:, i]) for i, embedding in enumerate(self.embeddings)]
        features = paddle.concat(features, axis=1)
        features = self.fc1(features)
        features = self.res1(features)
        pred = self.fc2(features)
        return pred


def fit(model, train_loader, optimizer, criterion, device):
    model.train()
    pred_list = []
    label_list = []
    for features, label in tqdm(train_loader):
        pred = model(features)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        pred_list.extend(pred[:, 1].cpu().detach().numpy())
        label_list.extend(label.cpu().detach().numpy())
    score = roc_auc_score(label_list, pred_list)
    return score


def validate(model, val_loader, device):
    model.eval()
    pred_list = []
    label_list = []
    for features, label in tqdm(val_loader):
        pred = model(features)
        pred_list.extend(pred[:, 1].cpu().detach().numpy())
        label_list.extend(label.cpu().detach().numpy())
    score = roc_auc_score(label_list, pred_list)
    return score


def main():
    data_path = 'data/广告点击率预估挑战赛.csv'  # 修改为你的文件路径
    data = pd.read_csv(data_path)
    data['gender'] = data['gender'].map({'Male': 1, 'Female': 0})
    data = data.fillna(-1)
    data['hour'] = data['date'].map(lambda x: int(x[-5:-3]))
    data['minute'] = data['date'].map(lambda x: int(x[-2:]))
    data['day'] = data['date'].map(lambda x: int(x[3:5]))
    device = paddle.device.get_device()
    embedded_cols = ['user_id', 'product', 'campaign_id', 'webpage_id',
                     'product_category_id', 'user_group_id', 'gender', 'age_level',
                     'user_depth', 'var_1', 'hour', 'minute', 'day']
    embedded_cols_dict = {col: data[col].nunique() for col in embedded_cols}
    embedding_sizes = [(n_categories, min(50, (n_categories + 1) // 2)) for _, n_categories in
                       embedded_cols_dict.items()]
    for cat_feat in embedded_cols:
        data[cat_feat] = LabelEncoder().fit_transform(data[cat_feat])

    train_data = data[data['day'] != 4]
    val_data = data[data['day'] == 4]
    train_dataset = ClickDataset(train_data)
    valid_dataset = ClickDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = DeepCrossing(embedding_sizes).to(device)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)
    criterion = nn.loss.CrossEntropyLoss()

    best_val_score = float('-inf')
    last_improve = 0
    best_model = None

    for epoch in range(10):
        train_score = fit(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, valid_loader, device)
        if val_score > best_val_score:
            best_val_score = val_score
            best_model = copy.deepcopy(model)
            last_improve = epoch
            improve = '*'
        else:
            improve = ''

        if epoch - last_improve > 3:
            break

        print(
            f'Epoch: {epoch} Train Score: {train_score}, Valid Score: {val_score} {improve}'
        )


if __name__ == "__main__":
    main()
