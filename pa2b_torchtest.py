import pandas as pd
from tqdm import trange
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, Subset

# Loading datasets
higher_ed_df = pd.read_csv('dataset/subset_higher_ed_ednum.csv')
some_college_df = pd.read_csv('dataset/subset_some_college_ednum.csv')
less_than_hs_df = pd.read_csv('dataset/subset_less_than_hs_ednum.csv')

# Dropping 'fnlgwt' and 'Education' features from all datasets as they are not useful or duplicated
higher_ed_df = higher_ed_df.drop(['fnlgwt', 'Education'], axis=1)
some_college_df = some_college_df.drop(['fnlgwt', 'Education'], axis=1)
less_than_hs_df = less_than_hs_df.drop(['fnlgwt', 'Education'], axis=1)


# Encoding categorical features in all datasets
def encode_features(df):
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    return df, label_encoders


higher_ed_df_encoded, higher_ed_encoders = encode_features(higher_ed_df.copy())
some_college_df_encoded, some_college_encoders = encode_features(some_college_df.copy())
less_than_hs_df_encoded, less_than_hs_encoders = encode_features(less_than_hs_df.copy())

# Defining the input features and target variable for each subset
X_higher_ed = higher_ed_df_encoded.drop('Income', axis=1)
y_higher_ed = higher_ed_df_encoded['Income']

X_some_college = some_college_df_encoded.drop('Income', axis=1)
y_some_college = some_college_df_encoded['Income']

X_less_than_hs = less_than_hs_df_encoded.drop('Income', axis=1)
y_less_than_hs = less_than_hs_df_encoded['Income']


class CustomANN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(CustomANN, self).__init__()
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.ReLU()]

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def tune_ann_pytorch(X, y, hidden_layer_sizes, title, device, k_folds=5, epochs=1000):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    dataset = TensorDataset(X_tensor, y_tensor)

    kfold = KFold(n_splits=k_folds, shuffle=True)

    results = {}

    for size in hidden_layer_sizes:
        fold_accuracies = []

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
            train_subsampler = Subset(dataset, train_ids)
            test_subsampler = Subset(dataset, test_ids)

            train_loader = DataLoader(train_subsampler, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_subsampler, batch_size=64, shuffle=False)

            model = CustomANN(input_size=X.shape[1], hidden_layers=size, output_size=len(np.unique(y)))
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            best_accuracy = 0
            pbar = trange(epochs, desc=f'Fold {fold}, Size {size}', leave=True)
            for _ in pbar:
                # Training
                model.train()
                train_loss = 0
                for data, target in train_loader:
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                accuracy = correct / total
                pbar.set_postfix(train_loss=f'{train_loss:.4f}', val_accuracy=f'{accuracy:.4f}')
                if accuracy > best_accuracy:
                    best_accuracy = accuracy

            fold_accuracies.append(best_accuracy)

        mean_accuracy = np.mean(fold_accuracies)
        results[size] = mean_accuracy
        print(f'Hidden Layer Size: {size}, Mean Validation Accuracy: {mean_accuracy}')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot([str(size) for size in hidden_layer_sizes], list(results.values()), marker='o')
    plt.title(f'5-Fold CV Accuracy vs. Hidden Layer Sizes for {title}')
    plt.xlabel('Number of Neurons in Hidden Layer(s)')
    plt.ylabel('CV Mean Accuracy (%)')
    plt.grid(True)
    plt.show()


# import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

hidden_layer_sizes = [(50,), (100,), (100, 50), (100, 100), (50, 50, 50)]
tune_ann_pytorch(X_higher_ed, y_higher_ed, hidden_layer_sizes, "Higher Education Subset", device)
tune_ann_pytorch(X_some_college, y_some_college, hidden_layer_sizes, "Some College Subset", device)
tune_ann_pytorch(X_less_than_hs, y_less_than_hs, hidden_layer_sizes, "Less than High School Subset", device)
