import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datasets import X,Y
import matplotlib.pyplot as plt
    
def normData():
# 1. Подготовка данных

    # Нормализация данных 
    X_mean, X_std = X.mean(dim=0), X.std(dim=0)
    Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)
    X_normalized = (X - X_mean) / X_std
    Y_normalized = (Y - Y_mean) / Y_std
    noisy_X = X + torch.randn_like(X) * 5
    return noisy_X, Y_normalized

# 2. Создание модели
class TwoToTwoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)  # 2 входа -> 32 нейрона
        self.fc2 = nn.Linear(32, 32) # скрытый слой
        self.fc3 = nn.Linear(32, 2)  # 2 выхода
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

model = TwoToTwoNet()

# 3. Обучение модели
def train(X_normalized,Y_normalized):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = TensorDataset(X_normalized, Y_normalized)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Цикл обучения
    losses = []
    for epoch in range(500):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


def predict(input_values):
    model.load_state_dict(torch.load('newModel6.pth'))
    # print(input_values)
    # model.eval()
    # with torch.no_grad():
    #     input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
    #     prediction = model(input_tensor)
    #     return prediction.squeeze().numpy()
    model.eval()
    with torch.no_grad():

        X_mean, X_std = X.mean(dim=0), X.std(dim=0)
        Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)
        test_inputs = torch.tensor([[input_values[0], input_values[1]]], dtype=torch.float32)
        test_inputs = (test_inputs - X_mean) / X_std  # Нормализация
        
        predictions = model(test_inputs)
        predictions = predictions * Y_std + Y_mean  # Денормализация
        
        return predictions.tolist()
        # print("\nПредсказания для новых данных:")
        # for inp, pred in zip(test_inputs, predictions):
        #     original_input = inp * X_std + X_mean
        #     print(f"Вход: {original_input.numpy()} -> Предсказание: {pred.numpy()}")



def test():
# 5. Проверка на тестовых данных
    model.eval()
    with torch.no_grad():

        X_mean, X_std = X.mean(dim=0), X.std(dim=0)
        Y_mean, Y_std = Y.mean(dim=0), Y.std(dim=0)
        test_inputs = torch.tensor([[250, 260], [300, 210]], dtype=torch.float32)
        test_inputs = (test_inputs - X_mean) / X_std  # Нормализация
        
        predictions = model(test_inputs)
        predictions = predictions * Y_std + Y_mean  # Денормализация
        
        #print("\nПредсказания для новых данных:")
        for inp, pred in zip(test_inputs, predictions):
            original_input = inp * X_std + X_mean
            #print(f"Вход: {original_input.numpy()} -> Предсказание: {pred.numpy()}")

# 6. Сохранение модели


if __name__ == "__main__":
    X_normalized, Y_normalized = normData()
    train(X_normalized,Y_normalized)
    test()
    torch.save(model.state_dict(), 'newModel6.pth')
