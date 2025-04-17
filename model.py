import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import datasets


num_samples = 100
num_features = 3
num_classes = 6

# train_dataset = TensorDataset(datasets.X, datasets.Y)
# test_dataset = TensorDataset(datasets.Xtest, datasets.Ytest)

# batch_size = 6
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size)


input_size = num_features
hidden_size = 64

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 128)  # 2 входных нейрона
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)    # 6 выходных нейронов
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

# Генерация данных с нужными диапазонами
def generate_data(num_samples=20000):
    np.random.seed(42)
    
    # Входные значения от 300 до 600 (2 признака)
    X = np.random.uniform(300, 600, (num_samples, 2))
    
    # Целевые диапазоны для 6 выходов
    target_ranges = [
        (-25, 134),   # Выход 1
        (0, 83),      # Выход 2
        (-89, 83),    # Выход 3
        (0, 80),      # Выход 4
        (-85, 85),    # Выход 5
        (-20, 31)     # Выход 6
    ]
    
    # Генерация выходных значений
    y = np.zeros((num_samples, 6))
    for i in range(6):
        low, high = target_ranges[i]
        y[:, i] = np.random.uniform(low, high, num_samples)
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

# Функция обучения
def train_model(model, train_loader, val_loader, epochs=150, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Сохранение лучшей модели
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model_2input.pth')
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

model = NeuralNetwork()
# Функция для предсказания
def predict(input_values):
    model.load_state_dict(torch.load('best_model_2input.pth'))
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_values).unsqueeze(0)
        prediction = model(input_tensor)
        return prediction.squeeze().numpy()

# Тестирование
# test_inputs = [
#     [300, 450],
#     [400, 500],
#     [600, 350]
# ]

# print("\nПримеры предсказаний:")
# for inp in test_inputs:
#     pred = predict(model, inp)
#     formatted_pred = [f"{x:.2f}" for x in pred]
#     print(f"Вход: {inp} -> Выход: {formatted_pred}")

# # Оценка на тестовом наборе
# model.eval()
# test_loss = 0.0
# criterion = nn.MSELoss()

# with torch.no_grad():
#     for inputs, targets in test_loader:
#         outputs = model(inputs)
#         test_loss += criterion(outputs, targets).item()

#print(f"\nTest Loss: {test_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    # Подготовка данных
    X, y = generate_data()
    dataset = TensorDataset(X, y)

    # Разделение данных (70% train, 15% val, 15% test)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Инициализация и обучение модели
    model = NeuralNetwork()
    train_model(model, train_loader, val_loader, epochs=200)

    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_model_2input.pth'))
