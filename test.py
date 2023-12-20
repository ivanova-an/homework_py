import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Загрузка данных
data = pd.read_csv('flight_delays.csv')

# Преобразование категориальных признаков
data = pd.get_dummies(data, columns=['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest'])

# Разделение данных на X и y
X = data.drop('dep_delayed_15min', axis=1)
y = data['dep_delayed_15min'].map({'N': 0, 'Y': 1})

# Разделение на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация модели градиентного бустинга
model = GradientBoostingClassifier()

# Подбор гиперпараметров с использованием GridSearch
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

grid_search = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
grid_search.fit(X_train, y_train)

# Получение наилучших параметров
best_params = grid_search.best_params_

# Обучение модели с лучшими параметрами
best_model = GradientBoostingClassifier(**best_params)
best_model.fit(X_train, y_train)

# Предсказание на тестовом наборе
y_pred = best_model.predict_proba(X_test)[:, 1]

# Оценка roc_auc_score
roc_auc = roc_auc_score(y_test, y_pred)

print("Best Parameters:", best_params)
print("roc_auc_score:", roc_auc)
