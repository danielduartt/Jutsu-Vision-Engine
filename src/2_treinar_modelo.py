import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print("Carregando dataset...")
try:
    df = pd.read_csv('dataset_jutsu.csv')
except FileNotFoundError:
    print("Erro: Arquivo dataset_jutsu.csv não encontrado. Rode o coletor de dados primeiro.")
    exit()

X = df.drop('target', axis=1)
y = df['target']

# Divisão em Treino e Teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Treinando modelo XGBoost Multiclasse...")
model = xgb.XGBClassifier(
    objective='multi:softprob', 
    num_class=4,                
    n_estimators=150,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# O model.fit() e a avaliação continuam exatamente iguais!
model.fit(X_train, y_train)

print("\nAvaliação do Modelo:")
preds = model.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, preds) * 100:.2f}%")
print(classification_report(y_test, preds))

# Exportar modelo para ser usado em tempo real
model.save_model('modelo_kage_bunshin.json')
print("Modelo salvo como 'modelo_kage_bunshin.json'. Pronto para produção!")