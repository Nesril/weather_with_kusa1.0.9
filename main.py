import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
from inspect import signature
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from kusa.client import SecureDatasetClient
import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score,f1_score
from sklearn.preprocessing import label_binarize
import numpy as np

# 🔧 Framework flag: sklearn | tensorflow | pytorch
TRAINING_FRAMEWORK = "tensorflow"

load_dotenv()

# ✅ Framework-aware training factory
def train_model_factory(framework="sklearn", model_class=None, fixed_params=None):
    fixed_params = fixed_params or {}
    print("framework ",framework)
    if framework == "sklearn":
        def train_model(X, y, **params):
            sig = signature(model_class.__init__)
            accepted = set(sig.parameters.keys())
            valid = {k: v for k, v in {**fixed_params, **params}.items() if k in accepted}
            return model_class(**valid).fit(X, y)
        return train_model

    elif framework == "tensorflow":
        def train_model(X, y, X_val=None, y_val=None, **params):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X.shape[1],)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(
                loss='binary_crossentropy',
                optimizer=params.get("optimizer", "adam"),
                metrics=['accuracy']
            )
            model.fit(
                X, y,
                validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
                epochs=params.get("epochs", 10),
                verbose=1
            )
            return model
        return train_model


    elif framework == "pytorch":

        class SimpleNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.net(x)

        def train_model(X, y, X_val=None, y_val=None, **params):
            input_dim = X.shape[1]
            model = SimpleNN(input_dim)
            loss_fn = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get("lr", 0.001))

            loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

            for epoch in range(params.get("epochs", 10)):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    loss.backward()
                    optimizer.step()

            return model
        return train_model


    else:
        raise ValueError("Unsupported framework selected")

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    

def plot_precision_recall_curve(y_true, y_proba, title="Precision-Recall Curve"):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={avg_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    


def plot_threshold_analysis(y_true, y_proba, title="Threshold Analysis"):
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p, r = precision_recall_curve(y_true, preds)[:2]
        precisions.append(p[1])
        recalls.append(r[1])
        f1s.append(f1_score(y_true, preds))

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions, label="Precision", color="blue")
    plt.plot(thresholds, recalls, label="Recall", color="green")
    plt.plot(thresholds, f1s, label="F1 Score", color="red")
    plt.axvline(x=0.5, linestyle='--', color='gray', label="Threshold = 0.5")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def main():
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    PUBLIC_ID="Xv8xvecyaKEU"
    SECRET_KEY = os.getenv("SECRET_KEY")
    print("PUBLIC_ID ",PUBLIC_ID)
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

    initialization = client.initialize()
    print("📦 Fetching and decrypting dataset...")
    client.fetch_and_decrypt_batch(batch_size=500, batch_number=1)

    print("⚙️ Configuring preprocessing...")
    client.configure_preprocessing({ 
        "tokenizer": "spacy",
        "stopwords": True,
        "reduction": "tfidf",  # Optional for NLP; skip or change for tabular tfidf,pca or tfidf_pca
        "target_column": "RainTomorrow",
        "target_encoding": "auto"  # or dict, or "none" dict means like  {"ham": 0, "spam": 1}

    })
    client.run_preprocessing()

    print(f"🎯 Building training function for {TRAINING_FRAMEWORK}...")
    if TRAINING_FRAMEWORK == "sklearn":
        train_model = train_model_factory("sklearn", model_class=RandomForestClassifier)
        hyperparams = {"n_estimators": 200, "learning_rate": 0.05,"class_weight": "balanced"}
    elif TRAINING_FRAMEWORK == "tensorflow":
        train_model = train_model_factory("tensorflow")
        hyperparams = {"epochs": 10, "optimizer": "adam"}
    elif TRAINING_FRAMEWORK == "pytorch":
        train_model = train_model_factory("pytorch")
        hyperparams = {"epochs": 10, "lr": 0.001}

    print("🚀 Training model...")
    client.train(
         user_train_func=train_model, 
         hyperparams=hyperparams, target_column="RainTomorrow",
         task_type="classification", framework=TRAINING_FRAMEWORK, 
    )

    print("📈 Evaluating model...")
    results = client.evaluate()
    print("\n✅ Evaluation Accuracy:", results["accuracy"])
    print("📊 Classification Report:\n", results["report"])

    print("📉 Visualizing confusion matrix...")
    

    y_true = client._SecureDatasetClient__y_val
    y_pred = client.predict(client._SecureDatasetClient__X_val)
    
    # 🔥 Confusion Matrix
    plot_confusion_matrix(y_true, y_pred)

    # 🔥 Precision-Recall Curve
    if TRAINING_FRAMEWORK == "sklearn":
        y_proba = client._SecureDatasetClient__trained_model.predict_proba(client._SecureDatasetClient__X_val)[:, 1]
    elif TRAINING_FRAMEWORK == "tensorflow":
        y_proba = client._SecureDatasetClient__trained_model.predict(client._SecureDatasetClient__X_val).flatten()
    elif TRAINING_FRAMEWORK == "pytorch":
        model = client._SecureDatasetClient__trained_model
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(client._SecureDatasetClient__X_val.values, dtype=torch.float32)
            y_proba = model(inputs).numpy().flatten()

    plot_precision_recall_curve(y_true, y_proba)

    # Show thresholds
    plot_threshold_analysis(y_true, y_proba)

    print("💾 Saving trained model...")
    
    client.save_model(f"secure_model_{TRAINING_FRAMEWORK}.model")

    print("\n✅ Done! 🎉")

    # print("\n🧪 Preview:\n", initialization.get("preview").head())

if __name__ == "__main__":
    main()


 

# precesion recall curve