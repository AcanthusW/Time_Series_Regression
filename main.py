import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
import utils

def main():
    # load dataset 
    data = utils.load_data("data/dataset.csv")  # 替换为你的数据路径
    X = data.drop("target_column", axis=1)
    y = data["target_column"]
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = utils.train_random_forest(X_train, y_train)
    
    # 评估模型
    mse = utils.evaluate_model(model, X_test, y_test)
    print(f"Model Mean Squared Error: {mse}")

    # 保存模型到 results 目录
    joblib.dump(model, "results/trained_model.pkl")
    print("Model saved to results/trained_model.pkl")

if __name__ == "__main__":
    main()
