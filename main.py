import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Hàm tải và xử lý dữ liệu ban đầu
def load_and_prepare_data():
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(period="max")
    hist = hist.loc["1990-01-01":].copy()
    st.write("### Dữ liệu sau khi ban đầu:")
    st.write(hist.head(5))
    # Reset index để chuyển Date thành cột và bỏ cột Date
    hist.reset_index(inplace=True)
    hist.drop(columns=["Date"], inplace=True)

    hist.drop(columns=["Dividends", "Stock Splits"], inplace=True)
    hist["Tomorrow_Close"] = hist["Close"].shift(-1)
    latest_row = hist.tail(3)
    latest_features = latest_row[["Open", "High", "Low", "Close", "Volume"]]
    hist.dropna(inplace=True)

    st.write("### Dữ liệu sau khi loại bỏ NaN (5 dòng đầu tiên):")
    st.write(hist.head(5))

    return hist, latest_features

# Hàm phân tích dữ liệu ban đầu
def analyze_initial_data(hist):
    features = ["Open", "High", "Low", "Close", "Volume"]
    train = hist.iloc[:-100]
    X_train = train[features]
    y_train = train["Tomorrow_Close"]

    # 1. Thống kê mô tả
    st.write("### Thống kê mô tả của dữ liệu")
    st.write(X_train.describe())

    # 2. Phân tích đơn biến
    st.write("### Phân tích đơn biến")
    
    # 2.1. Histogram để xem phân bố
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(X_train[feature], bins=50, color="blue", alpha=0.7)
        ax.set_title(f"Phân bố của {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Tần suất")
        ax.grid(True)
        st.pyplot(fig)

    # # 2.2. Boxplot để phát hiện ngoại lai
    # for feature in features:
    #     fig, ax = plt.subplots(figsize=(8, 5))
    #     sns.boxplot(x=X_train[feature], ax=ax, color="lightblue")
    #     ax.set_title(f"Boxplot của {feature}")
    #     ax.grid(True)
    #     st.pyplot(fig)

    # 2.3. Scatter plot với Tomorrow_Close
    for feature in features:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X_train[feature], y_train, alpha=0.5, c="blue")
        ax.set_title(f"{feature} vs Tomorrow_Close")
        ax.set_xlabel(feature)
        ax.set_ylabel("Tomorrow_Close")
        ax.grid(True)
        st.pyplot(fig)

    # 3. Phân tích nhị biến
    st.write("\n### Phân tích nhị biến")
    
    # 3.1. Heatmap tương quan
    train_with_target = X_train.copy()
    train_with_target["Tomorrow_Close"] = y_train
    corr_matrix = train_with_target.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Heatmap Tương Quan")
    st.pyplot(fig)

    # 3.2. Pairplot để xem toàn bộ mối quan hệ
    st.write("### Pairplot giữa các đặc trưng")
    pair_plot = sns.pairplot(train_with_target, vars=features, diag_kind="hist", plot_kws={"alpha": 0.5})
    st.pyplot(pair_plot.figure)

    st.write("Đã xong phân tích dữ liệu")

# Hàm Linear Regression với PCA

def train_linear_regression(X_train, y_train, X_test, y_test, features):
    # Hiển thị dữ liệu trước khi chuẩn hóa (5 dòng đầu tiên)
    st.write("Dữ liệu trước khi chuẩn hóa (5 dòng đầu tiên):")
    st.write(pd.DataFrame(X_train, columns=features).head())

    # Step 1: Standardization
    st.write("### Bước 1: Chuẩn hóa dữ liệu")
    st.write("Mục tiêu: Chuẩn hóa dữ liệu để các đặc trưng đóng góp đồng đều vào phân tích.")
    st.write("Công thức chuẩn hóa:")
    st.latex(r"Z = \frac{X - \mu}{\sigma}")
    st.write("Trong đó:")
    st.write("- \(Z\): Giá trị đã chuẩn hóa")
    st.write("- \(X\): Giá trị ban đầu")
    st.write("- \(\mu\): Trung bình của đặc trưng")
    st.write("- \(\sigma\): Độ lệch chuẩn của đặc trưng")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hiển thị dữ liệu sau khi chuẩn hóa (5 dòng đầu tiên)
    st.write("Dữ liệu sau khi chuẩn hóa (5 dòng đầu tiên):")
    st.write(pd.DataFrame(X_train_scaled, columns=features).head())

    # Step 2: Covariance Matrix Computation
    st.write("### Bước 2: Tính ma trận hiệp phương sai")
    st.write("Mục tiêu: Hiểu mối quan hệ giữa các đặc trưng bằng cách tính ma trận hiệp phương sai.")
    cov_matrix = np.cov(X_train_scaled.T)
    cov_matrix_df = pd.DataFrame(cov_matrix, index=features, columns=features)
    st.write("Ma trận hiệp phương sai:")
    st.write(cov_matrix_df)

    # Hiển thị heatmap của ma trận hiệp phương sai
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cov_matrix_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Heatmap của Ma trận Hiệp phương sai")
    st.pyplot(fig)

    # Step 3: Compute Eigenvectors and Eigenvalues
    st.write("### Bước 3: Tính Eigenvectors và Eigenvalues")
    st.write("Mục tiêu: Xác định các thành phần chính bằng cách tính eigenvectors và eigenvalues của ma trận hiệp phương sai.")
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sắp xếp theo thứ tự giảm dần
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    st.write("Eigenvalues (giá trị riêng):")
    st.write(eigenvalues)
    st.write("Eigenvectors (vector riêng):")
    eigenvectors_df = pd.DataFrame(eigenvectors, index=features, columns=[f"PC{i+1}" for i in range(len(features))])
    st.write(eigenvectors_df)

    # Biểu đồ phần trăm phương sai được giải thích bởi từng thành phần chính
    st.write("### Phần trăm phương sai được giải thích bởi từng thành phần chính")
    explained_variance_ratio = eigenvalues / eigenvalues.sum() * 100  # Chuyển thành phần trăm
    num_components = len(explained_variance_ratio)

    # Xác định tên đặc trưng chính cho từng thành phần
    pc_names = []
    for i in range(num_components):
        max_idx = np.argmax(np.abs(eigenvectors[:, i]))
        pc_names.append(f"PC{i+1}_{features[max_idx]}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pc_names, explained_variance_ratio, color="skyblue", alpha=0.7, label="Phương sai")
    ax.plot(pc_names, explained_variance_ratio, marker='o', color="black", linestyle='-', linewidth=1, label="Xu hướng")
    ax.set_xlabel("Thành phần chính (PC)")
    ax.set_ylabel("Phần trăm phương sai được giải thích (%)")
    ax.set_title("Phân bố phương sai của các thành phần chính")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Step 4: Feature Vector
    st.write("### Bước 4: Tạo Feature Vector")
    st.write("Mục tiêu: Chọn các thành phần chính dựa trên eigenvalues và tạo feature vector.")
    # Chọn số thành phần chính để giải thích ít nhất 95% phương sai
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= 95) + 1
    st.write(f"Số thành phần chính được chọn để giải thích ít nhất 95% phương sai: {n_components}")

    feature_vector = eigenvectors[:, :n_components]
    st.write("Feature vector (các eigenvector được chọn):")
    feature_vector_df = pd.DataFrame(feature_vector, index=features, columns=[pc_names[i] for i in range(n_components)])
    st.write(feature_vector_df)

    # Last Step: Recast the Data Along the Principal Components Axes
    st.write("### Bước cuối: Chiếu dữ liệu lên không gian các thành phần chính")
    st.write("Công thức tính tập dữ liệu sau PCA:")
    st.latex(r"\text{PCA Data} = X_{\text{scaled}} \times \text{Feature Vector}")
    st.write("Trong đó:")
    st.write("- \(X_{\text{scaled}}\): Dữ liệu đã chuẩn hóa")
    st.write("- \(\text{Feature Vector}\): Ma trận các eigenvector được chọn")

    X_train_pca = X_train_scaled @ feature_vector
    X_test_pca = X_test_scaled @ feature_vector
    X_train_pca_df = pd.DataFrame(X_train_pca, columns=[pc_names[i] for i in range(n_components)])
    X_train_pca_df["Tomorrow_Close"] = y_train.values

    # Biểu diễn dữ liệu sau khi chiếu lên PC1 và PC2 (nếu có ít nhất 2 thành phần)
    if n_components >= 2:
        st.write("### Scatter Plot: PC1 vs PC2")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter(X_train_pca_df[pc_names[0]], X_train_pca_df[pc_names[1]], alpha=0.5, c=y_train, cmap="viridis")
        plt.colorbar(scatter, ax=ax, label="Tomorrow_Close")
        ax.set_title(f"Dữ liệu sau PCA: {pc_names[0]} vs {pc_names[1]} (màu theo Tomorrow_Close)")
        ax.set_xlabel(pc_names[0])
        ax.set_ylabel(pc_names[1])
        ax.grid(True)
        st.pyplot(fig)
    elif n_components == 1:
        st.write("### Histogram của PC1")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(X_train_pca_df[pc_names[0]], bins=30, color="blue", alpha=0.7)
        ax.set_title(f"Phân bố của {pc_names[0]} sau PCA")
        ax.set_xlabel(pc_names[0])
        ax.set_ylabel("Tần suất")
        ax.grid(True)
        st.pyplot(fig)

    # Scatter plot giữa từng thành phần chính và Tomorrow_Close
    for i in range(n_components):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(X_train_pca_df[pc_names[i]], X_train_pca_df["Tomorrow_Close"], alpha=0.5, c="green")
        ax.set_title(f"{pc_names[i]} vs Tomorrow_Close")
        ax.set_xlabel(pc_names[i])
        ax.set_ylabel("Tomorrow_Close")
        ax.grid(True)
        st.pyplot(fig)

    st.write("Dữ liệu sau PCA (5 dòng đầu tiên):")
    st.write(X_train_pca_df.head())

    # Huấn luyện và dự đoán với dữ liệu đã chiếu
    lr_model = LinearRegression()
    lr_model.fit(X_train_pca, y_train)
    lr_predictions = lr_model.predict(X_test_pca)

    return lr_model, lr_predictions, scaler, feature_vector  # Trả về feature_vector thay vì None# Hàm Random Forest Regression
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    st.write(f"✅ RMSE: {rmse:.2f}")
    st.write(f"✅ MAE: {mae:.2f}")
    return model, predictions

# Hàm so sánh hiệu suất
def compare_models(lr_predictions, rf_predictions, y_test):
    lr_mse = mean_squared_error(y_test, lr_predictions)
    lr_mae = mean_absolute_error(y_test, lr_predictions)
    lr_r2 = r2_score(y_test, lr_predictions)

    rf_mse = mean_squared_error(y_test, rf_predictions)
    rf_mae = mean_absolute_error(y_test, rf_predictions)
    rf_r2 = r2_score(y_test, rf_predictions)

    st.write("### Linear Regression Metrics:")
    st.write(f"MSE: {lr_mse:.2f}")
    st.write(f"MAE: {lr_mae:.2f}")
    st.write(f"R2 Score: {lr_r2:.2f}")

    st.write("### Random Forest Metrics:")
    st.write(f"MSE: {rf_mse:.2f}")
    st.write(f"MAE: {rf_mae:.2f}")
    st.write(f"R2 Score: {rf_r2:.2f}")

    if lr_r2 > rf_r2:
        st.write("\nLinear Regression có hiệu suất tốt hơn (dựa trên R2 Score).")
    else:
        st.write("\nRandom Forest có hiệu suất tốt hơn (dựa trên R2 Score).")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_test.values, label="Actual Tomorrow_Close", color="blue")
    ax.plot(lr_predictions, label="Linear Regression Predictions", color="orange")
    ax.plot(rf_predictions, label="Random Forest Predictions", color="green")
    ax.set_title("Actual vs Predicted Values")
    ax.set_xlabel("Test Samples")
    ax.set_ylabel("Tomorrow_Close")
    ax.legend()
    st.pyplot(fig)

# Hàm dự đoán cho 3 dòng cuối
# Hàm dự đoán cho 3 dòng cuối
def predict_latest_data(latest_features, lr_model, rf_model, scaler, feature_vector):
    latest_data_scaled = scaler.transform(latest_features)
    latest_data_pca = latest_data_scaled @ feature_vector
    lr_latest_predictions = lr_model.predict(latest_data_pca)

    st.write("### Dự đoán với Linear Regression cho 3 dòng cuối:")
    for i, pred in enumerate(lr_latest_predictions):
        st.write(f"Dòng {i+1}: {pred:.2f}")

    rf_latest_predictions = rf_model.predict(latest_features)
    st.write("\n### Dự đoán với Random Forest cho 3 dòng cuối:")
    for i, pred in enumerate(rf_latest_predictions):
        st.write(f"Dòng {i+1}: {pred:.2f}")

# Hàm dự đoán cho dữ liệu giả định
# Hàm dự đoán cho dữ liệu giả định
def predict_user_input(user_input, lr_model, rf_model, scaler, feature_vector):
    user_df = pd.DataFrame([user_input])
    user_scaled = scaler.transform(user_df)
    user_pca = user_scaled @ feature_vector
    lr_prediction = lr_model.predict(user_pca)[0]

    rf_prediction = rf_model.predict(user_df)[0]

    st.write("### Dự đoán với Linear Regression cho dữ liệu giả định:", lr_prediction)
    st.write("### Dự đoán với Random Forest cho dữ liệu giả định:", rf_prediction)

# Main Streamlit app
def main():
    st.title("Phân tích và Dự đoán Giá Cổ phiếu với Linear Regression và Random Forest")

    # Tải và phân tích dữ liệu
    hist, latest_features = load_and_prepare_data()
    features = ["Open", "High", "Low", "Close", "Volume"]
    train = hist.iloc[:-100]
    test = hist.iloc[-100:]
    X_train = train[features]
    y_train = train["Tomorrow_Close"]
    X_test = test[features]
    y_test = test["Tomorrow_Close"]

    analyze_initial_data(hist)
    
    st.write("### Linear Regression ")

    # Huấn luyện và dự đoán
    lr_model, lr_predictions, scaler, pca = train_linear_regression(X_train, y_train, X_test, y_test, features)
    st.write("### Random Forest Regression")
    rf_model, rf_predictions = train_random_forest(X_train, y_train, X_test, y_test)

    # So sánh hiệu suất
    compare_models(lr_predictions, rf_predictions, y_test)

    # Dự đoán 3 dòng cuối
    predict_latest_data(latest_features, lr_model, rf_model, scaler, pca)

    # Thêm phần nhập dữ liệu từ người dùng
    st.write("### Nhập dữ liệu để dự đoán")
    with st.form("user_input_form"):
        open_value = st.number_input("Open", value=0.0, step=0.01)
        high_value = st.number_input("High", value=0.0, step=0.01)
        low_value = st.number_input("Low", value=0.0, step=0.01)
        close_value = st.number_input("Close", value=0.0, step=0.01)
        volume_value = st.number_input("Volume", value=0.0, step=1000000.0)
        submit = st.form_submit_button("Gửi để dự đoán")

    if submit:
        user_input = {
            "Open": open_value,
            "High": high_value,
            "Low": low_value,
            "Close": close_value,
            "Volume": volume_value
        }
        predict_user_input(user_input, lr_model, rf_model, scaler, pca)

if __name__ == "__main__":
    main()