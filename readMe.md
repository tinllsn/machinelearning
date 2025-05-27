# Stock Market Prediction using Random Forest

Ứng dụng dự đoán thị trường chứng khoán sử dụng mô hình Random Forest và các chỉ báo kỹ thuật.

## Tính năng chính

- Phân tích dữ liệu S&P 500 từ năm 2022
- Tính toán các chỉ báo kỹ thuật (MA10, MA50, MA200, Bollinger Bands)
- Dự đoán xu hướng thị trường sử dụng Random Forest
- Hiển thị biểu đồ tương tác với Plotly
- Phân tích khối lượng giao dịch

## Yêu cầu hệ thống

- Python 3.7 trở lên
- Các thư viện Python cần thiết:
  - streamlit
  - yfinance
  - pandas
  - scikit-learn
  - matplotlib
  - plotly

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install streamlit yfinance pandas scikit-learn matplotlib plotly
```

2. Nếu gặp lỗi khi cài đặt, hãy nâng cấp pip:
```bash
python -m pip install --upgrade pip
```

## Cách chạy ứng dụng

1. Mở Command Prompt (CMD) hoặc Terminal
2. Di chuyển đến thư mục chứa file `appRdF.py`
3. Chạy lệnh:
```bash
streamlit run appRdF.py
```

Ứng dụng sẽ tự động mở trong trình duyệt web của bạn tại địa chỉ http://localhost:8501

## Lưu ý

- Đảm bảo có kết nối internet để tải dữ liệu thị trường
- Dữ liệu được cập nhật tự động mỗi khi chạy ứng dụng
- Mô hình sử dụng ngưỡng dự đoán 0.6 cho độ chính xác cao hơn