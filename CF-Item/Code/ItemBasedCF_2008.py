import json
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import train_test_split, GridSearchCV
from surprise.accuracy import rmse, mae
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import gc


# 1. Đọc và tiền xử lý dữ liệu
def load_and_preprocess_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            review = json.loads(line)
            data.append({
                'user_id': review['reviewerID'],
                'parent_asin': review['asin'],
                'rating': review['overall']
            })

    df = pd.DataFrame(data)
    df = df.groupby(['user_id', 'parent_asin'])['rating'].mean().reset_index()

    min_ratings = 15  # Tăng lên 10 để giảm số lượng item
    user_counts = df['user_id'].value_counts()
    item_counts = df['parent_asin'].value_counts()
    df_filtered = df[df['user_id'].isin(user_counts[user_counts >= min_ratings].index)]
    df_filtered = df_filtered[df_filtered['parent_asin'].isin(item_counts[item_counts >= min_ratings].index)]

    print(f"Số người dùng sau khi lọc: {df_filtered['user_id'].nunique()}")
    print(f"Số sản phẩm sau khi lọc: {df_filtered['parent_asin'].nunique()}")
    print(f"Số đánh giá sau khi lọc: {len(df_filtered)}")

    return df_filtered


# 2. Tìm tham số tối ưu bằng GridSearchCV
def find_best_params(data):
    param_grid = {
        'k': [20, 30, 50],
        'min_k': [5],
        'sim_options': {
            'name': ['msd'],  # Chỉ dùng 'msd' để tránh lỗi bộ nhớ
            'user_based': [False]  # Chuyển sang Item-Based CF
        }
    }
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=['rmse', 'mae'], cv=3, n_jobs=1)
    gs.fit(data)

    best_params = gs.best_params['rmse']
    print(f"Tham số tối ưu (RMSE): {best_params}")

    gc.collect()
    return best_params


# 3. Huấn luyện và đánh giá mô hình
def evaluate_model(data, best_params, n_runs=5):
    rmse_list, mae_list = [], []
    precision_list, recall_list, f1_list = [], [], []
    coverage_list = []

    for run in range(1, n_runs + 1):
        print(f"\n--- Chạy lần {run} ---")

        trainset, testset = train_test_split(data, test_size=0.2, random_state=run)

        model = KNNWithMeans(
            k=best_params['k'],
            min_k=best_params['min_k'],
            sim_options=best_params['sim_options'],
            verbose=True
        )

        start_time = time.time()
        model.fit(trainset)
        print(f"Thời gian huấn luyện: {time.time() - start_time:.2f} giây")

        start_time = time.time()
        predictions = model.test(testset)
        print(f"Thời gian dự đoán: {time.time() - start_time:.2f} giây")

        rmse_value = rmse(predictions, verbose=False)
        mae_value = mae(predictions, verbose=False)

        threshold = 4.0
        true_labels = [1 if pred.r_ui >= threshold else 0 for pred in predictions]
        pred_labels = [1 if pred.est >= threshold else 0 for pred in predictions]

        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)

        recommended_items = set(pred.iid for pred in predictions if pred.est >= threshold)
        total_items = len(set(pred.iid for pred in predictions))
        coverage = len(recommended_items) / total_items if total_items > 0 else 0

        rmse_list.append(rmse_value)
        mae_list.append(mae_value)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        coverage_list.append(coverage)

        print(f"RMSE: {rmse_value:.4f} | MAE: {mae_value:.4f}")
        print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
        print(f"Coverage: {coverage:.4f}")

        top_n = get_top_n(predictions, n=10)
        recommendations = [{'user_id': uid, 'recommended_asins': [iid for iid, _ in ratings]}
                           for uid, ratings in top_n.items()]
        pd.DataFrame(recommendations).to_csv(f'recommendations_run{run}.csv', index=False)
        print(f"Đã lưu gợi ý vào recommendations_run{run}.csv")

        del model, predictions, top_n, recommendations
        gc.collect()

    return rmse_list, mae_list, precision_list, recall_list, f1_list, coverage_list


# 4. Hàm lấy top N gợi ý
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# 5. Trực quan hóa kết quả
def visualize_results(runs, rmse_list, mae_list, precision_list, recall_list, f1_list, coverage_list):
    plt.figure(figsize=(10, 6))
    plt.plot(runs, rmse_list, marker='o', label='RMSE', color='blue')
    plt.plot(runs, mae_list, marker='o', label='MAE', color='orange')
    plt.xlabel('Lần chạy')
    plt.ylabel('Giá trị')
    plt.title('RMSE và MAE qua các lần chạy')
    plt.legend()
    plt.grid(True)
    plt.savefig('rmse_mae_plot.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(runs, precision_list, marker='o', label='Precision', color='green')
    plt.plot(runs, recall_list, marker='o', label='Recall', color='red')
    plt.plot(runs, f1_list, marker='o', label='F1-score', color='purple')
    plt.plot(runs, coverage_list, marker='o', label='Coverage', color='cyan')
    plt.xlabel('Lần chạy')
    plt.ylabel('Giá trị')
    plt.title('Precision, Recall, F1-score và Coverage qua các lần chạy')
    plt.legend()
    plt.grid(True)
    plt.savefig('metrics_plot.png')
    plt.close()

    metrics = ['RMSE', 'MAE', 'Precision', 'Recall', 'F1-score', 'Coverage']
    avg_values = [np.mean(rmse_list), np.mean(mae_list), np.mean(precision_list),
                  np.mean(recall_list), np.mean(f1_list), np.mean(coverage_list)]
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, avg_values, color=['blue', 'orange', 'green', 'red', 'purple', 'cyan'])
    plt.xlabel('Chỉ số')
    plt.ylabel('Giá trị trung bình')
    plt.title('Trung bình các chỉ số sau các lần chạy')
    plt.savefig('avg_metrics_bar_itemBased.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')

    table_data = [
        ['Run'] + [f'Run {i}' for i in runs] + ['Average'],
        ['RMSE'] + [f'{x:.4f}' for x in rmse_list] + [f'{np.mean(rmse_list):.4f}'],
        ['MAE'] + [f'{x:.4f}' for x in mae_list] + [f'{np.mean(mae_list):.4f}'],
        ['Precision'] + [f'{x:.4f}' for x in precision_list] + [f'{np.mean(precision_list):.4f}'],
        ['Recall'] + [f'{x:.4f}' for x in recall_list] + [f'{np.mean(recall_list):.4f}'],
        ['F1-score'] + [f'{x:.4f}' for x in f1_list] + [f'{np.mean(f1_list):.4f}'],
        ['Coverage'] + [f'{x:.4f}' for x in coverage_list] + [f'{np.mean(coverage_list):.4f}']
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    plt.savefig('results_table.png', bbox_inches='tight')
    plt.close()


# 6. Hàm chính
def main():
    file_path = r"C:\Users\lethi\Downloads\reviews_Clothing_Shoes_and_Jewelry.json\reviews_Clothing_Shoes_and_Jewelry.json"

    df_filtered = load_and_preprocess_data(file_path)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df_filtered[['user_id', 'parent_asin', 'rating']], reader)

    best_params = find_best_params(data)

    n_runs = 5
    results = evaluate_model(data, best_params, n_runs)
    rmse_list, mae_list, precision_list, recall_list, f1_list, coverage_list = results

    print("\n--- Kết quả trung bình ---")
    for metric, values in zip(
            ['RMSE', 'MAE', 'Precision', 'Recall', 'F1-score', 'Coverage'],
            [rmse_list, mae_list, precision_list, recall_list, f1_list, coverage_list]
    ):
        print(f"{metric} trung bình: {np.mean(values):.4f}")

    visualize_results(range(1, n_runs + 1), *results)


if __name__ == "__main__":
    main()