import json
import mysql.connector

# === Cấu hình kết nối MySQL ===
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456789',        # Thay bằng mật khẩu của bạn nếu có
    'database': 'recommendservice',
    'charset': 'utf8mb4'
}

# === Đường dẫn file JSON ===
JSON_FILE_PATH = '../dl_output_full_30/best_batch.json'

# === Kết nối DB ===
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# === Load dữ liệu JSON ===
with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# === Duyệt từng entry và insert/update ===
for entry in data:
    try:
        asin = entry.get('asin')
        recommend_asins = json.dumps(entry.get('recommend_asins', []), ensure_ascii=False)
        avg_sim = float(entry.get('avg_similarity', 0))
        p10 = float(entry.get('precision@10', 0))
        p20 = float(entry.get('precision@20', 0))
        p30 = float(entry.get('precision@30', 0))

        sql = """
            INSERT INTO asin_recommendations (
                asin, recommend_asins, avg_similarity, precision_10, precision_20, precision_30
            ) VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                recommend_asins = VALUES(recommend_asins),
                avg_similarity = VALUES(avg_similarity),
                precision_10 = VALUES(precision_10),
                precision_20 = VALUES(precision_20),
                precision_30 = VALUES(precision_30)
        """

        cursor.execute(sql, (asin, recommend_asins, avg_sim, p10, p20, p30))

    except Exception as e:
        print(f"[ERROR] Failed to insert asin {asin}: {e}")

# === Kết thúc ===
conn.commit()
cursor.close()
conn.close()

print("✅ Import completed.")
