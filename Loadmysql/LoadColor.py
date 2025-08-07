import mysql.connector
import random
import json

# Kết nối MySQL
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='123456789',   # đổi user/pass cho phù hợp
    database='productservice'  # đổi DB nếu khác
)
cursor = conn.cursor(dictionary=True)

# Lấy tất cả product_id có trong bảng products
cursor.execute("SELECT product_id FROM products")
product_ids = [row['product_id'] for row in cursor.fetchall()]

# Đọc sẵn tất cả màu trong bảng Color vào dict, tránh query lặp
cursor.execute("SELECT color_id, name_color, code_color FROM color WHERE color_id BETWEEN 1 AND 50")
color_data = {str(row['color_id']): row for row in cursor.fetchall()}

for idx, product_id in enumerate(product_ids):
    num_colors = random.randint(3, 5)
    selected_colors = random.sample(list(color_data.keys()), num_colors)  # random các color_id không trùng nhau
    colors = []
    for color_id in selected_colors:
        color_info = color_data.get(color_id)
        if color_info:
            colors.append({
                "color_id": color_id,
                "name_color": color_info['name_color'],
                "code_color": color_info['code_color'],
                "status": "0"
            })
    json_value = json.dumps(colors, ensure_ascii=False)

    # Update color_asin cho sản phẩm hiện tại
    sql = "UPDATE products SET color_asin = %s WHERE product_id = %s"
    cursor.execute(sql, (json_value, product_id))

    if (idx + 1) % 1000 == 0:
        print(f'Đã update {idx+1} sản phẩm...')
        conn.commit()  # commit định kỳ

conn.commit()  # commit cuối
cursor.close()
conn.close()
print(f'Hoàn tất update cho {len(product_ids)} sản phẩm!')
