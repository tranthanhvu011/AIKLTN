# import json
# import mysql.connector

#
#
# with open('../newFile.json', 'r', encoding='utf-8') as f:
#     products = [json.loads(line) for line in f if line.strip()]
#
#
# # Connect to MySQL database
# cnx = mysql.connector.connect(
#     host='localhost',
#     port=3306,
#     user='root',
#     password='123456789',
#     database='productservice'
# )
#
# cursor = cnx.cursor()
#
# # Insert each product into the database
# for product in products:
#     asin = product['asin']
#     product_id = asin  # Assuming product_id is the same as asin
#     product_title = product.get('title', None)
#     product_price = product.get('price', None)
#     product_thumbnail = product['imUrl']
#     brand_name = product.get('brand', None)
#     product_status = 1
#     sales_rank_category = list(product['salesRank'].keys())[0] if 'salesRank' in product and product[
#         'salesRank'] else None
#
#     cursor.execute("""
#         INSERT INTO products (asin, product_title, product_price, product_thumbnail, brand_name, product_status, created_at, updated_at, salesRank)
#         VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)
#     """, (
#      asin, product_title, product_price, product_thumbnail, brand_name, product_status, sales_rank_category))

# Commit and close
# cnx.commit()
# cursor.close()
# cnx.close()
import json
import mysql.connector

# def extract_product_type(categories_str: str) -> str:
#     text = categories_str.lower()

#     if any(kw in text for kw in ['shirt', 't-shirt', 'top', 'blouse', 'tee']):
#         return 'shirt'
#     elif any(kw in text for kw in ['pants', 'jeans', 'trousers', 'leggings', 'shorts']):
#         return 'pants'
#     elif any(kw in text for kw in ['dress', 'gown', 'skirt']):
#         return 'dress'
#     elif any(kw in text for kw in ['shoes', 'sneakers', 'boots', 'sandals', 'footwear']):
#         return 'shoes'
#     elif any(kw in text for kw in ['jacket', 'coat', 'vest', 'blazer', 'parka']):
#         return 'outerwear'
#     elif any(kw in text for kw in ['socks', 'hat', 'scarf', 'belt', 'gloves', 'earrings']):
#         return 'accessory'
#     else:
#         return 'unknown'


# Đọc file JSON (mỗi dòng là 1 object)
with open('../newFile.json', 'r', encoding='utf-8') as f:
    products = [json.loads(line) for line in f if line.strip()]

# Kết nối MySQL
cnx = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456789',
    database='productservice'
)
cursor = cnx.cursor()
for product in products:
    asin = product.get('asin')
    if not asin or not isinstance(asin, str):
        continue

    asin_clean = asin.strip()
    if not asin_clean.isalnum():
        continue

    # Lấy description
    description = product.get('description')
    if isinstance(description, list):
        description = " ".join(description)
    elif not isinstance(description, str):
        description = ""

    # Lấy categories JSON
    raw_categories = product.get('categories', [])
    if not isinstance(raw_categories, list):
        raw_categories = []
    
    # Chuyển sang JSON string để insert vào MySQL JSON column
    category_json_str = json.dumps(raw_categories, ensure_ascii=False)

    # Ghép toàn bộ categories thành text để phân loại
    # all_category_text = " ".join(" > ".join(cat) for cat in raw_categories if isinstance(cat, list))
    # product_type = extract_product_type(all_category_text)

    # Insert vào bảng categories
    # cursor.execute("""
    #     INSERT INTO categories (product_asin, categories, description, product_type)
    #     VALUES (%s, %s, %s, %s)
    # """, (
    #     asin_clean,
    #     category_json_str,
    #     description,
    #     product_type
    # ))
    cursor.execute("""
        INSERT INTO categories (product_asin, categories, description)
        VALUES (%s, %s, %s)
    """, (
        asin_clean,
        category_json_str,
        description,
    ))


# Commit và đóng kết nối
cnx.commit()
cursor.close()
cnx.close()


# import json
# import mysql.connector
#
# # Đọc file JSON (mỗi dòng là 1 JSON object)
# with open('../newFile.json', 'r', encoding='utf-8') as f:
#     products = [json.loads(line) for line in f if line.strip()]
#
# # Kết nối tới MySQL
# cnx = mysql.connector.connect(
#     host='localhost',
#     port=3306,
#     user='root',
#     password='123456789',
#     database='productservice'
# )
#
# cursor = cnx.cursor()
#
# # Duyệt từng sản phẩm và insert vào DB
# for product in products:
#     asin = product.get('asin')
#
#     # Bỏ qua nếu asin rỗng hoặc không hợp lệ
#     if not asin or not isinstance(asin, str):
#         continue
#
#     asin_clean = asin.strip()
#     if not asin_clean.isalnum():  # đảm bảo là chuỗi hợp lệ
#         continue
#
#     image_data = f"{asin_clean}.jpg"
#
#     # Đảm bảo image_data kết thúc bằng .jpg và không chứa ký tự lạ
#     if not image_data.lower().endswith(".jpg") or len(image_data) < 8:
#         continue
#
#     try:
#         cursor.execute("""
#             INSERT INTO product_images (product_asin, image_data)
#             VALUES (%s, %s)
#         """, (asin_clean, image_data))
#     except mysql.connector.Error as e:
#         print(f"Lỗi khi insert {asin_clean}: {e}")
#
# # Ghi vào database
# cnx.commit()
# cursor.close()
# cnx.close()

