import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. เตรียมข้อมูล (Data Preparation)
# ==========================================
# ข้อมูลจำลอง: กลุ่ม 0 (สีฟ้า มุมซ้ายล่าง), กลุ่ม 1 (สีแดง มุมขวาบน)
X_train = np.array([
    [1, 2], [1, 4], [2, 2], [2, 3], # กลุ่ม 0
    [8, 9], [9, 8], [9, 9], [8, 8]  # กลุ่ม 1
])
y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# ข้อมูลใหม่ 1 จุด ที่เราต้องการทำนาย
X_test = np.array([4, 4]) 
k = 4 # กำหนดจำนวนเพื่อนบ้าน

# ==========================================
# 2. กระบวนการคำนวณ KNN (ยุบเหลือ 1 บรรทัดด้วย Walrus Operator)
# ==========================================

predicted_label = np.argmax(np.bincount(nearest_labels := y_train[nearest_indices := np.argsort(distances := np.sqrt(np.sum((X_train - X_test)**2, axis=1)))[:k]]))

# ==========================================
# 3. วาดกราฟแสดงผล (Visualization)
# ==========================================
plt.figure(figsize=(8, 6))

# พล็อตจุดข้อมูล กลุ่ม 0 (สีฟ้า)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
            c='blue', s=100, label='Class 0')

# พล็อตจุดข้อมูล กลุ่ม 1 (สีแดง)
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
            c='red', s=100, label='Class 1')

# พล็อตจุดข้อมูลใหม่ (รูปดาวสีเขียว)
plt.scatter(X_test[0], X_test[1], 
            c='green', marker='*', s=300, 
            label=f'New Point (Predicted: {predicted_label})')

# วาดเส้นประเชื่อมไปยังเพื่อนบ้านที่ใกล้ที่สุด k ตัว
for i in nearest_indices:
    plt.plot([X_test[0], X_train[i, 0]], [X_test[1], X_train[i, 1]], 'k--', alpha=0.5)

# วาดวงกลมรัศมีครอบคลุมเพื่อนบ้าน k ตัว (เพื่อให้เห็นขอบเขตการพิจารณา)
max_dist = distances[nearest_indices[-1]] # ระยะทางของเพื่อนบ้านตัวที่ไกลที่สุดใน k ตัว
circle = plt.Circle((X_test[0], X_test[1]), max_dist, color='green', fill=False, linestyle=':')
plt.gca().add_patch(circle)

# ตกแต่งกราฟ
plt.title(f'KNN Visualization (K={k})')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axis('equal') # ทำให้สัดส่วนแกน X และ Y เท่ากัน วงกลมจะได้ไม่เบี้ยว

plt.show()