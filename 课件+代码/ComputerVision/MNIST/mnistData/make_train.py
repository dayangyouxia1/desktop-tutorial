import os
import shutil
import csv

# 定义源文件夹和目标文件夹路径
source_folder = 'mnist_train'
target_folder = 'train_data'

# 确保目标文件夹存在，不存在则创建
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 定义CSV文件路径
csv_filename = 'train_data/label.csv'

# 打开CSV文件并写入表头
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])

    # 遍历0-9文件夹并移动图片
    for label in range(10):
        label_folder = os.path.join(source_folder, str(label))
        if os.path.exists(label_folder):
            for image_name in os.listdir(label_folder):
                image_id = os.path.splitext(image_name)[0]
                source_image_path = os.path.join(label_folder, image_name)
                target_image_path = os.path.join(target_folder, image_name)

                # 移动图片到目标文件夹
                shutil.move(source_image_path, target_image_path)

                # 写入CSV文件
                writer.writerow([image_id, label])

print(f"所有图片已移动到 {target_folder} 文件夹，标签已记录在 {csv_filename} 文件中。")
