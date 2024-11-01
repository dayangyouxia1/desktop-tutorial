import os
import shutil
import pandas as pd

# 定义文件夹路径
mnist_train_folder = 'mnistData\mnist_test'
student_data_folder = 'student_data'

# 初始化NUM_train
NUM_train = 0 

# 遍历student_data中的每个子文件夹
for student_folder in os.listdir(student_data_folder):
    student_folder_path = os.path.join(student_data_folder, student_folder)

    # 检查是否是文件夹
    if os.path.isdir(student_folder_path):
        csv_file_path = os.path.join(student_folder_path, 'label.csv')

        # 检查label.csv文件是否存在
        if os.path.exists(csv_file_path):
            # 读取CSV文件的后40行
            df = pd.read_csv(csv_file_path)
            df = df.tail(10)

            for _, row in df.iterrows():
                image_id = row['id']
                label = row['label']

                # 定义源图片路径和目标文件夹路径
                source_image_path = os.path.join(student_folder_path, f'{image_id}.png')
                target_folder_path = os.path.join(mnist_train_folder, str(label))

                # 确保目标文件夹存在
                if not os.path.exists(target_folder_path):
                    os.makedirs(target_folder_path)

                # 定义目标图片路径
                target_image_path = os.path.join(target_folder_path, f'{NUM_train}.png')

                # 复制图片到目标文件夹
                if os.path.exists(source_image_path):
                    shutil.copy(source_image_path, target_image_path)
                    print(f"已复制 {source_image_path} 到 {target_image_path}")

                    # 更新NUM_train
                    NUM_train += 1
                else:
                    print(f"源图片不存在: {source_image_path}")

print("所有图片处理完成。")

"""
mnistData下的训练文件夹mnist_test里面有10个名为0-9的文件夹分别放0-9类的图片。
图片名为id名0-59999.png，NUM_train=60000。
学生训练文件夹student_data下有若干个名为如2024111111的文件夹，里面有一个label.csv文件，读取前60条{id, label}，
把id.png图片复制到mnist_train下对应的label文件夹下，命名为NUM_train.png，NUM_train++。
重复处理完student_data里的所有文件
"""
