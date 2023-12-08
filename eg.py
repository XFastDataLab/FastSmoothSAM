import os


def delete_extra_images(folder_A, folder_B):
    files_A = set(os.listdir(folder_A))  # 获取A文件夹中的所有文件名
    files_B = set(os.listdir(folder_B))  # 获取B文件夹中的所有文件名

    # 遍历B文件夹中的文件，删除在A文件夹中的图片
    for file in files_B:
        if file not in files_A:
            file_path = os.path.join(folder_B, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


def delete_extra_txt(folder_A, folder_B):
    files_A = set(os.listdir(folder_A))  # 获取A文件夹中的所有文件名
    files_B = set(os.listdir(folder_B))  # 获取B文件夹中的所有文件名

    # 遍历B文件夹中的文件，删除在A文件夹中的图片对应的txt
    for file in files_B:
        if file.split('.')[0] + ".jpg" in files_A:
            file_path = os.path.join(folder_B, file)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


if __name__ == '__main__':
    # 指定A和B文件夹的路径
    folder_A = 'E:/projects/sam1w/1'
    folder_B = 'E:/projects/sam1w/2'

    delete_extra_images(folder_A, folder_B)
    # delete_extra_txt(folder_A, folder_B)

