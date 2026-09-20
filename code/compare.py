import os

def read_metric(file_path):
    # 读取文件，过滤空行，只保留带冒号的有效行
    valid_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()
        for line in raw_lines:
            line_strip = line.strip()
            if line_strip and ":" in line_strip:
                valid_lines.append(line_strip)
    # 取前两行指标
    acc_str = valid_lines[0].split(":")[1].strip().replace("%", "")
    f1_str = valid_lines[1].split(":")[1].strip()
    acc = float(acc_str) / 100
    f1 = float(f1_str)
    return acc, f1

def main():
    print("="*50)
    print("方法对比")
    print("="*50)

    hog_file = r"D:\emotion_exp\exp_result\hog_svm_results.txt"
    if os.path.exists(hog_file):
        hog_acc, hog_f1 = read_metric(hog_file)
    else:
        print("请先运行 HOGSVM.py 生成 hog_svm_results.txt")
        return

    deep_file = r"D:\emotion_exp\exp_result\deep_results.txt"
    if os.path.exists(deep_file):
        try:
            deep_acc, deep_f1 = read_metric(deep_file)
        except Exception as e:
            print(f"读取deep_results.txt失败：{e}")
            deep_acc = float(input("手动输入深度学习准确率(不含%): ")) / 100
            deep_f1 = float(input("手动输入深度学习F1-score: "))
    else:
        deep_acc = float(input("深度学习准确率(%): ")) / 100
        deep_f1 = float(input("深度学习F1-score: "))

    print(f"\n{'方法':<15s} {'准确率':<10s} {'F1-score':<10s}")
    print("-"*35)
    print(f"{'HOG+SVM':<15s} {hog_acc*100:<9.2f}% {hog_f1:<10.4f}")
    print(f"{'深度学习':<15s} {deep_acc*100:<9.2f}% {deep_f1:<10.4f}")
    print("-"*35)
    print(f"{'指标提升':<15s} {(deep_acc-hog_acc)*100:<9.2f}% {(deep_f1-hog_f1):<10.4f}")

if __name__ == "__main__":
    main()