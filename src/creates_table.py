import re
import matplotlib.pyplot as plt
from src.utils import core

tool_list = ["amazon", "face_alignment", "mlkit"]
threshold_list = [70, 90, 100]

def print_graph(means, graph_name, positions=None, x_label=None):
    plt.figure(figsize=(12, 7))
    plt.boxplot(means, positions, showmeans=True, meanline=True, tick_labels=positions, whis=2.5)
    plt.xlabel(x_label)
    plt.ylabel('Accuracy(%)')
    plt.savefig(f'{graph_name}.png')
def get_bb_accuracy(line):
    match = re.search(r"l_accuracy:[^,]*", line)
    return match.group().split(":")[1].strip()

def run():
    all_bounding_boxes = []
    for tool_name in tool_list:
        file_path = f"result/{tool_name}/landmarks.txt"
        bb_accuracy_list = []
        print(f"{tool_name}:")
        for line in core.get_file_lines(file_path):
            bb_accuracy_list.append(float(get_bb_accuracy(line)))
        for threshold in threshold_list:
           above_t = len(list(filter(lambda n: float(n) >= threshold, bb_accuracy_list)) )
           print(f"Threshold: {threshold} result: {above_t*100/len(bb_accuracy_list)}")
        all_bounding_boxes.append(bb_accuracy_list)
    print_graph(all_bounding_boxes, "accuracy_landmarks", tool_list, "Bibliotecas")

run()