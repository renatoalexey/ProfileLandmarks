import re
import matplotlib.pyplot as plt
from src.utils import core

tool_list = {"amazon": "Amazon", "face_alignment": "Face Alignment", "mlkit": "ML Kit"}
threshold_list = [70, 90, 100]

def print_graph(means, graph_name, positions=None, x_label=None):
    #plt.figure(figsize=(12, 7))
    plt.boxplot(means, positions, tick_labels=positions, whis=2.5)
    plt.xlabel(x_label)
    plt.ylabel(r'Percentual de $\it{landmarks}$ contidos no' '\n' r'interior das $\it{bounding boxes}$')
    plt.savefig(f'{graph_name}.png', bbox_inches='tight')

def get_bb_accuracy(line):
    match = re.search(r"l_accuracy:[^,]*", line)
    return match.group().split(":")[1].strip()

def run():
    all_bounding_boxes = []
    label_list = []
    for tool_name in tool_list.keys():
        file_path = f"result/{tool_name}/landmarks.txt"
        bb_accuracy_list = []
        label_list.append(tool_list[tool_name])
        print(f"{tool_name}:")
        for line in core.get_file_lines(file_path):
            bb_accuracy_list.append(float(get_bb_accuracy(line)))
        for threshold in threshold_list:
           above_t = len(list(filter(lambda n: float(n) >= threshold, bb_accuracy_list)) )
           print(f"Threshold: {threshold} result: {above_t*100/len(bb_accuracy_list)}")
        all_bounding_boxes.append(bb_accuracy_list)
    print_graph(all_bounding_boxes, "accuracy_landmarks", label_list, "Ferramentas")

def get_worst_bbs(tool_name):
    file_path = f"result/{tool_name}/landmarks.txt"
    bb_accuracy_list = []
     
    for i, line in enumerate(core.get_file_lines(file_path), start=1):
        bb_accuracy_list.append((i, float(get_bb_accuracy(line))))

    top = sorted(bb_accuracy_list, key=lambda x: x[1])[:10]
    print(top)

#get_worst_bbs('face_alignment')
run()