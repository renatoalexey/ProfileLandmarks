import re
from src.utils import core

tool_list = ["amazon", "face_alignment", "mlkit"]
threshold_list = [70, 90, 100]

def get_bb_accuracy(line):
    match = re.search(r"l_accuracy:[^,]*", line)
    return match.group().split(":")[1].strip()

def run():
    for tool_name in tool_list:
        file_path = f"result/{tool_name}/landmarks.txt"
        bb_accuracy_list = []
        print(f"{tool_name}:")
        for line in core.get_file_lines(file_path):
            bb_accuracy_list.append(get_bb_accuracy(line))
        for threshold in threshold_list:
           above_t = len(list(filter(lambda n: float(n) >= threshold, bb_accuracy_list)) )
           print(f"Threshold: {threshold} result: {above_t*100/len(bb_accuracy_list)}")

run()