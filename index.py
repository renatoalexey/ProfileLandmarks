import os
from src.amazon import detect_distances

def run():
    detect_distances.run()

if os.path.exists('output/cfp_amazon_result.txt'):
    os.remove('output/cfp_amazon_result.txt')  

run()