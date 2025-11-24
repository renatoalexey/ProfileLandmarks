from enum import Enum

class CorrespondentMLKit(Enum):

    CFP = ({1: 28, 2: 27, 8: 19, 9: 41, 10: 43, 11: 45, 12: 71, 13: 60, 14: 62, 15: 64, 16: 69, 22: 129, 23: 128, 25: 99, 26: 92, 27: 106, 29: 118, 28: 122})

    def __init__(self, points):
        self.points = points