from enum import Enum

class CorrespondentFaceAlignment(Enum):

    CFP = ({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 9, 9: 17, 10: 19, 11: 22, 12: 36, 14: 37, 16: 39, 17: 47, 18: 27, 19: 28, 20: 29, 21: 30, 24: 31, 27: 52, 25: 48, 28: 12})

    def __init__(self, points):
        self.points = points