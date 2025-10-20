from enum import Enum

class CorrespondentFaceAlignment(Enum):

    CFP = ({1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 10, 10: 18, 11: 20, 12: 23, 13: 37, 15: 38, 17: 40, 18: 48, 19: 28, 20: 29, 21: 30, 22: 31, 25: 32, 28: 53, 26: 49, 29: 13})

    def __init__(self, points):
        self.points = points