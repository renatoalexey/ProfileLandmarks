from enum import Enum

class CorrespondentMLKit(Enum):

    CFP = ({2: 29, 3: 28, 9: 20, 10: 42, 11: 44, 12: 46, 13: 72, 14: 61, 15: 63, 16: 65, 17: 70, 23: 130, 24: 129, 26: 100, 27: 93, 28: 107, 30: 119, 29: 123})

    def __init__(self, points):
        self.points = points