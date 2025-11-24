from enum import Enum

class CorrespondentAmazon(Enum):

    CFP = ({4: 26, 7: 27, 9: 5, 10: 7, 11: 10, 12: 11, 13: 13, 14: 9, 15: 15, 16: 14, 17: 18, 21: 4, 24: 19, 25: 3, 26: 21})
    #CORRESPONDENT = ({1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

    def __init__(self, points):
        self.points = points