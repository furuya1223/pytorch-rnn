import numpy as np


class Util:
    num_characters = 8
    characters = ['B', 'E', 'P', 'S', 'T', 'V', 'X', '@']

    @staticmethod
    def get_index(character):
        return Util.characters.index(character)

    @staticmethod
    def get_one_hot_vector(character):
        vector = np.zeros(Util.num_characters)
        vector[Util.get_index(character)] = 1
        return vector
