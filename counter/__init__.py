import numpy as np

from common.utils import ppmi
from counter.similarity import cos
import matplotlib.pyplot as plt

class WordCounter:

    def __init__(self, text):
        self.text = text

    def get_words(self):
        return self.text.lower() \
            .replace('.', ' .') \
            .split(" ")

    def convert_id_to_word(self):
        return {id: word for word, id in self.convert_word_to_id().items()}

    def convert_word_to_id(self):
        words = list(set(self.get_words()))
        return {words[i]: i for i in range(len(words))}


def create_co_matrix(word_ids, size: int, window_size=1):
    '''
    创建共现矩阵
    :param word_ids: id单词列表
    :param size: 矩阵的行列大小
    :param window_size: 滑动窗口的大小
    :return:
    '''
    co_matrix = np.zeros((size, size))

    for idx in range(len(word_ids)):

        for window_s in range(1, window_size + 1):
            # 根据当前位置，左右滑动
            left_idx = idx - window_s
            right_idx = idx + window_s

            # 把index转换成相应的word_id，更新共现矩阵
            word_id = word_ids[idx]
            if left_idx >= 0:
                left_word_id = word_ids[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < len(word_ids):
                right_word_id = word_ids[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def top(C, word, word_to_id, id_to_word):
    word = word.lower()

    if word not in word_to_id:
        print(f"error word {word}")
        return

    current_id = word_to_id[word]
    sim = np.zeros(len(C))

    for i in range(len(C)):
        if i == current_id:
            continue
        sim[i] = cos(C[current_id], C[i])

    for i in (-1 * sim).argsort():
        print(f'similarity {sim[i]}, word {id_to_word[i]}')


if __name__ == '__main__':
    text = 'You say goodbye and i say hello.'
    word_counter = WordCounter(text)
    id_to_word = word_counter.convert_id_to_word()
    word_to_id = word_counter.convert_word_to_id()
    words = word_counter.get_words()

    # 将单词列表转换为id
    word_id_list = [word_to_id[w] for w in words]

    C = create_co_matrix(word_id_list, len(word_to_id), 1)

    top(C, 'you', word_to_id, id_to_word)

    W = ppmi(C)
    U, S, V = np.linalg.svd(W)

    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

    plt.scatter(U[:,0], U[:,1])
    plt.show()