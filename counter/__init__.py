import numpy as np


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


text = 'You say goodbye and i say hello.'
word_counter = WordCounter(text)
id_to_word = word_counter.convert_id_to_word()
word_to_id = word_counter.convert_word_to_id()
words = word_counter.get_words()

# 将单词列表转换为id
word_id_list = [word_to_id[w] for w in words]

print(create_co_matrix(word_id_list, len(word_to_id), 1))
