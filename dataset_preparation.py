def build_corpus(data_dir):
    word_lists = []
    tag_lists = []
    with open(data_dir, 'r') as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                index, word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
        return word_lists, tag_lists


def build_map_words(lists1, lists2, list3):
    lists = lists1 + lists2 + list3
    maps = {"<PAD>": 0, "<UNK>": 1}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def build_map_tags(lists1, lists2):
    lists = lists1 + lists2
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps


def build_corpus_test(data_dir):
    word_lists = []

    with open(data_dir, 'r') as f:
        word_list = []

        for line in f:
            if line != '\n':
                index, word = line.strip('\n').split()
                word_list.append(word)

            else:
                word_lists.append(word_list)
                word_list = []
        return word_lists


def sentence_vectorizer(train_x, word2id):
    word_vectors = list()
    tmp_x = list()
    for words in train_x:
        for word in words:
            tmp_x.append(word2id[word])
        word_vectors.append(tmp_x)
        tmp_x = list()

    return word_vectors


def tag_vectorizer(train_y, tag2id):
    train_y_vec = list()
    for tags in train_y:
        tmp_yy = list()
        for label in tags:
            tmp_yy.append(tag2id[label])
        train_y_vec.append(tmp_yy)
    return train_y_vec
