import pickle
# import src.utils.build_vocab
from src.utils.build_vocab import Vocabulary
from src.utils.load_save import load_file, save_file
import numpy as np
import pandas as pd
import argparse
from collections import Counter
import nltk
import torch
from src.utils.basic_utils import load_json
from src.utils.metrics import get_wups
import os


def main():
    with open('/data/fukunhao/NExT-QA/nextqa_oe/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    # label2ans = {}
    # ans2label = {}

    # for i in range(4000):
    #     label2ans[i] = vocab.idx2word[i]
    #     ans2label[label2ans[i]] = i

    # print(len(vocab.word2idx))
    print(vocab.word2idx)
    # a = [1, 2, 3]
    # b = a * 2
    # print(b)


def generate_multi_vocab(path):
    data = pd.read_csv(path)
    labels = data['answer']
    label_list = np.array(labels.values.tolist())
    phrase_list = []

    for index, label in enumerate(label_list):
        label_replace = label.replace(' ', '_')
        phrase_list.append(label_replace.replace('_’_', "'"))

    vocab = build_vocab(phrase_list)
    return vocab


def build_vocab(annos):
    """Build a simple vocabulary wrapper."""

    counter = Counter()

    for ans in annos:
        # qns, ans = vqa['question'], vqa['answer']
        # text = qns # qns +' ' +ans
        text = str(ans)
        tokens = nltk.tokenize.word_tokenize(text)
        # print(tokens)
        counter.update(tokens)

    counter = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    save_file(dict(counter), 'dataset/VideoQA/multi_word_count.json')
    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [item[0] for item in counter]

    print(len(words))
    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    return vocab


def view_pth(path, show_value=False):
    model = torch.load(path, torch.device('cuda:3'))
    print('type:', type(model))
    print('length:', len(model))
    for item in model['model'].keys():
        print(item)


def process_json(path, output_path, ref_file_add):
    data = load_json(path)
    multi_ref_ans = False
    if os.path.exists(ref_file_add):
        add_ref = load_file(ref_file_add)
        multi_ref_ans = True
    else:
        add_ref = {}

    preds = []
    questions = []
    answers = []
    q_type = []
    wups_0 = []
    wups_9 = []
    # w_0 = 0
    # w_9 = 0
    for item in data:
        question = item['data']['question']
        pred = item['answer']
        answer = item['data']['answer']
        ans_type = item['data']['answer_type']
        vid = item['data']['vid_id']
        qid = item['question_id']
        if multi_ref_ans and (vid in add_ref):
            if ans_type not in ['DB', 'DC']:
                w_0 = max(get_wups(pred, answer, 0),
                          get_wups(pred, add_ref[vid][qid], 0))
                w_9 = max(get_wups(pred, answer, 0.9),
                          get_wups(pred, add_ref[vid][qid], 0.9))
            else:
                w_0 = 1 if pred == answer else 0
                w_9 = w_0
        else:
            if ans_type not in ['DB', 'DC']:
                w_0 = get_wups(pred, answer, 0)
                w_9 = get_wups(pred, answer, 0.9)
            else:
                w_0 = 1 if pred == answer else 0
                w_9 = w_0

        wups_0.append(w_0)
        wups_9.append(w_9)

        questions.append(question)
        preds.append(pred)
        answers.append(answer)
        q_type.append(ans_type)

    data_dict = {'questions': questions,
                 'q_type': q_type,
                 'preds': preds,
                 'answers': answers,
                 'wups_0': wups_0,
                 'wups_9': wups_9}

    frame = pd.DataFrame(data_dict)
    frame.to_csv(output_path, header=True, index=True)


if __name__ == '__main__':

    # main()
    # data_path = '/data/fukunhao/NExT-QA/nextqa_oe/train.csv'
    # save_path = '/data/fukunhao/NExT-QA/nextqa_oe/multi_vocab.pkl'
    # vocab = generate_multi_vocab(data_path)
    # with open(save_path, 'wb') as f:
    #     pickle.dump(vocab, f)
    # print("Total vocabulary size: {}".format(len(vocab)))
    # print("Saved the vocabulary wrapper to '{}'".format(save_path))
    # a = torch.tensor([1, 2, 3])
    # b = torch.tensor([4, 5, 6])
    # path = '/data/fukunhao/ext_ckpt/model_base.pth'
    # view_pth(path, True)
    path = '/data/fukunhao/ALPRO/finetune/next_qa_oe/20230222151203/results_test/step_10800_1_mean_3/results_all.json'
    output_path = '/data/fukunhao/Data_Record/ALPRO_20230222151203_test_result_3.csv'
    ref_file_add = '/data/fukunhao/NExT-QA/nextqa_oe/add_reference_answer_test.json'
    process_json(path, output_path, ref_file_add)





