# -*- coding:utf-8 -*-
import os
# os.environ["CHAINER_TYPE_CHECK"] = "0"

import pickle
import argparse
import nltk
import numpy as np
import chainer
from chainer import cuda, optimizers, serializers
from util import ConvCorpus
from seq2seq import Seq2Seq
from wer import wer


# parse command line args
parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', default='data/pair_corpus.txt', type=str, help='Data file directory')
parser.add_argument('--epoch', '-e', default=1, type=int, help='number of epochs to learn')
parser.add_argument('--feature_num', '-f', default=1024, type=int, help='dimension of feature layer')
parser.add_argument('--hidden_num', '-hi', default=1024, type=int, help='dimension of hidden layer')
parser.add_argument('--batchsize', '-b', default=100, type=int, help='learning minibatch size')
parser.add_argument('--testsize', '-t', default=1000, type=int, help='number of text for testing a model')
args = parser.parse_args()

data_file = args.data
n_epoch = args.epoch
feature_num = args.feature_num
hidden_num = args.hidden_num
batchsize = args.batchsize
testsize = args.testsize


def main():

    print()
    print("###########################")
    print()
    print(args)
    print()
    print("#### create dictionary ####")
    print("###########################")
    print()

    ###########################
    #### create dictionary ####
    ###########################

    # this return the list containing list of tokenid 
    if os.path.exists('./data/corpus/dictionary.dict'):
        corpus = ConvCorpus(file_path=None, batch_size=batchsize)
        corpus.load(load_dir='./data/corpus/')
    else:
        print("data_file ",data_file)
        corpus = ConvCorpus(file_path=data_file, batch_size=batchsize)
        corpus.save(save_dir='./data/corpus/')
    print('Vocabulary Size (number of words) :', len(corpus.dic.token2id))

    print()
    print("######################")
    print("#### create model ####")
    print("######################")
    print()

    ######################
    #### create model ####
    ######################

    # data_file = args.data
    # n_epoch = args.epoch
    # feature_num = args.feature_num
    # hidden_num = args.hidden_num
    # batchsize = args.batchsize
    # testsize = args.testsize
    print("printing model parameters")
    print("data file name : ",args.data)
    print("epoch : ",args.epoch)
    print("feature number  : ",args.feature_num)
    print("hidden number : ",args.hidden_num)
    print("batch size : ",args.batchsize)
    print("test size : ",args.testsize)
    print()
    
    # :param vocab_size: input vocab size
    # :param feature_num: size of feature layer (embed layer)
    # :param hidden_num: size of hidden layer
    model = Seq2Seq(len(corpus.dic.token2id), feature_num=feature_num,
                    hidden_num=hidden_num, batch_size=batchsize, gpu_flg=-1)

    exit(1)
    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))
    # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))

    print()
    print("##########################")
    print("#### create ID corpus ####")
    print("##########################")
    print()

    ##########################
    #### create ID corpus ####
    ##########################

    input_mat = []
    output_mat = []
    max_input_ren = max_output_ren = 0

    for input_text, output_text in zip(corpus.posts, corpus.cmnts):

        # convert to list
        input_text.reverse()         # encode words in a reverse order
        input_text.insert(0, corpus.dic.token2id["<eos>"])
        output_text.append(corpus.dic.token2id["<eos>"])

        # update max sentence length
        max_input_ren = max(max_input_ren, len(input_text))
        max_output_ren = max(max_output_ren, len(output_text))

        input_mat.append(input_text)
        output_mat.append(output_text)

    # padding
    for li in input_mat:
        insert_num = max_input_ren - len(li)
        for _ in range(insert_num):
            li.insert(0, corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_ren - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])

    # create batch matrix
    input_mat = np.array(input_mat, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T

    # separate corpus into Train and Test
    perm = np.random.permutation(len(corpus.posts))
    test_input_mat = input_mat[:, perm[0:0 + testsize]]
    test_output_mat = output_mat[:, perm[0:0 + testsize]]
    train_input_mat = input_mat[:, perm[testsize:]]
    train_output_mat = output_mat[:, perm[testsize:]]

    list_of_references = []
    for text_ndarray in test_output_mat.T:
        reference = text_ndarray.tolist()
        references = [[w_id for w_id in reference if w_id != -1]]
        list_of_references.append(references)

    #############################
    #### train seq2seq model ####
    #############################

    print()
    print("#############################")
    print("#### train seq2seq model ####")
    print("#############################")
    print()

    accum_loss = 0
    train_loss_data = []
    test_loss_data = []
    bleu_score_data = []
    wer_score_data = []
    for num, epoch in enumerate(range(n_epoch)):
        total_loss = test_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.posts) - testsize)

        # for training
        for i in range(0, len(corpus.posts) - testsize, batchsize):

            # select batch data
            input_batch = train_input_mat[:, perm[i:i + batchsize]]
            output_batch = train_output_mat[:, perm[i:i + batchsize]]

            # Encode a sentence
            model.initialize()                     # initialize cell
            model.encode(input_batch, train=True)  # encode (output: hidden Variable)

            # Decode from encoded context
            end_batch = np.array([corpus.dic.token2id["<start>"] for _ in range(batchsize)])
            first_words = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_words, train=True)
            next_ids = first_words
            accum_loss += loss
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = w_ids
                accum_loss += loss

            # learn model
            model.cleargrads()     # initialize all grad to zero
            accum_loss.backward()  # back propagation
            optimizer.update()
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch: ', num, 'Batch_num', batch_num, 'batch loss: {:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

        # for testing
        list_of_hypotheses = []
        for i in range(0, testsize, batchsize):

            # select test batch data
            input_batch = test_input_mat[:, i:i + batchsize]
            output_batch = test_output_mat[:, i:i + batchsize]

            # Encode a sentence
            model.initialize()                     # initialize cell
            model.encode(input_batch, train=True)  # encode (output: hidden Variable)

            # Decode from encoded context
            end_batch = np.array([corpus.dic.token2id["<start>"] for _ in range(batchsize)])
            first_words = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_words, train=True)
            next_ids = np.argmax(predict_mat.data, axis=1)
            test_loss += loss
            if args.gpu >= 0:
                hypotheses = [cuda.to_cpu(next_ids)]
            else:
                hypotheses = [next_ids]
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = np.argmax(predict_mat.data, axis=1)
                test_loss += loss
                if args.gpu >= 0:
                    hypotheses.append(cuda.to_cpu(next_ids))
                else:
                    hypotheses.append(next_ids)

            # collect hypotheses for calculating BLEU score
            hypotheses = np.array(hypotheses).T
            for hypothesis in hypotheses:
                text_list = hypothesis.tolist()
                list_of_hypotheses.append([w_id for w_id in text_list if w_id != -1])

        # calculate BLEU score from test (develop) data
        bleu_score = nltk.translate.bleu_score.corpus_bleu(list_of_references, list_of_hypotheses,
                                                           weights=(0.25, 0.25, 0.25, 0.25))
        bleu_score_data.append(bleu_score)
        print('Epoch: ', num, 'BLEU SCORE: ', bleu_score)

        # calculate WER score from test (develop) data
        wer_score = 0
        for index, references in enumerate(list_of_references):
            wer_score += wer(references[0], list_of_hypotheses[index])
        wer_score /= len(list_of_references)
        wer_score_data.append(wer_score)
        print('Epoch: ', num, 'WER SCORE: ', wer_score)

        print("#############################")
        print("condition to save the model ",(epoch + 1) % 10 == 0)
        print("#############################")
        # save model and optimizer
        
        # save only after every 10 epochs
        # if (epoch + 1) % 10 == 0:
        print('-----', epoch + 1, ' times -----')
        print("#############################")
        print('save the model and optimizer')
        print("#############################")
        serializers.save_hdf5('data/' + str(epoch) + '.model', model)
        serializers.save_hdf5('data/' + str(epoch) + '.state', optimizer)

        # display the on-going status
        print('Epoch: ', num,
              'Train loss: {:.2f}'.format(total_loss),
              'Test loss: {:.2f}'.format(float(test_loss.data)))
        train_loss_data.append(float(total_loss / batch_num))
        test_loss_data.append(float(test_loss.data))

        # evaluate a test loss
        check_loss = test_loss_data[-10:]           # check out the last 10 loss data
        end_flg = [j for j in range(len(check_loss) - 1) if check_loss[j] < check_loss[j + 1]]
        if len(end_flg) > 8:
            print('Probably it is over-fitting. So stop to learn...')
            break

    # save loss data
    with open('./data/loss_train_data.pkl', 'wb') as f:
        pickle.dump(train_loss_data, f)
    with open('./data/loss_test_data.pkl', 'wb') as f:
        pickle.dump(test_loss_data, f)
    with open('./data/bleu_score_data.pkl', 'wb') as f:
        pickle.dump(bleu_score_data, f)
    with open('./data/wer_score_data.pkl', 'wb') as f:
        pickle.dump(wer_score_data, f)


if __name__ == "__main__":
    main()