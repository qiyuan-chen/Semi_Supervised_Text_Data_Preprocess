# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import random
random.seed(1)
# arguments to be parsed from command line
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str,
                help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str,
                help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int,
                help="number of augmented sentences per original sentence")
ap.add_argument("--alpha_sr", required=False, type=float,
                help="percent of words in each sentence to be replaced by synonyms")
ap.add_argument("--alpha_ri", required=False, type=float,
                help="percent of words in each sentence to be inserted")
ap.add_argument("--alpha_rs", required=False, type=float,
                help="percent of words in each sentence to be swapped")
ap.add_argument("--alpha_rd", required=False, type=float,
                help="percent of words in each sentence to be deleted")
args = ap.parse_args()

# the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join('eda_' + basename(args.input))

# number of augmented sentences to generate per original sentence
num_aug = 1  # default
# 设置增强数量为1，满足flitext要求
if args.num_aug:
    num_aug = args.num_aug

# how much to replace each word by synonyms
alpha_sr = 0.5  # default
if args.alpha_sr is not None:
    alpha_sr = args.alpha_sr

# how much to insert new words that are synonyms
alpha_ri = 0.5  # default
if args.alpha_ri is not None:
    alpha_ri = args.alpha_ri

# how much to swap words
alpha_rs = 0.5  # default
if args.alpha_rs is not None:
    alpha_rs = args.alpha_rs

# how much to delete words
alpha_rd = 0.3  # default
if args.alpha_rd is not None:
    alpha_rd = args.alpha_rd

if alpha_sr == alpha_ri == alpha_rs == alpha_rd == 0:
    ap.error('At least one alpha should be greater than zero')

# generate more data with standard augmentation


def gen_eda(train_orig, output_file, alpha_sr, alpha_ri, alpha_rs, alpha_rd, num_aug=9):

    writer = open(output_file, 'w')
    lines = open(train_orig, 'r').readlines()

    for i, line in enumerate(lines):
        """parts = line[:-1].split('\t')
        label = parts[0]
        sentence = parts[1] """
        # 修改数据集组织形式
        sentence = line[:-3]
        label = line[-2]
        if(len(sentence) > 3):
            aug_sentences = eda(sentence, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
                                alpha_rs=alpha_rs, p_rd=alpha_rd, num_aug=1)
        # 修改num_agu=1
        for aug_sentence in aug_sentences:
            writer.write(sentence+"\t"+label+"\t"+aug_sentence+'\n')
            break
            # 修改数据集的形式为 原始句子+增强句子+标签

    writer.close()
    print("generated augmented sentences with eda for " + train_orig +
          " to " + output_file + " with num_aug=" + str(num_aug))


# main function
if __name__ == "__main__":

    # generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha_sr=alpha_sr, alpha_ri=alpha_ri,
            alpha_rs=alpha_rs, alpha_rd=alpha_rd, num_aug=num_aug)
