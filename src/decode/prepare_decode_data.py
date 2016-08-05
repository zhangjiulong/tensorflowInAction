import os
import argparse
import logging
import subprocess
import sys
import shutil
import re


def _prepare_environment():
    """
    设置执行shell的环境变量, 为了以后调用一些Essen的脚本.
    """
    pwd = sys.path[0]
    os.environ["PATH"] = pwd + ":" + os.environ["PATH"]
    os.environ["LC_ALL"] = "C"


def _writelines(file_path, lines):
    """
    将一个列表写入到文件中, 主要的改进是会根据列表的结尾字符自动规划换行符.
    :param file_path: 需要写入的文件.
    :param lines: 需要写入文件的列表
    """
    with open(file_path, "w", encoding="utf-8") as file:
        for line in lines:
            if line.endswith("\n"):
                file.write(line)
            else:
                file.write(line + "\n")


def _prepare_phn_dict(dict_path, dict_dir):
    """
    通过原始的字典文件, 生成对应的各种标签文件, 以及数值化的词典
    :param dict_path: 词典文件
    :param dict_dir: 词典对应的各种文件保存的文件夹
    """
    os.mkdir(dict_dir)
    standard_dict = list()

    with open(dict_path, encoding="utf-8") as dict_file:
        for line in dict_file:
            if "<s>" not in line and "</s>" not in line:
                standard_dict.append(line)

    standard_dict.sort()
    std_lexicon_file_path = os.path.join(dict_dir, "lexicon_words.txt")
    _writelines(std_lexicon_file_path, standard_dict)

    phonemes = set()

    for line in standard_dict:
        word_phonemes = line.strip().split()[1:]
        phonemes |= set(word_phonemes)

    sorted_phonemes = sorted(phonemes)
    unit_noiseless_path = os.path.join(dict_dir, "units_nosil.txt")
    _writelines(unit_noiseless_path, sorted_phonemes)
    standard_dict.append("[BREATH] BRH\n")
    standard_dict.append('[NOISE] NSN\n')
    standard_dict.append('[COUGH] CGH\n')
    standard_dict.append("[SMACK] SMK\n")
    standard_dict.append("[UM] UM\n")
    standard_dict.append("[UH] UHH\n")
    standard_dict.append("<UNK> NSN\n")
    standard_dict.sort()
    noisy_lexicon_path = os.path.join(dict_dir, "lexicon.txt")
    _writelines(noisy_lexicon_path, standard_dict)
    sorted_phonemes.insert(0, "BRH")
    sorted_phonemes.insert(1, "CGH")
    sorted_phonemes.insert(2, "NSN")
    sorted_phonemes.insert(3, "SMK")
    sorted_phonemes.insert(4, "UM")
    sorted_phonemes.insert(5, "UHH")
    unit_path = os.path.join(dict_dir, "units.txt")

    with open(unit_path, "w", encoding="utf-8") as unit_file:
        for index, phoneme in enumerate(sorted_phonemes, 1):
            unit_file.write(phoneme + " " + str(index) + "\n")

    argument = ["utils/sym2int.pl", "-f", "2-", unit_path, noisy_lexicon_path]
    lexicon_number = subprocess.check_output(argument).decode("utf-8")
    lexicon_number_path = os.path.join(dict_dir, "lexicon_numbers.txt")

    with open(lexicon_number_path, "w",
              encoding="utf-8") as lexicon_number_file:
        lexicon_number_file.write(lexicon_number)


def _add_prob_to_lexicon(lexicon_path, prob_lexicon_path):
    """
    给词汇表文件加入概率, 主要是因为后面有相关的脚本文件一定需要概率.
    :param lexicon_path: 词汇表路径.
    :param prob_lexicon_path: 增加了概率的词汇表路径.
    """
    pattern = re.compile(r'(\S+\s+)(.+)')

    with open(lexicon_path, encoding="utf-8") as lexicon_file, \
            open(prob_lexicon_path, "w", encoding="utf-8") as prob_lexicon_file:
        for line in lexicon_file:
            prob_lexicon_file.write(pattern.sub(r"\11.0\t\2", line))


def _generate_disambig_list(store_dir):
    """
    生成消歧标签, 在确定化的过程中需要, 否则确定化无法正确进行.
    :param store_dir: 工作的文件夹, 里面需要有lexiconp.txt.
    :return 返回保存消歧标签列表的文件路径.
    """
    prob_lex_path = os.path.join(store_dir, "lexiconp.txt")
    disambig_prob_lex_path = os.path.join(store_dir, "lexiconp_disambig.txt")
    script_path = "add_lex_disambig.pl"
    add_lex_disambig_args = [script_path, prob_lex_path, disambig_prob_lex_path]
    ndisambig = int(subprocess.check_output(add_lex_disambig_args)) + 1
    ndisambig_path = os.path.join(store_dir, "disambig.list")

    with open(ndisambig_path, "w", encoding="utf-8") as ndisambig_file:
        for i in range(ndisambig):
            ndisambig_file.write("#" + str(i) + "\n")

    return ndisambig_path


def _ctc_compile_dict_token(src_dir, temp_dir, target_dir):
    dict_type = "phn"
    space_char = "<SPACE>"
    os.mkdir(temp_dir)
    os.mkdir(target_dir)
    shutil.copy(os.path.join(src_dir, "lexicon_numbers.txt"), target_dir)
    phoneme_unit_path = os.path.join(src_dir, "units.txt")
    shutil.copy(phoneme_unit_path, target_dir)
    lexicon_path = os.path.join(src_dir, "lexicon.txt")
    prob_lex_path = os.path.join(temp_dir, "lexiconp.txt")
    _add_prob_to_lexicon(lexicon_path, prob_lex_path)
    ndisambig_path = _generate_disambig_list(temp_dir)
    pure_phn_unit_path = os.path.join(temp_dir, "units.list")

    with open(phoneme_unit_path, encoding="utf-8") as phoneme_unit_file, \
            open(pure_phn_unit_path, "w", encoding="utf-8") as pure_phn_unit:
        for line in phoneme_unit_file:
            phoneme_unit = line.split()[0]
            pure_phn_unit.write(phoneme_unit + "\n")

    full_tokens_path = os.path.join(target_dir, "tokens.txt")

    with open(pure_phn_unit_path, encoding="utf-8") as phoneme_unit_file, \
            open(ndisambig_path, encoding="utf-8") as ndisambig_file, \
            open(full_tokens_path, "w", encoding="utf-8") as full_tokens_file:
        num = 0
        full_tokens_file.write("<eps> " + str(num) + "\n")
        num += 1
        full_tokens_file.write("<blk> " + str(num) + "\n")
        num += 1

        for line in phoneme_unit_file:
            full_tokens_file.write(line.strip() + str(num) + "\n")
            num += 1

        for line in ndisambig_file:
            full_tokens_file.write(line.strip() + str(num) + "\n")
            num += 1

    script = "utils/ctc_token_fst.py $dir/tokens.txt | fstcompile " \
             "--isymbols=$dir/tokens.txt --osymbols=$dir/tokens.txt " \
             "--keep_isymbols=false --keep_osymbols=false | " \
             "fstarcsort --sort_type=olabel > $dir/T.fst"

    subprocess.run(script, shell=True)


if __name__ == '__main__':
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                        level=logging.INFO)

    logging.info("开始构建解码WFST.")
    _prepare_environment()
    parser = argparse.ArgumentParser()
    parser.add_argument("dict", help="词典文件.")
    parser.add_argument("lm", help="语言模型压缩文件.")
    args = parser.parse_args()
    _prepare_phn_dict(args.dict, "dict_src_dir")
    logging.info("生成数值化词典文件.")

    argument = ["utils/ctc_compile_dict_token.sh",  # 编译词典和token.
                "dict_src_dir",  # 上一步结果文件夹.
                "dict_temp_dir",  # 中间结果文件夹, 主要是规范词表的中间结果.
                "dict_fst"]  # 结果文件夹.

    subprocess.run(argument)
    subprocess.run(["tedlium_decode_graph.sh", "dict_fst", args.lm])
    logging.info("构建完成.")
