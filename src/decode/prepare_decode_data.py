import os
import argparse
import logging
import subprocess
import sys
import shutil
import re
import math
from os import path


class DictionaryInfo:
  """用于保存词典相关的一些参数.

  Attributes:
    dict_dir: 词典最终结果所保存的文件夹.
    lex_disambig_path: 消歧的概率化词汇文件路径.
    ndisambig_path: 消歧编号的文件路径.
    lex_disambig_index: 词汇标签的消歧符号起始编号.
    token_disambig_index: token标签的消歧符号起始编号.
    word_disambig_index: 词的消歧符号起始编号.

  """


  def __init__(self, temp_dict_dir, dict_dir):
    self.dict_dir = dict_dir
    lex_disambig_path = path.join(temp_dict_dir, "lexiconp_disambig.txt")
    self.lex_disambig_path = lex_disambig_path
    prob_lex_path = os.path.join(temp_dict_dir, "lexiconp.txt")
    script_path = "utils/add_lex_disambig.pl"
    add_lex_disambig_args = [script_path, prob_lex_path, lex_disambig_path]
    ndisambig = int(subprocess.check_output(add_lex_disambig_args)) + 1
    self.ndisambig_path = os.path.join(temp_dict_dir, "disambig.list")

    with open(self.ndisambig_path, "w", encoding="utf-8") as ndisambig_file:
      for i in range(ndisambig):
        ndisambig_file.write("#" + str(i) + "\n")

    self.lex_disambig_index = ndisambig
    unit_path = os.path.join(temp_dict_dir, "units.list")
    phoneme_unit_path = os.path.join(dict_dir, "units.txt")

    with open(phoneme_unit_path, encoding="utf-8") as phoneme_unit_file, \
        open(unit_path, "w", encoding="utf-8") as pure_phn_unit:
      for line in phoneme_unit_file:
        phoneme_unit = line.split()[0]
        pure_phn_unit.write(phoneme_unit + "\n")

    pure_phn_unit_path = os.path.join(temp_dict_dir, "units.list")
    tokens_path = os.path.join(dict_dir, "tokens.txt")

    with open(pure_phn_unit_path, encoding="utf-8") as phoneme_unit_file, \
        open(self.ndisambig_path, encoding="utf-8") as ndisambig_file, \
        open(tokens_path, "w", encoding="utf-8") as full_tokens_file:
      num = 0
      full_tokens_file.write("<eps> " + str(num) + "\n")
      num += 1
      full_tokens_file.write("<blk> " + str(num) + "\n")
      num += 1

      for line in phoneme_unit_file:
        full_tokens_file.write(line.strip() + " " + str(num) + "\n")
        num += 1

      self.token_disambig_index = num

      for line in ndisambig_file:
        full_tokens_file.write(line.strip() + " " + str(num) + "\n")
        num += 1

      self.word_disambig_index = _index_word(prob_lex_path, dict_dir)


def _prepare_environment():
  """设置执行shell的环境变量, 为了以后调用一些Essen的脚本.

  """
  fst_bin_path = "/home/rukuang/software/eesen/src/fstbin/"
  decode_bin_path = "/home/rukuang/software/eesen/src/decoderbin/"
  pwd = sys.path[0]
  utils = path.join(pwd, "utils/")
  paths = [pwd, utils, fst_bin_path, decode_bin_path, os.environ["PATH"]]
  os.environ["PATH"] = ":".join(paths)
  os.environ["LC_ALL"] = "C"
  logging.debug("系统PATH变量: " + os.environ["PATH"])


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


def prepare_phn_dict(dict_path, dict_dir):
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


def __add_prob_to_lexicon(lexicon_path, prob_lexicon_path):
  """
  给词汇表文件加入概率, 主要是因为后面有相关的脚本文件一定需要概率.
  :param lexicon_path: 词汇表路径.
  :param prob_lexicon_path: 增加了概率的词汇表路径.
  """
  pattern = re.compile(r'(\S+\s+)(.+)')

  with open(lexicon_path, encoding="utf-8") as lexicon_file, \
      open(prob_lexicon_path, "w", encoding="utf-8") as prob_lexicon_file:
    for line in lexicon_file:
      prob_lexicon_file.write(re.sub(pattern, r"\g<1>1.0\t\g<2>", line))


def _index_word(prob_lex_path, target_dir):
  """给词典中的每个词附加编号.

  :param prob_lex_path: 概率化词汇表路径.
  :param target_dir: 目标文件夹.
  :return 消歧编号
  """
  word_path = os.path.join(target_dir, "words.txt")

  with open(prob_lex_path, encoding="utf-8") as prob_lex_file, \
      open(word_path, "w", encoding="utf-8") as words_num_file:
    words_num_file.write("<eps> 0\n")
    num = 1

    for line in prob_lex_file:
      word = line.split()[0]
      words_num_file.write(word + " " + str(num) + "\n")
      num += 1

    words_num_file.write("#0 " + str(num) + "\n")

  return num


def __get_token_fst_script(tokens_path, token_fst_path, fst_input_path):
  """拼接ctc_token_fst脚本的命令行参数.

  Args:
    tokens_path: 全token列表的文件路径.
    token_fst_path: token fst的路径.
    fst_input_path: fst脚本输入文件的路径.

  Returns:
    拼接好的命令行.

  """
  return ("fstcompile --isymbols={0} --osymbols={0} "
          "--keep_isymbols=false --keep_osymbols=false {2} | "
          "fstarcsort --sort_type=olabel > "
          "{1} || exit 1").format(tokens_path, token_fst_path, fst_input_path)


def __generate_token_fst_input_file(tokens_path, temp_dir):
  """生成符合fst格式的输入文件.

  该方法是ctc_token_fst的重写版本.

  :param tokens_path: token标签文件的路径.
  :param temp_dir: 生成fst格式文件的保存路径.
  :return: fst格式输入文件的路径.
  """
  fst_input_path = os.path.join(temp_dir, "fst_input")

  with open(tokens_path, encoding="utf-8") as tokens_file, \
      open(fst_input_path, "w", encoding="utf-8") as fst_input_file:
    fst_input_file.write('0 1 <eps> <eps>\n')
    fst_input_file.write('1 1 <blk> <eps>\n')
    fst_input_file.write('2 2 <blk> <eps>\n')
    fst_input_file.write("2 0 <eps> <eps>\n")
    node_x = 3

    for entry in tokens_file:
      fields = entry.strip().split(' ')
      phone = fields[0]

      if phone == '<eps>' or phone == '<blk>':
        continue

      if '#' in phone:
        line = str(0) + ' ' + str(0) + ' <eps> ' + phone
        fst_input_file.write(line + "\n")
      else:
        line = str(1) + ' ' + str(node_x) + ' ' + phone + ' ' + phone
        fst_input_file.write(line + "\n")
        line = str(node_x) + ' ' + str(node_x) + ' ' + phone + ' <eps>'
        fst_input_file.write(line + "\n")
        line = str(node_x) + ' ' + str(2) + ' ' + '<eps> <eps>'
        fst_input_file.write(line + "\n")

      node_x += 1

    fst_input_file.write("0\n")

  return fst_input_path


def __make_lex_fst(lex_path, pron_probs,
                   sil_prob=0, sil_phone=None, sil_disambig_sym=None):
  if sil_prob != 0:
    if sil_prob >= 1:
      raise ValueError("Sil prob 不能 >= 1.0")

    sil_cost = -math.log(sil_prob)
    no_sil_cost = -math.log(1.0 - sil_prob)


    def __is_sil(seq):
      seq_len = len(seq)
      if seq_len == 1 and seq[0] == sil_phone or (
                seq_len == 3 and seq[1] == sil_phone and seq[0]):
        # TODO(Kuang.Ru): 这里的条件判断需要调整.
        pass


    with open(lex_path, encoding="utf-8") as lex_file:
      # TODO(Kuang.Ru): 读取文件内容的部分需要完成.
      pass


def __format_lex_fst_script(dict_info):
  """构建标准的生成词汇fst的脚本.

  Args:
    dict_info: 词典相关的类, 里面包含了脚本所需的各种参数

  Returns:
    构建完成的shell执行脚本.

  """
  return ("utils/make_lexicon_fst.pl "
          "--pron-probs {0} "
          """0 "sil" '#'{1} | """
          "fstcompile --isymbols={2}/tokens.txt "
          "--osymbols={2}/words.txt --keep_isymbols=false "
          "--keep_osymbols=false | fstaddselfloops  "
          """"echo {3} |" """
          """"echo {4} |" | """
          "fstarcsort --sort_type=olabel > "
          "{2}/L.fst").format(dict_info.lex_disambig_path,
                              dict_info.lex_disambig_index,
                              dict_info.dict_dir,
                              dict_info.token_disambig_index,
                              dict_info.word_disambig_index)


def ctc_compile_dict_token(src_dir, temp_dir, target_dir):
  """生成词典和token的fst.

  Args:
    src_dir: 词典源文件夹, 由最原始的词典文件生成.
    temp_dir: 词典中间文件夹, 词典在转换过程中生成的中间文件保存位置.
    target_dir: 词典最终保存的文件夹.

  """
  dict_type = "phn"
  space_char = "<SPACE>"
  os.mkdir(temp_dir)
  os.mkdir(target_dir)
  shutil.copy(os.path.join(src_dir, "lexicon_numbers.txt"), target_dir)
  phoneme_unit_path = os.path.join(src_dir, "units.txt")
  shutil.copy(phoneme_unit_path, target_dir)
  lexicon_path = os.path.join(src_dir, "lexicon.txt")
  prob_lex_path = os.path.join(temp_dir, "lexiconp.txt")
  __add_prob_to_lexicon(lexicon_path, prob_lex_path)
  dict_info = DictionaryInfo(temp_dir, target_dir)
  tokens_path = os.path.join(target_dir, "tokens.txt")
  token_fst_path = os.path.join(target_dir, "T.fst")
  fst_input_path = __generate_token_fst_input_file(tokens_path, temp_dir)
  script = __get_token_fst_script(tokens_path, token_fst_path, fst_input_path)
  subprocess.run(script, shell=True)

  if dict_type == "phn":
    phn_script = __format_lex_fst_script(dict_info)
    subprocess.run(phn_script, shell=True)
  elif dict_type == "char":
    logging.info("构建基于词的词汇表, " + space_char + "作为分隔符.")

    char_script = ("utils/make_lexicon_fst.pl "
                   "--pron-probs $tmpdir/lexiconp_disambig.txt 0.5 "
                   """"$space_char" '#'$ndisambig |"""
                   "fstcompile --isymbols=$dir/tokens.txt "
                   "--osymbols=$dir/words.txt "
                   "--keep_isymbols=false --keep_osymbols=false |   "
                   """fstaddselfloops  "echo $token_disambig_symbol |" """
                   """""echo $word_disambig_symbol |" | """
                   "fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;")

    subprocess.run(char_script, shell=True)
  else:
    raise TypeError("无效的词典类型" + dict_type)

  logging.info("词典和Token的FST构建完成.")


def compose_all_fst(lang_dir, arpa_lm):
  """将所有的WFST合并成一张WFST.

  Args:
    lang_dir: 语言模型文件夹.
    arpa_lm: ARPA格式的语言模型.

  """
  oov_list = "/dev/null"
  out_lang_dir = lang_dir + "_test"
  shutil.rmtree(out_lang_dir, ignore_errors=True)
  shutil.copytree(lang_dir, out_lang_dir)

  form_grammar_fst_script = (
    """gunzip -c "{0}" | grep -v '<s> <s>' | grep -v '</s> <s>' | """
    "grep -v '</s> </s>' | arpa2fst - | fstprint | "
    "utils/remove_oovs.pl {1} | utils/eps2disambig.pl | utils/s2eps.pl | "
    "fstcompile --isymbols={2}/words.txt --osymbols={2}/words.txt  "
    "--keep_isymbols=false --keep_osymbols=false | fstrmepsilon | "
    "fstarcsort --sort_type=ilabel > {2}/G.fst"
  ).format(arpa_lm, oov_list, out_lang_dir)

  subprocess.run(form_grammar_fst_script, shell=True)

  compose_lex_and_grammar_fst_script = (
    "fsttablecompose {0}/L.fst {0}/G.fst | fstdeterminizestar --use-log=true | "
    "fstminimizeencoded | fstarcsort --sort_type=ilabel > {0}/LG.fst || exit 1"
  ).format(out_lang_dir)

  subprocess.run(compose_lex_and_grammar_fst_script, shell=True)

  compose_all_fst_script = ("fsttablecompose {0}/T.fst {0}/LG.fst > {0}/TLG.fst"
                            ).format(out_lang_dir)

  subprocess.run(compose_all_fst_script, shell=True)
  os.remove(path.join(out_lang_dir, "LG.fst"))
  logging.info("TLG.fst合并成功.")


if __name__ == '__main__':
  logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                      level=logging.DEBUG)

  logging.info("开始构建解码WFST.")
  _prepare_environment()
  parser = argparse.ArgumentParser()
  parser.add_argument("dict", help="词典文件.")
  parser.add_argument("lm", help="语言模型压缩文件.")
  args = parser.parse_args()
  prepare_phn_dict(args.dict, "dict_src_dir")
  logging.info("生成数值化词典文件.")

  # argument = ["utils/ctc_compile_dict_token.sh",  # 编译词典和token.
  #             "dict_src_dir",  # 上一步结果文件夹.
  #             "dict_temp_dir",  # 中间结果文件夹, 主要是规范词表的中间结果.
  #             "dict_fst"]  # 结果文件夹.
  #
  # subprocess.run(argument)
  ctc_compile_dict_token("dict_src_dir", "dict_temp_dir", "dict_fst")
  compose_all_fst("dict_fst", args.lm)
  logging.info("构建完成.")
