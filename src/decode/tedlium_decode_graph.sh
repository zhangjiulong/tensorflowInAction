#!/bin/bash
# Author: Kuang.Ru
# 需要
# This script compiles the ARPA-formatted language models into FSTs. Finally it composes the LM, lexicon
# and token FSTs together into the decoding graph.

# 将一些需要的程序载入路径.
. ./path.sh || exit 1;

oov_list=/dev/null

. parse_options.sh || exit 1;

lang_dir=$1
arpa_lm=$2

[ ! -f ${arpa_lm} ] && echo No such file ${arpa_lm} && exit 1;

out_lang_dir=${lang_dir}_test

rm -rf ${out_lang_dir}
cp -r ${lang_dir} ${out_lang_dir}

gunzip -c "$arpa_lm" | \
   grep -v '<s> <s>' | \
   grep -v '</s> <s>' | \
   grep -v '</s> </s>' | \
   arpa2fst - | fstprint | \
   utils/remove_oovs.pl ${oov_list} | \
   utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=${out_lang_dir}/words.txt \
     --osymbols=${out_lang_dir}/words.txt  --keep_isymbols=false --keep_osymbols=false | \
    fstrmepsilon | fstarcsort --sort_type=ilabel > ${out_lang_dir}/G.fst

# Compose the final decoding graph. The composition of L.fst and G.fst is determinized and
# minimized.
# 这些程序在fstbin中.
# fsttablecompose和compose是等价的, 只不过在某些特殊类型的输入上更有效率.
fsttablecompose ${out_lang_dir}/L.fst ${out_lang_dir}/G.fst | fstdeterminizestar --use-log=true | \
  fstminimizeencoded | fstarcsort --sort_type=ilabel > ${out_lang_dir}/LG.fst || exit 1;
fsttablecompose ${out_lang_dir}/T.fst ${out_lang_dir}/LG.fst > ${out_lang_dir}/TLG.fst || exit 1;
rm -rf ${out_lang_dir}/LG.fst

echo "Composing decoding graph TLG.fst succeeded"
