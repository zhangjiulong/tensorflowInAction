import fst


def _get_arc(state, input):
    """
    输出状态中符合要求的出边.
    """
    print(state.stateid)
    for arc in state.arcs:
        print(arc)
        if arc.ilabel == input:
            return arc

    return None


def _print_arcs(state):
    """
    将状态的出边打印出来.
    """
    for arc in state.arcs:
        print(arc)


def _get_num_label_id():
    """
    获取token映射表中消歧所使用的标量id.
    """
    with open("dict_fst_test/tokens.txt", encoding="utf-8") as tokens:
        for line in tokens:
            phoneme, id = line.strip().split()

            if "#" in phoneme:
                return int(id)


if __name__ == '__main__':
    search_graph = fst.read("dict_fst_test/TLG.fst")
    phone_seq = [1, 86, 0, 0, 0, 1, 36, 0, 0, 0, 1, 84, 0, 0, 0, 1, 13]
    state = search_graph[1]
    _print_arcs(search_graph[67])
    log_prob = 0
    output_seq = list()

    for i in phone_seq:
        arc = _get_arc(state, i)
        print("相关边为:", arc)
        log_prob += float(arc.weight)

        if arc.olabel != 0:
            output_seq.append(arc.olabel)

        state = search_graph[arc.nextstate]

    candidate_arcs = list()
    num_label_id = _get_num_label_id()
    print(output_seq, len(output_seq))

    if len(output_seq) == 0:
        for arc in state.arcs:
            if arc.ilabel >= num_label_id:
                candidate_arcs.append(arc)

        if len(candidate_arcs) != 0:
            min_logp = float(candidate_arcs[0].weight)
            min_arc = candidate_arcs[0]

            for arc in candidate_arcs[1:]:
                if float(arc.weight) < min_logp:
                    min_arc = arc
                    min_logp = float(arc.weight)

            print(min_arc)
            log_prob += min_logp
            output_seq.append(min_arc.olabel)

    print(log_prob)
    id_to_word = dict()

    with open("dict_fst_test/words.txt", encoding="utf-8") as words:
        for line in words:
            word, id = line.strip().split()
            id = int(id)
            id_to_word[id] = word

    for i in output_seq:
        print(id_to_word[i])
