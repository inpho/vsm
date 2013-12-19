




def bf_sim_walk(sim_fn, word, go_list, global_bnd):

    mem = { word:
            { 'pos': 0,
              'sim': sim_fn(word) }}

    while len(mem.keys()) < global_bnd:

        if not False in [len(v['sim']) == v['pos'] for v in mem.values()]:

            break

        for k,v in mem.items():

            pos, sim = v['pos'], v['sim']

            for p in sim[pos:]:

                mem[k]['pos'] += 1

                if p[0] in go_list:

                    if p[0] not in mem:

                        mem[p[0]] = { 'pos': 0,
                                      'sim': sim_fn(p[0]) }

                    break

    return mem.keys()



def df_sim_walk(sim_fn, word, go_list, local_bnd, global_bnd):

    mem = { word: True }

    while len(mem.keys()) < global_bnd:

        if not True in mem.values():

            break

        for t,f in mem.items():

            if f:

                mem[t] = False

                l = sim_fn(t)

                i = 0

                for t_,v in l:

                    if t_ in go_list:

                        i += 1

                        if t_ not in mem:

                            mem[t_] = False

                        if i == local_bnd - 1:

                            break

    return mem.keys()



def test_sim_fn(c):

    data = {
        'a': ['j', 'l', 'a', 'h', 'k', 'i', 'g', 'm', 'd', 'c', 'b', 'f', 'e'],
        'b': ['d', 'k', 'm', 'f', 'h', 'g', 'b', 'e', 'i', 'c', 'l', 'a', 'j'],
        'c': ['f', 'k', 'j', 'l', 'e', 'm', 'i', 'h', 'c', 'g', 'a', 'd', 'b'],
        'd': ['a', 'd', 'g', 'l', 'e', 'm', 'j', 'i', 'h', 'c', 'f', 'k', 'b'],
        'e': ['j', 'm', 'g', 'd', 'c', 'e', 'a', 'f', 'k', 'h', 'i', 'b', 'l'],
        'f': ['b', 'f', 'h', 'g', 'a', 'm', 'e', 'd', 'i', 'c', 'k', 'l', 'j'],
        'g': ['a', 'l', 'c', 'g', 'j', 'd', 'k', 'b', 'f', 'i', 'h', 'm', 'e'],
        'h': ['k', 'h', 'd', 'j', 'f', 'l', 'e', 'a', 'm', 'g', 'i', 'c', 'b'],
        'i': ['l', 'c', 'd', 'i', 'e', 'a', 'j', 'k', 'f', 'g', 'h', 'b', 'm'],
        'j': ['e', 'a', 'f', 'b', 'g', 'i', 'c', 'm', 'l', 'j', 'h', 'd', 'k'],
        'k': ['f', 'k', 'b', 'i', 'h', 'g', 'e', 'c', 'l', 'a', 'j', 'm', 'd'],
        'l': ['e', 'j', 'g', 'h', 'a', 'k', 'l', 'i', 'b', 'c', 'd', 'm', 'f'],
        'm': ['d', 'f', 'l', 'b', 'c', 'e', 'i', 'k', 'g', 'm', 'a', 'j', 'h']
        }

    for d,v in data.items():

        data[d] = [(l, 0) for l in v]

    return data[c]

