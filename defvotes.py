def text_to_votes(test):
    tl = test.split(";")
    tl = [i.split(",") for i in tl]
    tldict = {i[0]: int(i[2].split()[0]) for i in tl}
    return tldict


defvotes14 = text_to_votes(def2104)

for k, v in defvotes15.items():
    defvotes15[k] = defvotes15[k]/3

for k, v in defvotes14.items():
    defvotes14[k] = defvotes14[k]/3

all3 = {15: defvotes15, 14: defvotes14, 13:defvotes13}

def sum_past_yrs(all3):
    new = all3
    for k, v in new.items():
        count = 1
        prior = k-count
        # 15 - prior
        while prior > 12:
            # p, v in 15
            for player, votes in new[k].items():
                # p, in 14
                if player in new[prior].keys():
                    # p15 + p14
                    new[k][player] += new[prior][player]
            prior -= 1
    return new
        