import pandas as pd
from scipy.special import binom

server_speed = 0.07786936994  # количество секунд / (k * C_n^k) * число процессов

n_proc = 8
target_time_n_k = 2 * 24 * 3600 / 10  # Время в секундах на одну пару n, k. Сетка k от 1 до 10, будем считать 2 дня

df = pd.DataFrame(columns=["n", "k"])
for k in range(1, 11):
    n = 15000  # Число всех генов (с запасом)
    while 1:
        time_n_k = server_speed * k*binom(n, k) / n_proc  # Сколько будет считаться данная пара
        if time_n_k <= target_time_n_k:
            df.loc[len(df), ["n", "k"]] = [n, k]
            break
        else:
            n -= 1
    k += 1

df.to_csv("n_k.tsv", sep="\t", index=None)
