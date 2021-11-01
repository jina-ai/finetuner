from random import sample
from time import perf_counter

from finetuner.tuner.dataset.samplers import (RandomClassBatchSampler,
                                              SessionBatchSampler)

# labels = []

# for i in range(100_000):
#     labels += [i] * 6

# BS = 12
# s = RandomClassBatchSampler(labels, BS, 4)

# t = perf_counter()
# l = 0
# for b in s:
#     pass
#     l += 1
#     # print(b)
# print(perf_counter() - t)

# print(f'{300_000/BS}, {l=}')

# group_batches = []
# while len(id_group_counts) >= self._num_ids:
#     # group_batch = np.random.choice(
#     #     list(id_group_counts.skeys()), size=self._num_ids
#     # )

#     group_batch = sample(list(id_group_counts.keys()), self._num_ids)
#     group_batches.append(group_batch)

#     id_group_counts.subtract(group_batch)
#     id_group_counts = +id_group_counts  # Remove exhausted id groups


# t = perf_counter()
# K = 10_000
# for i in range(K):
#     a = list(range(K))
#     # a.remove(i)
#     del a[i]
# print(perf_counter() - t)

labels = []
for i in range(10_000):
    labels += [(i, 0)]
    labels += [(i, 1)] * 5
    labels += [(i, -1)] * 4

t = perf_counter()
s = SessionBatchSampler(labels, 32)
print(perf_counter() - t)

t = perf_counter()
print('go')
for i, p in enumerate(s):
    pass
print(i, len(s))
print(perf_counter() - t)
