import numpy as np
from itertools import combinations_with_replacement, combinations, starmap
import multiprocessing as mp

"""
The sum of the Cartesian product for the set of dice.
Enumerates all of the possible sums of the dice and returns
them in a consistent order.
"""
def cartesian_sum(dice):
  return np.sum(np.dstack(np.meshgrid(*dice)).reshape(-1, len(dice)), axis=1)

"""
Only check die that are in the range 1,1,1,...,1 to 1,2,3,...,n
"""
def die_space(n=6):
  space = combinations_with_replacement(range(1, 2 * n - 1), n-1)
  sym = tuple(i for i in range(2, n+1))

  for faces in space:
    yield (1, *faces)

    if faces == sym:
      break

"""
Inefficiently compute dice (of size n) that may be complimentary to die.
"""
def compliment_space(die, n=6):
  space = combinations_with_replacement(range(1, 2 * n - max(die) + 1), n-1)
  for faces in space:
    yield (1, *faces)

"""
Check if the dice have the target distribution.
"""
def target_equals_dist(dice, target):
  distribution = np.sort(np.ravel(cartesian_sum(dice)))
  if np.all(distribution == target):
    return dice

"""
Grid search over the space of possible dice.
"""
def grid_search_dice(n, parallel=False):
  results = []

  target = np.sort(cartesian_sum(2*[range(1, n+1)]))
  die_1_space = die_space(n)

  def dice_target(space=die_1_space, t=target):
    for die_1 in space:
      for compliment in compliment_space(die_1, n):
        yield ((die_1, compliment), t)

  if parallel:
    with mp.Pool(mp.cpu_count()) as p:
      result = p.starmap(target_equals_dist, dice_target(), 100)

    for dice in result:
      if dice:
        results.append(dice)
  else:
    for dice in starmap(target_equals_dist, dice_target()):
      if dice:
        results.append(dice)

  return results


if __name__ == "__main__":
  print(grid_search_dice(6, True))
