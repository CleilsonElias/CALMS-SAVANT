"""Compare the patch with the old formulas on ordinary finite configurations."""
import itertools
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from calms_savant.lora import allocate_ranks, stratum_gate


def old_allocate(values, base_rank, budget, max_rank):
    values = dict(sorted(values.items()))
    desired = {k: max(1, min(max_rank, int(math.floor(base_rank/v)))) for k, v in values.items()}
    result = {k: 1 for k in values}
    while sum(result.values()) < budget:
        eligible = [k for k in values if result[k] < desired[k]]
        if not eligible:
            break
        key = max(eligible, key=lambda k: (desired[k]/result[k], -values[k], k))
        result[key] += 1
    return result


def main():
    grid = (1e-4, .05, .5, 1., 2., 10., 1e4)
    ranks = gates = 0
    for values, base, cap, budget in itertools.product(itertools.product(grid, repeat=2),
                                                       (1, 7, 16), (1, 8, 16), (2, 9, 32)):
        values = dict(zip(('a', 'b'), values))
        if allocate_ranks(values, base_rank=base, max_rank=cap, budget=budget) != old_allocate(values, base, budget, cap):
            raise AssertionError('ordinary rank allocation changed')
        ranks += 1
    for value, beta, optional in itertools.product(grid, (1., 2., 10.), (0., .3, 1.)):
        if stratum_gate(value, beta=beta, optional_gate=optional) != min(1., value**(-beta))*optional:
            raise AssertionError('ordinary gate changed')
        gates += 1
    print(f'passed: {ranks} rank configurations and {gates} gate configurations unchanged')


if __name__ == '__main__':
    main()
