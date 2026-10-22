"""Shared-GPU admission only; no change to scientific mathematics or budgets."""
def admitted(free_mib,util_percent):
    return type(free_mib)is int and type(util_percent)is int and free_mib>=60000 and 0<=util_percent<=80
