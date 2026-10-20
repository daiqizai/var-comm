/*
 * Child-only resource adapter for the frozen bpgenc binary.
 * x265's legacy ThreadPool::getCpuCount calls sysconf(_SC_NPROCESSORS_ONLN)
 * and ignores sched affinity. Return the actual six-CPU allocation for that
 * one query. All other libc queries are delegated unchanged.
 * No codec math, presets, bitstream fields or binary bytes are modified.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>

long sysconf(int name)
{
    if (name == _SC_NPROCESSORS_ONLN) {
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof(set), &set) != 0 || CPU_COUNT(&set) != 6)
            _exit(88);
        return CPU_COUNT(&set);
    }
    static long (*real_sysconf)(int);
    if (!real_sysconf)
        real_sysconf = (long (*)(int))dlsym(RTLD_NEXT, "sysconf");
    if (!real_sysconf)
        _exit(89);
    return real_sysconf(name);
}
