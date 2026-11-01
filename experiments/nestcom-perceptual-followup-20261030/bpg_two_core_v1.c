/* Child-only adapter: expose the registered two-core allocation to x265.
 * Exact resource-only revision of the historical six-core adapter.
 * All other libc queries and codec binaries/options remain unchanged.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>
long sysconf(int name)
{
    if (name == _SC_NPROCESSORS_ONLN) {
        cpu_set_t set;
        if (sched_getaffinity(0, sizeof(set), &set) != 0 || CPU_COUNT(&set) != 2)
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
