// kldrift macOS helpers: SoC energy counters (IOReport), chip name, and the
// process memory footprint. Exposed to Rust through a tiny C ABI.
//
// IOReport is a private framework (the same one `powermetrics`, asitop and
// macmon read). It needs no root, but its symbols are undocumented, so every
// call is defensive and failures just turn energy reporting off.

#import <Foundation/Foundation.h>
#import <CoreFoundation/CoreFoundation.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <string.h>

typedef struct IOReportSubscription *IOReportSubscriptionRef;

extern CFDictionaryRef IOReportCopyChannelsInGroup(CFStringRef group, CFStringRef subgroup,
                                                   uint64_t a, uint64_t b, uint64_t c);
extern IOReportSubscriptionRef IOReportCreateSubscription(void *a, CFMutableDictionaryRef desired,
                                                          CFMutableDictionaryRef *subscribed,
                                                          uint64_t channel_id, CFTypeRef b);
extern CFDictionaryRef IOReportCreateSamples(IOReportSubscriptionRef sub,
                                             CFMutableDictionaryRef subscribed, CFTypeRef a);
extern CFDictionaryRef IOReportCreateSamplesDelta(CFDictionaryRef prev, CFDictionaryRef cur,
                                                  CFTypeRef a);
extern int64_t IOReportSimpleGetIntegerValue(CFDictionaryRef channel, int32_t idx);
extern CFStringRef IOReportChannelGetChannelName(CFDictionaryRef channel);
extern CFStringRef IOReportChannelGetUnitLabel(CFDictionaryRef channel);

typedef struct {
    double cpu_mj;
    double gpu_mj;
    double ane_mj;
    double dram_mj;
} kld_energy;

static IOReportSubscriptionRef g_sub = NULL;
static CFMutableDictionaryRef g_subscribed = NULL;
static CFDictionaryRef g_begin = NULL;

int kld_power_init(void) {
    if (g_sub) return 0;
    CFDictionaryRef chans = IOReportCopyChannelsInGroup(CFSTR("Energy Model"), NULL, 0, 0, 0);
    if (!chans) return -1;
    CFMutableDictionaryRef desired =
        CFDictionaryCreateMutableCopy(kCFAllocatorDefault, CFDictionaryGetCount(chans), chans);
    CFRelease(chans);
    CFMutableDictionaryRef subscribed = NULL;
    g_sub = IOReportCreateSubscription(NULL, desired, &subscribed, 0, NULL);
    CFRelease(desired);
    if (!g_sub || !subscribed) {
        g_sub = NULL;
        return -2;
    }
    g_subscribed = subscribed;
    return 0;
}

int kld_power_begin(void) {
    if (!g_sub) return -1;
    if (g_begin) {
        CFRelease(g_begin);
        g_begin = NULL;
    }
    g_begin = IOReportCreateSamples(g_sub, g_subscribed, NULL);
    return g_begin ? 0 : -2;
}

static double to_millijoules(int64_t value, CFStringRef unit) {
    if (!unit) return (double)value;
    if (CFStringCompare(unit, CFSTR("nJ"), 0) == kCFCompareEqualTo) return value / 1e6;
    if (CFStringCompare(unit, CFSTR("uJ"), 0) == kCFCompareEqualTo) return value / 1e3;
    if (CFStringCompare(unit, CFSTR("mJ"), 0) == kCFCompareEqualTo) return (double)value;
    if (CFStringCompare(unit, CFSTR("J"), 0) == kCFCompareEqualTo) return value * 1e3;
    return (double)value;
}

int kld_power_end(kld_energy *out) {
    memset(out, 0, sizeof(*out));
    if (!g_sub || !g_begin) return -1;
    CFDictionaryRef end = IOReportCreateSamples(g_sub, g_subscribed, NULL);
    if (!end) return -2;
    CFDictionaryRef delta = IOReportCreateSamplesDelta(g_begin, end, NULL);
    CFRelease(end);
    CFRelease(g_begin);
    g_begin = NULL;
    if (!delta) return -3;

    CFArrayRef channels = CFDictionaryGetValue(delta, CFSTR("IOReportChannels"));
    if (channels && CFGetTypeID(channels) == CFArrayGetTypeID()) {
        for (CFIndex i = 0; i < CFArrayGetCount(channels); i++) {
            CFDictionaryRef ch = CFArrayGetValueAtIndex(channels, i);
            NSString *name = (__bridge NSString *)IOReportChannelGetChannelName(ch);
            if (!name) continue;
            double mj = to_millijoules(IOReportSimpleGetIntegerValue(ch, 0),
                                       IOReportChannelGetUnitLabel(ch));
            // Channel names differ a little between chip generations.
            if ([name isEqualToString:@"CPU Energy"]) {
                out->cpu_mj += mj;
            } else if ([name isEqualToString:@"GPU Energy"] || [name isEqualToString:@"GPU"]) {
                out->gpu_mj += mj;
            } else if ([name hasPrefix:@"ANE"]) {
                out->ane_mj += mj;
            } else if ([name hasPrefix:@"DRAM"]) {
                out->dram_mj += mj;
            }
        }
    }
    CFRelease(delta);
    return 0;
}

// Physical footprint of this process (what Activity Monitor shows as
// "Memory"); on Apple Silicon it includes Metal buffers in unified memory.
uint64_t kld_phys_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return info.phys_footprint;
}

// Writes the chip name (e.g. "Apple M3 Pro") into buf; returns its length.
int kld_chip_name(char *buf, int len) {
    size_t size = (size_t)len;
    if (sysctlbyname("machdep.cpu.brand_string", buf, &size, NULL, 0) != 0) return 0;
    return (int)strnlen(buf, (size_t)len);
}

// Unified memory size in bytes.
uint64_t kld_total_memory(void) {
    uint64_t mem = 0;
    size_t size = sizeof(mem);
    sysctlbyname("hw.memsize", &mem, &size, NULL, 0);
    return mem;
}
