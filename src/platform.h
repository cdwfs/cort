#pragma once
/* Collection of cross-platform functions and macros */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

#if   defined(ZOMBO_STATIC)
#   define ZOMBO_DEF static
#else
#   define ZOMBO_DEF extern
#endif

#if   defined(_MSC_VER)
#   define ZOMBO_PLATFORM_WINDOWS
#elif defined(__APPLE__) || defined(__MACH__)
#   include <mach/clock.h>
#   include <mach/mach.h>
#   define ZOMBO_PLATFORM_APPLE
#elif defined(unix) || defined(__unix__) || defined(__unix)
#   include <unistd.h>
#   if   defined(_POSIX_VERSION)
#       define ZOMBO_PLATFORM_POSIX
#   else
#       error Unsupported platform (non-POSIX Unix)
#   endif
#   include <time.h>
#else
#   error Unsupported platform
#endif

#if   defined(_MSC_VER)
#   define ZOMBO_COMPILER_MSVC
#elif defined(__clang__)
#   define ZOMBO_COMPILER_CLANG
#elif defined(__GNUC__)
#   define ZOMBO_COMPILER_GNU
#else
#   error Unsupported compiler
#endif

#ifdef __cplusplus
#   define ZOMBO_INLINE inline
#else
#   if defined(ZOMBO_COMPILER_MSVC)
#       define ZOMBO_INLINE __forceinline
#   else
#       define ZOMBO_INLINE inline
#   endif
#endif

// Platform-specific header files
#if   defined(ZOMBO_PLATFORM_WINDOWS)
#   include <windows.h>
#elif defined(ZOMBO_PLATFORM_POSIX) || defined(ZOMBO_PLATFORM_APPLE)
#   include <sys/types.h>
#   include <ctype.h>
#   include <pthread.h>
#   include <time.h>
#   include <unistd.h>
#endif

// ZOMBO_DEBUGBREAK()
#if   defined(ZOMBO_COMPILER_MSVC)
#   define ZOMBO_DEBUGBREAK() __debugbreak()
#elif defined(ZOMBO_COMPILER_GNU) || defined(ZOMBO_COMPILER_CLANG)
#   if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199409L
#       define ZOMBO_DEBUGBREAK() __asm__("int $3")
#   else
#       define ZOMBO_DEBUGBREAK() assert(0)
#   endif
#else
#   error Unsupported compiler
#endif

// Custom assert macro that prints a formatted error message and breaks immediately from the calling code
// - ZOMBO_ASSERT(cond,msg,...): if cond is not true, print msg and assert.
// - ZOMBO_ASSERT_RETURN(cond,retval,msg,...): if cond is not true, print msg and assert, then return retval (for release builds)
// - ZOMBO_ERROR(msg): unconditionally print msg and assert.
#if defined(NDEBUG)
#   define ZOMBO_ASSERT(cond,msg,...) do { (void)( 1 ? (void)0 : (void)(cond) ); } while(0,0)
#   define ZOMBO_ASSERT_RETURN(cond,retval,msg,...) do { if (!(cond)) { return (retval); } } while(0,0)
#elif defined(ZOMBO_COMPILER_MSVC)
#   define ZOMBO_ASSERT(cond,msg,...) \
        __pragma(warning(push)) \
        __pragma(warning(disable:4127)) \
        do { \
            if (!(cond)) { \
                char *buffer = (char*)malloc(1024); \
                _snprintf_s(buffer, 1024, 1023, msg ## "\n", __VA_ARGS__); \
                buffer[1023] = 0; \
                OutputDebugStringA(buffer); \
                free(buffer); \
                IsDebuggerPresent() ? __debugbreak() : assert(cond); \
            } \
        } while(0,0) \
        __pragma(warning(pop))
#   define ZOMBO_ASSERT_RETURN(cond,retval,msg,...) \
        __pragma(warning(push)) \
        __pragma(warning(disable:4127)) \
        do { \
            if (!(cond)) { \
                char *buffer = (char*)malloc(1024); \
                _snprintf_s(buffer, 1024, 1023, msg ## "\n", __VA_ARGS__); \
                buffer[1023] = 0; \
                OutputDebugStringA(buffer); \
                free(buffer); \
                IsDebuggerPresent() ? __debugbreak() : assert(cond); \
                return (retval); \
            } \
        } while(0,0) \
        __pragma(warning(pop))
#elif defined(ZOMBO_COMPILER_GNU) || defined(ZOMBO_COMPILER_CLANG)
#   define ZOMBO_ASSERT(cond,msg,...) \
        do { \
            if (!(cond)) { \
                printf(msg "\n", ## __VA_ARGS__); \
                fflush(stdout); \
                ZOMBO_DEBUGBREAK(); \
            } \
        } while(0,0)
#   define ZOMBO_ASSERT_RETURN(cond,retval,msg,...) \
        do { \
            if (!(cond)) { \
                printf(msg "\n", ## __VA_ARGS__); \
                fflush(stdout); \
                ZOMBO_DEBUGBREAK(); \
                return (retval); \
            } \
        } while(0,0)
#else
#   error Unsupported compiler
#endif
#define ZOMBO_ERROR(msg,...) ZOMBO_ASSERT(0, msg, ## __VA_ARGS__)
#define ZOMBO_ERROR_RETURN(retval,msg,...) ZOMBO_ASSERT_RETURN(0, retval, msg, ## __VA_ARGS__)

// ZOMBO_RETVAL_CHECK(expected, expr): if the result of evaluating expr does not equal expected, assert.
#if   defined(ZOMBO_COMPILER_MSVC)
#   define ZOMBO_RETVAL_CHECK(expected, expr) do { \
            int err = (expr); \
            if (err != (expected)) { \
                printf("%s(%d): error in %s() -- %s returned %d\n", __FILE__, __LINE__, __FUNCTION__, #expr, err); \
                ZOMBO_DEBUGBREAK(); \
            } \
            assert(err == (expected)); \
            __pragma(warning(push)) \
            __pragma(warning(disable:4127)) \
        } while(0) \
        __pragma(warning(pop))
#elif defined(ZOMBO_COMPILER_GNU) || defined(ZOMBO_COMPILER_CLANG)
#   define ZOMBO_RETVAL_CHECK(expected, expr) do { \
            int err = (expr); \
            if (err != (expected)) { \
                printf("%s(%d): error in %s() -- %s returned %d\n", __FILE__, __LINE__, __FUNCTION__, #expr, err); \
                ZOMBO_DEBUGBREAK(); \
            } \
            assert(err == (expected)); \
        } while(0)
#else
#   error Unsupported compiler
#endif

// popcnt
#if   defined(ZOMBO_COMPILER_MSVC)
#   include <intrin.h>
#   define ZOMBO_POPCNT32(x) __popcnt(x)
#   define ZOMBO_POPCNT64(x) __popcnt64(x)
#elif defined(ZOMBO_COMPILER_CLANG)
#   include <smmintrin.h>
#   define ZOMBO_POPCNT32(x) _mm_popcnt_u32(x)
#   define ZOMBO_POPCNT64(x) _mm_popcnt_u64(x)
#elif defined(ZOMBO_COMPILER_GNU)
// TODO
#endif



// zomboAtomic*()
ZOMBO_DEF ZOMBO_INLINE uint32_t zomboAtomicAdd(uint32_t *dest, int32_t val)
{
#if   defined(ZOMBO_COMPILER_MSVC)
    return InterlockedAdd((LONG*)dest, (LONG)val);
#elif defined(ZOMBO_COMPILER_GNU) || defined(ZOMBO_COMPILER_CLANG)
    return __sync_fetch_and_add(dest, val);
#else
#   error Unsupported compiler
#endif
}

// zomboCpuCount()
ZOMBO_DEF ZOMBO_INLINE int32_t zomboCpuCount(void)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    return sysInfo.dwNumberOfProcessors;
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    return sysconf(_SC_NPROCESSORS_ONLN);
#else
#   error Unsupported compiler
#endif
}

// zomboClockTicks()
ZOMBO_DEF ZOMBO_INLINE uint64_t zomboClockTicks(void)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    uint64_t outTicks;
    QueryPerformanceCounter((LARGE_INTEGER*)&outTicks);
    return outTicks;
#elif defined(ZOMBO_PLATFORM_APPLE)
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    return (uint64_t)mts.tv_nsec + (uint64_t)mts.tv_sec*1000000000ULL;
#elif defined(ZOMBO_PLATFORM_POSIX)
#   if defined(_POSIX_TIMERS) && (_POSIX_TIMERS > 0)
    struct timespec ts;
    clock_gettime(1, &ts);
    return (uint64_t)ts.tv_nsec + (uint64_t)ts.tv_sec*1000000000ULL;
#   else
#       error no timer here!
#   endif
#else
#   error Unsupported compiler
#endif
}

// zomboTicksToSeconds()
ZOMBO_DEF ZOMBO_INLINE double zomboTicksToSeconds(uint64_t ticks)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    LARGE_INTEGER qpcFreq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&qpcFreq);
    return (double)ticks / (double)qpcFreq.QuadPart;
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    return (double)ticks / 1e9;
#else
#   error Unsupported compiler
#endif
}

// zomboProcessId()
ZOMBO_DEF ZOMBO_INLINE int zomboProcessId(void)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    return GetCurrentProcessId();
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    return getpid();
#else
#   error Unsupported compiler
#endif
}

// zomboThreadId()
ZOMBO_DEF ZOMBO_INLINE int zomboThreadId(void)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    return GetCurrentThreadId();
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    return (int)(intptr_t)pthread_self();
#else
#   error Unsupported compiler
#endif
}

// zomboSleepMsec()
ZOMBO_DEF ZOMBO_INLINE void zomboSleepMsec(uint32_t msec)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    Sleep(msec);
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    struct timespec ts = {0, msec*1000};
    nanosleep(&ts, NULL);
#else
#   error Unsupported compiler
#endif
}

// zomboFopen()
ZOMBO_DEF ZOMBO_INLINE FILE *zomboFopen(const char *path, const char *mode)
{
#if   defined(ZOMBO_PLATFORM_WINDOWS)
    FILE *f = NULL;
    errno_t ferr = fopen_s(&f, path, mode);
    return (ferr == 0) ? f : NULL;
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
    return fopen(path, mode);
#endif
}

// zombo*nprintf()
#if   defined(ZOMBO_PLATFORM_WINDOWS)
#   define zomboSnprintf( str, size, fmt, ...)  _snprintf_s((str), (size), _TRUNCATE, (fmt), ## __VA_ARGS__)
#   define zomboVsnprintf(str, size, fmt, ap)	_vsnprintf_s((str), (size), _TRUNCATE, (fmt), (ap)
#   define zomboScanf(format, ...)				scanf_s((format), __VA_ARGS__)
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
#   define zomboSnprintf( str, size, fmt, ...)  snprintf((str), (size), (fmt), ## __VA_ARGS__)
#   define zomboVsnprintf(str, size, fmt, ap)	vsnprintf((str), (size), (fmt), (ap)
#   define zomboScanf(format, ...)				scanf((format), __VA_ARGS__)
#endif

// zomboStr*()
#if   defined(ZOMBO_PLATFORM_WINDOWS)
#   define zomboStrcasecmp(s1, s2)				_stricmp( (s1), (s2) )
#   define zomboStrncasecmp(s1, s2, n)			_strnicmp( (s1), (s2), (n) )
#   define zomboStrncpy(dest, src, n)			strncpy_s( (dest), (n), (src), (n) )
#elif defined(ZOMBO_PLATFORM_APPLE) || defined(ZOMBO_PLATFORM_POSIX)
#   define zomboStrcasecmp(s1, s2)				strcasecmp( (s1), (s2) )
#   define zomboStrncasecmp(s1, s2, n)			strncasecmp( (s1), (s2), (n) )
#   define zomboStrncpy(dest, src, n)			strncpy( (dest), (src), (n) )
#endif

#ifdef __cplusplus
}
#endif
