/* cds_pcg
 * Random number generator using the PCG algorithm.
 * Mostly copied verbatim from the reference Minimal C Implementation
 * [http://www.pcg-random.org/download.html#id1], repackaged as an STB-style
 * single-header-file library.
 *
 * Do this:
 *   #define CDS_PCG_IMPLEMENTATION
 * before including this file in *one* C/C++ file to provide the function
 * implementations.
 *
 * For a unit test on gcc/Clang:
 *   cc -Wall -pthread -std=c89 -D_POSIX_C_SOURCE=199309L -g -x c -DCDS_PCG_TEST -o test_cds_pcg.exe cds_pcg.h
 *
 * For a unit test on Visual C++:
 *   "%VS120COMNTOOLS%\..\..\VC\vcvarsall.bat"
 *   cl -W4 -nologo -TC -DCDS_PCG_TEST /Fetest_cds_pcg.exe cds_pcg.h
 * Debug-mode:
 *   cl -W4 -Od -Z7 -FC -MTd -nologo -TC -DCDS_PCG_TEST /Fetest_cds_pcg.exe cds_pcg.h
 */

#if defined(CDS_PCG_STATIC)
#   define CDS_PCG_DEF static
#else
#   define CDS_PCG_DEF extern
#endif


/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

#ifndef PCG_BASIC_H_INCLUDED
#define PCG_BASIC_H_INCLUDED 1

#include <stdint.h>

#if __cplusplus
extern "C" {
#endif

struct pcg_state_setseq_64 {    // Internals are *Private*.
    uint64_t state;             // RNG state.  All values are possible.
    uint64_t inc;               // Controls which RNG sequence (stream) is
                                // selected. Must *always* be odd.
};
typedef struct pcg_state_setseq_64 pcg32_random_t;

// If you *must* statically initialize it, here's one.

#define PCG32_INITIALIZER   { 0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL }

// pcg32_srandom(initstate, initseq)
// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

CDS_PCG_DEF
void pcg32_srandom(uint64_t initstate, uint64_t initseq);
CDS_PCG_DEF
void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate,
                     uint64_t initseq);

// pcg32_random()
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

CDS_PCG_DEF
uint32_t pcg32_random(void);
CDS_PCG_DEF
uint32_t pcg32_random_r(pcg32_random_t* rng);

// pcg32_boundedrand(bound):
// pcg32_boundedrand_r(rng, bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound

CDS_PCG_DEF
uint32_t pcg32_boundedrand(uint32_t bound);
CDS_PCG_DEF
uint32_t pcg32_boundedrand_r(pcg32_random_t* rng, uint32_t bound);


#if defined(CDS_PCG_RANDOMF01)
// pcg32_randomf01():
// pcg32_randomf01_r(rng):
//     Generate a random double in the range [0,1), rounded down
//     to the nearest 1/(2^32)
    
CDS_PCG_DEF
double pcg32_randomf01(void);
CDS_PCG_DEF
double pcg32_randomf01_r(pcg32_random_t* rng);
#endif // CDS_PCG_RANDOMF01
    
#if __cplusplus
}
#endif

#endif // PCG_BASIC_H_INCLUDED

/*************************************************************************/

#if defined(CDS_PCG_TEST)
#   if !defined(CDS_PCG_IMPLEMENTATION)
#       define CDS_PCG_IMPLEMENTATION
#   endif
#endif

#if defined(CDS_PCG_IMPLEMENTATION)

#ifdef _MSC_VER
#	pragma warning(push)
#	pragma warning(disable:4244) /* implicit uint64_t -> uint32_t trunctation */
#	pragma warning(disable:4146) /* unary minus operator applied to unsigned value */
#endif

/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

#if 0 // cds_pcg: not necessary
#include "pcg_basic.h" 
#endif

// state for global RNGs

static pcg32_random_t pcg32_global = PCG32_INITIALIZER;

// pcg32_srandom(initstate, initseq)
// pcg32_srandom_r(rng, initstate, initseq):
//     Seed the rng.  Specified in two parts, state initializer and a
//     sequence selection constant (a.k.a. stream id)

void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
{
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

void pcg32_srandom(uint64_t seed, uint64_t seq)
{
    pcg32_srandom_r(&pcg32_global, seed, seq);
}

// pcg32_random()
// pcg32_random_r(rng)
//     Generate a uniformly distributed 32-bit random number

uint32_t pcg32_random_r(pcg32_random_t* rng)
{
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint32_t pcg32_random()
{
    return pcg32_random_r(&pcg32_global);
}


// pcg32_boundedrand(bound):
// pcg32_boundedrand_r(rng, bound):
//     Generate a uniformly distributed number, r, where 0 <= r < bound

uint32_t pcg32_boundedrand_r(pcg32_random_t* rng, uint32_t bound)
{
    // To avoid bias, we need to make the range of the RNG a multiple of
    // bound, which we do by dropping output less than a threshold.
    // A naive scheme to calculate the threshold would be to do
    //
    //     uint32_t threshold = 0x100000000ull % bound;
    //
    // but 64-bit div/mod is slower than 32-bit div/mod (especially on
    // 32-bit platforms).  In essence, we do
    //
    //     uint32_t threshold = (0x100000000ull-bound) % bound;
    //
    // because this version will calculate the same modulus, but the LHS
    // value is less than 2^32.

    uint32_t threshold = -bound % bound;

    // Uniformity guarantees that this loop will terminate.  In practice, it
    // should usually terminate quickly; on average (assuming all bounds are
    // equally likely), 82.25% of the time, we can expect it to require just
    // one iteration.  In the worst case, someone passes a bound of 2^31 + 1
    // (i.e., 2147483649), which invalidates almost 50% of the range.  In 
    // practice, bounds are typically small and only a tiny amount of the range
    // is eliminated.
    for (;;) {
        uint32_t r = pcg32_random_r(rng);
        if (r >= threshold)
            return r % bound;
    }
}


uint32_t pcg32_boundedrand(uint32_t bound)
{
    return pcg32_boundedrand_r(&pcg32_global, bound);
}

#if defined(CDS_PCG_RANDOMF01)
#include <math.h>
double pcg32_randomf01_r(pcg32_random_t* rng)
{
    return ldexp((double)pcg32_random_r(rng), -32);
}

double pcg32_randomf01(void)
{
    return ldexp((double)pcg32_random_r(&pcg32_global), -32);
}
#endif // CDS_PCG_RANDOMF01



#ifdef _MSC_VER
#   pragma warning(pop)
#endif

#endif // CDS_PCG_IMPLEMENTATION

#if defined(CDS_PCG_TEST)
/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     http://www.pcg-random.org
 */

/*
 * This file was mechanically generated from tests/check-pcg32.c
 */

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#if 0 // cds_pcg: not necessary
#include "pcg_basic.h"
#endif

int main(int argc, char** argv)
{
    // Read command-line options

    int rounds = 5;
    bool nondeterministic_seed = false;
    int round, i;

    ++argv;
    --argc;
    if (argc > 0 && strcmp(argv[0], "-r") == 0) {
        nondeterministic_seed = true;
        ++argv;
        --argc;
    }
    if (argc > 0) {
        rounds = atoi(argv[0]);
    }

    // In this version of the code, we'll use a local rng, rather than the
    // global one.

    pcg32_random_t rng;

    // You should *always* seed the RNG.  The usual time to do it is the
    // point in time when you create RNG (typically at the beginning of the
    // program).
    //
    // pcg32_srandom_r takes two 64-bit constants (the initial state, and the
    // rng sequence selector; rngs with different sequence selectors will
    // *never* have random sequences that coincide, at all) - the code below
    // shows three possible ways to do so.

    if (nondeterministic_seed) {
        // Seed with external entropy -- the time and some program addresses
        // (which will actually be somewhat random on most modern systems).
        // A better solution, entropy_getbytes, using /dev/random, is provided
        // in the full library.

        pcg32_srandom_r(&rng, time(NULL) ^ (intptr_t)&printf, 
            (intptr_t)&rounds);
    } else {
        // Seed with a fixed constant

        pcg32_srandom_r(&rng, 42u, 54u);
    }

    printf("pcg32_random_r:\n"
           "      -  result:      32-bit unsigned int (uint32_t)\n"
           "      -  period:      2^64   (* 2^63 streams)\n"
           "      -  state type:  pcg32_random_t (%zu bytes)\n"
           "      -  output func: XSH-RR\n"
           "\n",
           sizeof(pcg32_random_t));

    for (round = 1; round <= rounds; ++round) {
        printf("Round %d:\n", round);
        /* Make some 32-bit numbers */
        printf("  32bit:");
        for (i = 0; i < 6; ++i)
            printf(" 0x%08x", pcg32_random_r(&rng));
        printf("\n");

        /* Toss some coins */
        printf("  Coins: ");
        for (i = 0; i < 65; ++i)
            printf("%c", pcg32_boundedrand_r(&rng, 2) ? 'H' : 'T');
        printf("\n");

        /* Roll some dice */
        printf("  Rolls:");
        for (i = 0; i < 33; ++i) {
            printf(" %d", (int)pcg32_boundedrand_r(&rng, 6) + 1);
        }
        printf("\n");

        /* Deal some cards */
        enum { SUITS = 4, NUMBERS = 13, CARDS = 52 };
        char cards[CARDS];

        for (i = 0; i < CARDS; ++i)
            cards[i] = i;

        for (i = CARDS; i > 1; --i) {
            int chosen = pcg32_boundedrand_r(&rng, i);
            char card = cards[chosen];
            cards[chosen] = cards[i - 1];
            cards[i - 1] = card;
        }

        printf("  Cards:");
        static const char number[] = {'A', '2', '3', '4', '5', '6', '7',
                                      '8', '9', 'T', 'J', 'Q', 'K'};
        static const char suit[] = {'h', 'c', 'd', 's'};
        for (i = 0; i < CARDS; ++i) {
            printf(" %c%c", number[cards[i] / SUITS], suit[cards[i] % SUITS]);
            if ((i + 1) % 22 == 0)
                printf("\n\t");
        }
        printf("\n");

        printf("\n");
    }

    return 0;
}
#endif // CDS_PCG_TEST
