/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Aug 17, 2020
*******************************************************************/

// Fast linear array search for the index of element 'q' in the 'n'-element array pointed to by 'p'
//
// Data type      | Function             | Approx. speedup vs. typical linear search
// ---------------|----------------------|--------------------------------------------
// I8/U8          | FastSearch8          | 22x
// I16/U16        | FastSearch16         | 18x
// I32/U32        | FastSearch32         | 12x
// I64/U64        | FastSearch64         | 7x
// Float          | FastSearchFloat      | 13x
// Double         | FastSearchDouble     | 7x
//
// Returns index of the first (lowest-index) matching element, if a match is found.
// Returns ~0U (0xFFFFFFFF) if not found.
//
// Common function 'FastSearch' is also available. It uses the type of the array elements
// to dispatch the appropriate function.
//
// FOR EVEN FASTER PERFORMANCE: even faster versions of each function are available
// if the array can be over-allocated, such that the functions can safely read
// past the end of the arrays.
//
// Append _OverAlloc to the end of the above function names for these versions.
//
// Obviously, you MUST over-allocate the arrays to use these safely.
//
// Use the following helper function to query the required 'n' (number of elements)
// to allocate for your desired element type and desired 'n':
//   FastSearch*_OverAllocN(const U32 n)
//
// Common function 'FastSearch_OverAllocN' is also available. It uses a templated
// type to dispatch the appropriate function.
//
// For power users: technically you need only ensure that the reads
// do not access a page without read access, but over-allocation is the easiest
// way to achieve this.
//

#pragma once

#include <cstdint>
#include <immintrin.h>

using I8 = int8_t;
using I16 = int16_t;
using I32 = int32_t;
using I64 = int64_t;

using U8 = uint8_t;
using U16 = uint16_t;
using U32 = uint32_t;
using U64 = uint64_t;

#if defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#elif defined(__clang__) || defined(__GNUC__)
#define LIKELY(x) __builtin_expect((x), 1)
#define UNLIKELY(x) __builtin_expect((x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// versions that work without over-allocation (slower)

inline U32 FastSearchFloat(const void* const __restrict p, const U32 n, const float q)
{
    const __m256 vq = _mm256_set1_ps(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 31; iCoarse += 32)
    {
        const __m256 vA = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)p + iCoarse +  0), _CMP_EQ_OQ);
        const __m256 vB = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)p + iCoarse +  8), _CMP_EQ_OQ);
        const __m256 vC = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)p + iCoarse + 16), _CMP_EQ_OQ);
        const __m256 vD = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)p + iCoarse + 24), _CMP_EQ_OQ);
        const __m256 vM = _mm256_or_ps(_mm256_or_ps(vA, vB), _mm256_or_ps(vC, vD));
        if (UNLIKELY(!_mm256_testz_ps(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_ps(vA) <<  0;
            const U32 mB = (U32)_mm256_movemask_ps(vB) <<  8;
            const U32 mC = (U32)_mm256_movemask_ps(vC) << 16;
            const U32 mD = (U32)_mm256_movemask_ps(vD) << 24;
            const U32 m = (mA | mB) | (mC | mD);
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n) - 7; iCoarse += 8)
    {
        const __m256 vM = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)p + iCoarse), _CMP_EQ_OQ);
        const U32 m = (U32)_mm256_movemask_ps(vM);
        if (UNLIKELY(m))
        {
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const float*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearchDouble(const void* const __restrict p, const U32 n, const double q)
{
    const __m256d vq = _mm256_set1_pd(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 31; iCoarse += 32)
    {
        const __m256d vA = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse +  0), _CMP_EQ_OQ);
        const __m256d vB = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse +  4), _CMP_EQ_OQ);
        const __m256d vC = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse +  8), _CMP_EQ_OQ);
        const __m256d vD = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse + 12), _CMP_EQ_OQ);
        const __m256d vE = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse + 16), _CMP_EQ_OQ);
        const __m256d vF = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse + 20), _CMP_EQ_OQ);
        const __m256d vG = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse + 24), _CMP_EQ_OQ);
        const __m256d vH = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse + 28), _CMP_EQ_OQ);
        const __m256d vM = _mm256_or_pd(
            _mm256_or_pd(_mm256_or_pd(vA, vB), _mm256_or_pd(vC, vD)),
            _mm256_or_pd(_mm256_or_pd(vE, vF), _mm256_or_pd(vG, vH))
        );
        if (UNLIKELY(!_mm256_testz_pd(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_pd(vA) <<  0;
            const U32 mB = (U32)_mm256_movemask_pd(vB) <<  4;
            const U32 mC = (U32)_mm256_movemask_pd(vC) <<  8;
            const U32 mD = (U32)_mm256_movemask_pd(vD) << 12;
            const U32 mE = (U32)_mm256_movemask_pd(vE) << 16;
            const U32 mF = (U32)_mm256_movemask_pd(vF) << 20;
            const U32 mG = (U32)_mm256_movemask_pd(vG) << 24;
            const U32 mH = (U32)_mm256_movemask_pd(vH) << 28;
            const U32 m = ((mA | mB) | (mC | mD)) | ((mE | mF) | (mG | mH));
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n) - 3; iCoarse += 4)
    {
        const __m256d vM = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)p + iCoarse), _CMP_EQ_OQ);
        const U32 m = (U32)_mm256_movemask_pd(vM);
        if (UNLIKELY(m))
        {
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const double*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearch8(const void* const __restrict p, const U32 n, const U8 q)
{
    const __m256i vq = _mm256_set1_epi8(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 127; iCoarse += 128)
    {
        const __m256i vA = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 32)));
        const __m256i vC = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 64)));
        const __m256i vD = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 96)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U64 mA1 = (U64)(U32)_mm256_movemask_epi8(vA) <<  0;
            const U64 mA2 = (U64)(U32)_mm256_movemask_epi8(vB) << 32;
            const U64 mA = mA1 | mA2;
            const U64 mB1 = (U64)(U32)_mm256_movemask_epi8(vC) <<  0;
            const U64 mB2 = (U64)(U32)_mm256_movemask_epi8(vD) << 32;
            const U64 mB = mB1 | mB2;
            const U32 iFine = mA ? (U32)_tzcnt_u64(mA) : 64U + (U32)_tzcnt_u64(mB);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n) - 31; iCoarse += 32)
    {
        const __m256i vM = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse)));
        const U32 m = (U32)_mm256_movemask_epi8(vM);
        if (UNLIKELY(m))
        {
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const U8*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearch16(const void* const __restrict p, const U32 n, const U16 q)
{
    const __m256i vq = _mm256_set1_epi16(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 63; iCoarse += 64)
    {
        const __m256i vA = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((const U16*)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((const U16*)p + iCoarse + 16)));
        const __m256i vC = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((const U16*)p + iCoarse + 32)));
        const __m256i vD = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((const U16*)p + iCoarse + 48)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U64 mA1 = (U64)(U32)_mm256_movemask_epi8(vA) <<  0;
            const U64 mA2 = (U64)(U32)_mm256_movemask_epi8(vB) << 32;
            const U64 mA = mA1 | mA2;
            const U64 mB1 = (U64)(U32)_mm256_movemask_epi8(vC) <<  0;
            const U64 mB2 = (U64)(U32)_mm256_movemask_epi8(vD) << 32;
            const U64 mB = mB1 | mB2;
            const U64 iFine = mA ? (U32)_tzcnt_u64(mA) : 64U + (U32)_tzcnt_u64(mB);
            return U32(iCoarse + (iFine >> 1));
        }
    }

    for (; iCoarse < I64(n) - 15; iCoarse += 16)
    {
        const __m256i vM = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((const U16*)p + iCoarse)));
        const U32 m = (U32)_mm256_movemask_epi8(vM);
        if (UNLIKELY(m))
        {
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + (iFine >> 1));
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const U16*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearch32(const void* const __restrict p, const U32 n, const U32 q)
{
    const __m256i vq = _mm256_set1_epi32(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 31; iCoarse += 32)
    {
        const __m256i vA = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((const U32*)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((const U32*)p + iCoarse +  8)));
        const __m256i vC = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((const U32*)p + iCoarse + 16)));
        const __m256i vD = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((const U32*)p + iCoarse + 24)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vA)) <<  0;
            const U32 mB = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vB)) <<  8;
            const U32 mC = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vC)) << 16;
            const U32 mD = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vD)) << 24;
            const U32 m = (mA | mB) | (mC | mD);
            const U64 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n) - 7; iCoarse += 8)
    {
        const __m256i vM = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((const U32*)p + iCoarse)));
        const U32 m = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vM));
        if (UNLIKELY(m))
        {
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const U32*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearch64(const void* const __restrict p, const U32 n, const U64 q)
{
    const __m256i vq = _mm256_set1_epi64x(q);
    I64 iCoarse = 0;

    for (; iCoarse < I64(n) - 31; iCoarse += 32)
    {
        const __m256i vA = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse +  4)));
        const __m256i vC = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse +  8)));
        const __m256i vD = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 12)));
        const __m256i vE = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 16)));
        const __m256i vF = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 20)));
        const __m256i vG = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 24)));
        const __m256i vH = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 28)));
        const __m256i vM = _mm256_or_si256(
                _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD)),
                _mm256_or_si256(_mm256_or_si256(vE, vF), _mm256_or_si256(vG, vH))
        );
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vA)) <<  0;
            const U32 mB = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vB)) <<  4;
            const U32 mC = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vC)) <<  8;
            const U32 mD = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vD)) << 12;
            const U32 mE = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vE)) << 16;
            const U32 mF = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vF)) << 20;
            const U32 mG = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vG)) << 24;
            const U32 mH = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vH)) << 28;
            const U32 m = ((mA | mB) | (mC | mD)) | ((mE | mF) | (mG | mH));
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n) - 7; iCoarse += 8)
    {
        const __m256i vA = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 0)));
        const __m256i vB = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((const U64*)p + iCoarse + 4)));
        const __m256i vM = _mm256_or_si256(vA, vB);
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vA)) << 0;
            const U32 mB = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vB)) << 4;
            const U32 m = mA | mB;
            const U32 iFine = (U32)_tzcnt_u32(m);
            return U32(iCoarse + iFine);
        }
    }

    for (; iCoarse < I64(n); ++iCoarse)
    {
        if (UNLIKELY(*((const U64*)p + iCoarse) == q))
            return U32(iCoarse);
    }

    return ~0U;
}

inline U32 FastSearch(const I8* const __restrict p, const U32 n, const I8 q)
{
    return FastSearch8(p, n, q);
}

inline U32 FastSearch(const U8* const __restrict p, const U32 n, const U8 q)
{
    return FastSearch8(p, n, q);
}

inline U32 FastSearch(const I16* const __restrict p, const U32 n, const I16 q)
{
    return FastSearch16(p, n, q);
}

inline U32 FastSearch(const U16* const __restrict p, const U32 n, const U16 q)
{
    return FastSearch16(p, n, q);
}

inline U32 FastSearch(const I32* const __restrict p, const U32 n, const I32 q)
{
    return FastSearch32(p, n, q);
}

inline U32 FastSearch(const U32* const __restrict p, const U32 n, const U32 q)
{
    return FastSearch32(p, n, q);
}

inline U32 FastSearch(const I64* const __restrict p, const U32 n, const I64 q)
{
    return FastSearch64(p, n, q);
}

inline U32 FastSearch(const U64* const __restrict p, const U32 n, const U64 q)
{
    return FastSearch64(p, n, q);
}

inline U32 FastSearch(const float* const __restrict p, const U32 n, const float q)
{
    return FastSearchFloat(p, n, q);
}

inline U32 FastSearch(const double* const __restrict p, const U32 n, const double q)
{
    return FastSearchDouble(p, n, q);
}

// versions that require over-allocation (faster)

inline constexpr U32 FastSearchFloat_OverAllocN(const U32 n)
{
    return (n + 31U) & ~31U;
}

inline U32 FastSearchFloat_OverAlloc(const void* const __restrict p, const U32 n, const float q)
{
    const U64 vn = U64(n) << 2;
    const __m256 vq = _mm256_set1_ps(q);
    for (U64 iCoarse = 0; iCoarse < vn; iCoarse += 128)
    {
        const __m256 vA = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)((uintptr_t)p + iCoarse +  0)), _CMP_EQ_OQ);
        const __m256 vB = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)((uintptr_t)p + iCoarse + 32)), _CMP_EQ_OQ);
        const __m256 vC = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)((uintptr_t)p + iCoarse + 64)), _CMP_EQ_OQ);
        const __m256 vD = _mm256_cmp_ps(vq, _mm256_loadu_ps((const float*)((uintptr_t)p + iCoarse + 96)), _CMP_EQ_OQ);
        const __m256 vM = _mm256_or_ps(_mm256_or_ps(vA, vB), _mm256_or_ps(vC, vD));
        if (UNLIKELY(!_mm256_testz_ps(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_ps(vA) <<  0;
            const U32 mB = (U32)_mm256_movemask_ps(vB) <<  8;
            const U32 mC = (U32)_mm256_movemask_ps(vC) << 16;
            const U32 mD = (U32)_mm256_movemask_ps(vD) << 24;
            const U32 m = (mA | mB) | (mC | mD);
            const U64 iFine = (U32)_tzcnt_u32(m);
            const U64 res = iCoarse + (iFine << 2);
            return res < vn ? U32(res >> 2) : ~0U;
        }
    }
    return ~0U;
}

inline constexpr U32 FastSearchDouble_OverAllocN(const U32 n)
{
    return (n + 31U) & ~31U;
}

inline U32 FastSearchDouble_OverAlloc(const void* const __restrict p, const U32 n, const double q)
{
    const U64 vn = U64(n) << 3;
    const __m256d vq = _mm256_set1_pd(q);
    for (U64 iCoarse = 0; iCoarse < vn; iCoarse += 256)
    {
        const __m256d vA = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse +   0)), _CMP_EQ_OQ);
        const __m256d vB = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse +  32)), _CMP_EQ_OQ);
        const __m256d vC = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse +  64)), _CMP_EQ_OQ);
        const __m256d vD = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse +  96)), _CMP_EQ_OQ);
        const __m256d vE = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse + 128)), _CMP_EQ_OQ);
        const __m256d vF = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse + 160)), _CMP_EQ_OQ);
        const __m256d vG = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse + 192)), _CMP_EQ_OQ);
        const __m256d vH = _mm256_cmp_pd(vq, _mm256_loadu_pd((const double*)((uintptr_t)p + iCoarse + 224)), _CMP_EQ_OQ);
        const __m256d vM = _mm256_or_pd(
            _mm256_or_pd(_mm256_or_pd(vA, vB), _mm256_or_pd(vC, vD)),
            _mm256_or_pd(_mm256_or_pd(vE, vF), _mm256_or_pd(vG, vH))
        );
        if (UNLIKELY(!_mm256_testz_pd(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_pd(vA) <<  0;
            const U32 mB = (U32)_mm256_movemask_pd(vB) <<  4;
            const U32 mC = (U32)_mm256_movemask_pd(vC) <<  8;
            const U32 mD = (U32)_mm256_movemask_pd(vD) << 12;
            const U32 mE = (U32)_mm256_movemask_pd(vE) << 16;
            const U32 mF = (U32)_mm256_movemask_pd(vF) << 20;
            const U32 mG = (U32)_mm256_movemask_pd(vG) << 24;
            const U32 mH = (U32)_mm256_movemask_pd(vH) << 28;
            const U32 m = ((mA | mB) | (mC | mD)) | ((mE | mF) | (mG | mH));
            const U64 iFine = (U32)_tzcnt_u32(m);
            const U64 res = iCoarse + (iFine << 3);
            return res < vn ? U32(res >> 3) : ~0U;
        }
    }
    return ~0U;
}

inline constexpr U32 FastSearch8_OverAllocN(const U32 n)
{
    return (n + 127U) & ~127U;
}

inline U32 FastSearch8_OverAlloc(const void* const __restrict p, const U32 n, const U8 q)
{
    const __m256i vq = _mm256_set1_epi8(q);
    for (U32 iCoarse = 0; iCoarse < I64(n); iCoarse += 128)
    {
        const __m256i vA = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 32)));
        const __m256i vC = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 64)));
        const __m256i vD = _mm256_cmpeq_epi8(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 96)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U64 mA1 = (U64)(U32)_mm256_movemask_epi8(vA) <<  0;
            const U64 mA2 = (U64)(U32)_mm256_movemask_epi8(vB) << 32;
            const U64 mA = mA1 | mA2;
            const U64 mB1 = (U64)(U32)_mm256_movemask_epi8(vC) <<  0;
            const U64 mB2 = (U64)(U32)_mm256_movemask_epi8(vD) << 32;
            const U64 mB = mB1 | mB2;
            const U32 iFine = mA ? (U32)_tzcnt_u64(mA) : 64U + (U32)_tzcnt_u64(mB);
            const U32 res = iCoarse + iFine;
            return res < n ? res : ~0U;
        }
    }
    return ~0U;
}

inline constexpr U32 FastSearch16_OverAllocN(const U32 n)
{
    return (n + 63U) & ~63U;
}

inline U32 FastSearch16_OverAlloc(const void* const __restrict p, const U32 n, const U16 q)
{
    const U64 vn = U64(n) << 1;
    const __m256i vq = _mm256_set1_epi16(q);
    for (U64 iCoarse = 0; iCoarse < vn; iCoarse += 128)
    {
        const __m256i vA = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 32)));
        const __m256i vC = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 64)));
        const __m256i vD = _mm256_cmpeq_epi16(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 96)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U64 mA1 = (U64)(U32)_mm256_movemask_epi8(vA) <<  0;
            const U64 mA2 = (U64)(U32)_mm256_movemask_epi8(vB) << 32;
            const U64 mA = mA1 | mA2;
            const U64 mB1 = (U64)(U32)_mm256_movemask_epi8(vC) <<  0;
            const U64 mB2 = (U64)(U32)_mm256_movemask_epi8(vD) << 32;
            const U64 mB = mB1 | mB2;
            const U64 iFine = mA ? (U32)_tzcnt_u64(mA) : 64U + (U32)_tzcnt_u64(mB);
            const U64 res = iCoarse + iFine;
            return res < vn ? U32(res >> 1) : ~0U;
        }
    }
    return ~0U;
}

inline constexpr U32 FastSearch32_OverAllocN(const U32 n)
{
    return (n + 31U) & ~31U;
}

inline U32 FastSearch32_OverAlloc(const void* const __restrict p, const U32 n, const U32 q)
{
    const U64 vn = U64(n) << 2;
    const __m256i vq = _mm256_set1_epi32(q);
    for (U64 iCoarse = 0; iCoarse < vn; iCoarse += 128)
    {
        const __m256i vA = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  0)));
        const __m256i vB = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 32)));
        const __m256i vC = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 64)));
        const __m256i vD = _mm256_cmpeq_epi32(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 96)));
        const __m256i vM = _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD));
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vA)) <<  0;
            const U32 mB = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vB)) <<  8;
            const U32 mC = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vC)) << 16;
            const U32 mD = (U32)_mm256_movemask_ps(_mm256_castsi256_ps(vD)) << 24;
            const U32 m = (mA | mB) | (mC | mD);
            const U64 iFine = (U32)_tzcnt_u32(m);
            const U64 res = iCoarse + (iFine << 2);
            return res < vn ? U32(res >> 2) : ~0U;
        }
    }
    return ~0U;
}

inline constexpr U32 FastSearch64_OverAllocN(const U32 n)
{
    return (n + 31U) & ~31U;
}

inline U32 FastSearch64_OverAlloc(const void* const __restrict p, const U32 n, const U64 q)
{
    const U64 vn = U64(n) << 3;
    const __m256i vq = _mm256_set1_epi64x(q);
    for (U64 iCoarse = 0; iCoarse < vn; iCoarse += 256)
    {
        const __m256i vA = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +   0)));
        const __m256i vB = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  32)));
        const __m256i vC = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  64)));
        const __m256i vD = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse +  96)));
        const __m256i vE = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 128)));
        const __m256i vF = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 160)));
        const __m256i vG = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 192)));
        const __m256i vH = _mm256_cmpeq_epi64(vq, _mm256_loadu_si256((const __m256i*)((uintptr_t)p + iCoarse + 224)));
        const __m256i vM = _mm256_or_si256(
                _mm256_or_si256(_mm256_or_si256(vA, vB), _mm256_or_si256(vC, vD)),
                _mm256_or_si256(_mm256_or_si256(vE, vF), _mm256_or_si256(vG, vH))
        );
        if (UNLIKELY(!_mm256_testz_si256(vM, vM)))
        {
            const U32 mA = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vA)) <<  0;
            const U32 mB = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vB)) <<  4;
            const U32 mC = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vC)) <<  8;
            const U32 mD = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vD)) << 12;
            const U32 mE = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vE)) << 16;
            const U32 mF = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vF)) << 20;
            const U32 mG = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vG)) << 24;
            const U32 mH = (U32)_mm256_movemask_pd(_mm256_castsi256_pd(vH)) << 28;
            const U32 m = ((mA | mB) | (mC | mD)) | ((mE | mF) | (mG | mH));
            const U64 iFine = (U32)_tzcnt_u32(m);
            const U64 res = iCoarse + (iFine << 3);
            return res < vn ? U32(res >> 3) : ~0U;
        }
    }
    return ~0U;
}

inline U32 FastSearch_OverAlloc(const I8* const __restrict p, const U32 n, const I8 q)
{
    return FastSearch8_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const U8* const __restrict p, const U32 n, const U8 q)
{
    return FastSearch8_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const I16* const __restrict p, const U32 n, const I16 q)
{
    return FastSearch16_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const U16* const __restrict p, const U32 n, const U16 q)
{
    return FastSearch16_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const I32* const __restrict p, const U32 n, const I32 q)
{
    return FastSearch32_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const U32* const __restrict p, const U32 n, const U32 q)
{
    return FastSearch32_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const I64* const __restrict p, const U32 n, const I64 q)
{
    return FastSearch64_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const U64* const __restrict p, const U32 n, const U64 q)
{
    return FastSearch64_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const float* const __restrict p, const U32 n, const float q)
{
    return FastSearchFloat_OverAlloc(p, n, q);
}

inline U32 FastSearch_OverAlloc(const double* const __restrict p, const U32 n, const double q)
{
    return FastSearchDouble_OverAlloc(p, n, q);
}

template <class T>
inline constexpr U32 FastSearch_OverAllocN(const U32 n);

template <>
constexpr U32 FastSearch_OverAllocN<I8>(const U32 n)
{
    return FastSearch8_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<U8>(const U32 n)
{
    return FastSearch8_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<I16>(const U32 n)
{
    return FastSearch16_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<U16>(const U32 n)
{
    return FastSearch16_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<I32>(const U32 n)
{
    return FastSearch32_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<U32>(const U32 n)
{
    return FastSearch32_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<I64>(const U32 n)
{
    return FastSearch64_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<U64>(const U32 n)
{
    return FastSearch64_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<float>(const U32 n)
{
    return FastSearchFloat_OverAllocN(n);
}

template <>
constexpr U32 FastSearch_OverAllocN<double>(const U32 n)
{
    return FastSearchDouble_OverAllocN(n);
}
