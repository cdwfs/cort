#pragma once
#include <stdint.h>
#include <math.h>
#include <xmmintrin.h>

#ifdef _MSC_VER
#   define CDSF3_INLINE __forceinline
#   define CDSF3_VECTORCALL __vectorcall
#   define CDSF3_CONST extern const __declspec(selectany)
#else
#   define CDSF3_INLINE
#   define CDSF3_CONST
#endif

// Shuffle helpers.
// Examples: SHUFFLE3(v, 0,1,2) leaves the vector unchanged.
//           SHUFFLE3(v, 0,0,0) splats the X coord out.
#define CDSF3_SHUFFLE(V, X,Y,Z) float3(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(Z,Z,Y,X)))

struct float3
{
    CDSF3_INLINE float3() {}
    CDSF3_INLINE explicit float3(const float *p) { m = _mm_set_ps(p[2], p[2], p[1], p[0]); }
    CDSF3_INLINE explicit float3(float x, float y, float z) { m = _mm_set_ps(z, z, y, x); }
    CDSF3_INLINE explicit float3(__m128 v) { m = v; }

    CDSF3_INLINE float CDSF3_VECTORCALL x() const { return _mm_cvtss_f32(m); }
    CDSF3_INLINE float CDSF3_VECTORCALL y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(1,1,1,1))); }
    CDSF3_INLINE float CDSF3_VECTORCALL z() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(2,2,2,2))); }

    CDSF3_INLINE float3 CDSF3_VECTORCALL yzx() const { return CDSF3_SHUFFLE(*this, 1, 2, 0); }
    CDSF3_INLINE float3 CDSF3_VECTORCALL zxy() const { return CDSF3_SHUFFLE(*this, 2, 0, 1); }

    CDSF3_INLINE void CDSF3_VECTORCALL store(float *p) { p[0] = x(); p[1] = y(); p[2] = z(); }

    // Single-element setters. Avoid these if possible; creating new vectors is usually preferable.
    void CDSF3_VECTORCALL setX(float x)
    {
        m = _mm_move_ss(m, _mm_set_ss(x));
    }
    void CDSF3_VECTORCALL setY(float y)
    {
        __m128 temp	= _mm_move_ss(m, _mm_set_ss(y));
        temp = _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(3, 2, 0, 0));
        m = _mm_move_ss(temp, m);
    }
    void CDSF3_VECTORCALL setZ(float z)
    {
        __m128 temp	= _mm_move_ss(m, _mm_set_ss(z));
        temp = _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(3, 0, 1, 0));
        m = _mm_move_ss(temp, m);
    }
    // Access-by-index functions. These do not use a high-performance path, but are occasionally necessary.
    CDSF3_INLINE float  CDSF3_VECTORCALL operator[](size_t i) const { return m.m128_f32[i]; }
    CDSF3_INLINE float& CDSF3_VECTORCALL operator[](size_t i)       { return m.m128_f32[i]; }

    CDSF3_INLINE float CDSF3_VECTORCALL r() const     { return x(); }
    CDSF3_INLINE float CDSF3_VECTORCALL g() const     { return y(); }
    CDSF3_INLINE float CDSF3_VECTORCALL b() const     { return z(); }
    CDSF3_INLINE void  CDSF3_VECTORCALL setR(float r) { return setX(r); }
    CDSF3_INLINE void  CDSF3_VECTORCALL setG(float g) { return setY(g); }
    CDSF3_INLINE void  CDSF3_VECTORCALL setB(float b) { return setZ(b); }

    __m128 m;
};

CDSF3_INLINE float3 CDSF3_VECTORCALL float3i(int x, int y, int z) { return float3( (float)x, (float)y, (float)z ); }

struct float4_constu
{
    union
    {
        uint32_t u[4];
        __m128 v;
    };
    CDSF3_INLINE operator __m128() const { return v; }
};
CDSF3_CONST float4_constu vsignbits = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

CDSF3_INLINE float3  CDSF3_VECTORCALL operator+ (float3  a, float3 b) { a.m = _mm_add_ps(a.m, b.m); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator- (float3  a, float3 b) { a.m = _mm_sub_ps(a.m, b.m); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator* (float3  a, float3 b) { a.m = _mm_mul_ps(a.m, b.m); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator/ (float3  a, float3 b) { a.m = _mm_div_ps(a.m, b.m); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator* (float3  a, float  b) { a.m = _mm_mul_ps(a.m, _mm_set1_ps(b)); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator/ (float3  a, float  b) { a.m = _mm_div_ps(a.m, _mm_set1_ps(b)); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator* (float   a, float3 b) { b.m = _mm_mul_ps(_mm_set1_ps(a), b.m); return b; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator/ (float   a, float3 b) { b.m = _mm_div_ps(_mm_set1_ps(a), b.m); return b; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator+=(float3 &a, float3 b) { a = a+b; return a; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator-=(float3 &a, float3 b) { a = a-b; return a; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator*=(float3 &a, float3 b) { a = a*b; return a; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator/=(float3 &a, float3 b) { a = a/b; return a; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator*=(float3 &a, float  b) { a = a*b; return a; }
CDSF3_INLINE float3& CDSF3_VECTORCALL operator/=(float3 &a, float  b) { a = a/b; return a; }

typedef float3 bool3;
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator==(float3  a, float3 b) { a.m = _mm_cmpeq_ps(a.m, b.m); return a; }
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator!=(float3  a, float3 b) { a.m = _mm_cmpneq_ps(a.m, b.m); return a; }
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator< (float3  a, float3 b) { a.m = _mm_cmplt_ps(a.m, b.m); return a; }
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator> (float3  a, float3 b) { a.m = _mm_cmpgt_ps(a.m, b.m); return a; }
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator<=(float3  a, float3 b) { a.m = _mm_cmple_ps(a.m, b.m); return a; }
CDSF3_INLINE bool3   CDSF3_VECTORCALL operator>=(float3  a, float3 b) { a.m = _mm_cmpge_ps(a.m, b.m); return a; }
CDSF3_INLINE float3  CDSF3_VECTORCALL operator-(float3 a) { return float3(_mm_setzero_ps()) - a; }
CDSF3_INLINE unsigned CDSF3_VECTORCALL mask(float3 v) { return _mm_movemask_ps(v.m) & 0x7; } // bit0..2 are x,y,z
CDSF3_INLINE bool CDSF3_VECTORCALL any(bool3 v) { return mask(v) != 0; }
CDSF3_INLINE bool CDSF3_VECTORCALL all(bool3 v) { return mask(v) == 0x7; }

CDSF3_INLINE float3 CDSF3_VECTORCALL abs(float3 a) { a.m = _mm_andnot_ps(vsignbits, a.m); return a; }
CDSF3_INLINE float3 CDSF3_VECTORCALL vmin(float3 a, float3 b) { a.m = _mm_min_ps(a.m, b.m); return a; }
CDSF3_INLINE float3 CDSF3_VECTORCALL vmax(float3 a, float3 b) { a.m = _mm_max_ps(a.m, b.m); return a; }

CDSF3_INLINE float CDSF3_VECTORCALL hmin(float3 v)
{
    v = vmin(v, CDSF3_SHUFFLE(v, 1,0,2));
    return vmin(v, CDSF3_SHUFFLE(v, 2,0,1)).x();
}
CDSF3_INLINE float CDSF3_VECTORCALL hmax(float3 v)
{
    v = vmax(v, CDSF3_SHUFFLE(v, 1,0,2));
    return vmax(v, CDSF3_SHUFFLE(v, 2,0,1)).x();
}

CDSF3_INLINE float3 CDSF3_VECTORCALL cross(float3 a, float3 b)
{
    return (a.zxy()*b - a*b.zxy()).zxy();
}

CDSF3_INLINE float3 CDSF3_VECTORCALL clamp(float3 t, float3 a, float3 b) { return vmin(vmax(t,a), b); }
CDSF3_INLINE float  CDSF3_VECTORCALL sum(float3 v) { return v.x() + v.y() + v.z(); }
CDSF3_INLINE float  CDSF3_VECTORCALL dot(float3 a, float3 b) { return sum(a*b); }
CDSF3_INLINE float  CDSF3_VECTORCALL length(float3 v) { return sqrtf(dot(v,v)); }
CDSF3_INLINE float  CDSF3_VECTORCALL lengthSq(float3 v) { return dot(v,v); }
CDSF3_INLINE float3 CDSF3_VECTORCALL normalize(float3 v) { return v / length(v); }
CDSF3_INLINE float3 CDSF3_VECTORCALL lerp(float3 a, float3 b, float t) { return a + (b-a)*t; }
CDSF3_INLINE float3 CDSF3_VECTORCALL reflect(float3 v, float3 n) { return v - 2*dot(v,n)*n; }
