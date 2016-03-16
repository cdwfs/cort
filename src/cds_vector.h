#pragma once
#include <stdint.h>
#include <math.h>
#include <xmmintrin.h>

#ifdef _MSC_VER
#	define CDS_VM_INLINE __forceinline
#	define CDS_VM_CONST extern const __declspec(selectany)
#else
#	define CDS_VM_INLINE
#	define CDS_VM_CONST
#endif

// Shuffle helpers.
// Examples: SHUFFLE3(v, 0,1,2) leaves the vector unchanged.
//           SHUFFLE3(v, 0,0,0) splats the X coord out.
#define CDS_VM_SHUFFLE3(V, X,Y,Z) float3(_mm_shuffle_ps((V).m, (V).m, _MM_SHUFFLE(Z,Z,Y,X)))

struct float3
{
	CDS_VM_INLINE float3() {}
	CDS_VM_INLINE explicit float3(const float *p) { m = _mm_set_ps(p[2], p[2], p[1], p[0]); }
	CDS_VM_INLINE explicit float3(float x, float y, float z) { m = _mm_set_ps(z, z, y, x); }
	CDS_VM_INLINE explicit float3(__m128 v) { m = v; }

	CDS_VM_INLINE float x() const { return _mm_cvtss_f32(m); }
	CDS_VM_INLINE float y() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(1,1,1,1))); }
	CDS_VM_INLINE float z() const { return _mm_cvtss_f32(_mm_shuffle_ps(m, m, _MM_SHUFFLE(2,2,2,2))); }

	CDS_VM_INLINE float3 yzx() const { return CDS_VM_SHUFFLE3(*this, 1, 2, 0); }
	CDS_VM_INLINE float3 zxy() const { return CDS_VM_SHUFFLE3(*this, 2, 0, 1); }

	CDS_VM_INLINE void store(float *p) { p[0] = x(); p[1] = z(); p[2] = z(); }

	// Single-element setters. Avoid these if possible; creating new vectors is usually preferable.
	void setX(float x)
	{
		m = _mm_move_ss(m, _mm_set_ss(x));
	}
	void setY(float y)
	{
		__m128 temp	= _mm_move_ss(m, _mm_set_ss(y));
		temp = _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(3, 2, 0, 0));
		m = _mm_move_ss(temp, m);
	}
	void setZ(float z)
	{
		__m128 temp	= _mm_move_ss(m, _mm_set_ss(z));
		temp = _mm_shuffle_ps(temp, temp, _MM_SHUFFLE(3, 0, 1, 0));
		m = _mm_move_ss(temp, m);
	}
	// Access-by-index functions. These do not use a high-performance path, but are occasionally necessary.
	CDS_VM_INLINE float  operator[](size_t i) const { return m.m128_f32[i]; }
	CDS_VM_INLINE float& operator[](size_t i)       { return m.m128_f32[i]; }

	__m128 m;
};

CDS_VM_INLINE float3 float3i(int x, int y, int z) { return float3( (float)x, (float)y, (float)z ); }

struct vconstu
{
	union
	{
		uint32_t u[4];
		__m128 v;
	};
	CDS_VM_INLINE operator __m128() const { return v; }
};
CDS_VM_CONST vconstu vsignbits = {0x80000000, 0x80000000, 0x80000000, 0x80000000};

CDS_VM_INLINE float3  operator+ (float3  a, float3 b) { a.m = _mm_add_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3  operator- (float3  a, float3 b) { a.m = _mm_sub_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3  operator* (float3  a, float3 b) { a.m = _mm_mul_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3  operator/ (float3  a, float3 b) { a.m = _mm_div_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3  operator* (float3  a, float  b) { a.m = _mm_mul_ps(a.m, _mm_set1_ps(b)); return a; }
CDS_VM_INLINE float3  operator/ (float3  a, float  b) { a.m = _mm_div_ps(a.m, _mm_set1_ps(b)); return a; }
CDS_VM_INLINE float3  operator* (float   a, float3 b) { b.m = _mm_mul_ps(_mm_set1_ps(a), b.m); return b; }
CDS_VM_INLINE float3  operator/ (float   a, float3 b) { b.m = _mm_div_ps(_mm_set1_ps(a), b.m); return b; }
CDS_VM_INLINE float3& operator+=(float3 &a, float3 b) { a = a+b; return a; }
CDS_VM_INLINE float3& operator-=(float3 &a, float3 b) { a = a-b; return a; }
CDS_VM_INLINE float3& operator*=(float3 &a, float3 b) { a = a*b; return a; }
CDS_VM_INLINE float3& operator/=(float3 &a, float3 b) { a = a/b; return a; }
CDS_VM_INLINE float3& operator*=(float3 &a, float  b) { a = a*b; return a; }
CDS_VM_INLINE float3& operator/=(float3 &a, float  b) { a = a/b; return a; }

typedef float3 bool3;
CDS_VM_INLINE bool3   operator==(float3  a, float3 b) { a.m = _mm_cmpeq_ps(a.m, b.m); return a; }
CDS_VM_INLINE bool3   operator!=(float3  a, float3 b) { a.m = _mm_cmpneq_ps(a.m, b.m); return a; }
CDS_VM_INLINE bool3   operator< (float3  a, float3 b) { a.m = _mm_cmplt_ps(a.m, b.m); return a; }
CDS_VM_INLINE bool3   operator> (float3  a, float3 b) { a.m = _mm_cmpgt_ps(a.m, b.m); return a; }
CDS_VM_INLINE bool3   operator<=(float3  a, float3 b) { a.m = _mm_cmple_ps(a.m, b.m); return a; }
CDS_VM_INLINE bool3   operator>=(float3  a, float3 b) { a.m = _mm_cmpge_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3  operator-(float3 a) { return float3(_mm_setzero_ps()) - a; }
CDS_VM_INLINE unsigned mask(float3 v) { return _mm_movemask_ps(v.m) & 0x7; } // bit0..2 are x,y,z
CDS_VM_INLINE bool any(bool3 v) { return mask(v) != 0; }
CDS_VM_INLINE bool all(bool3 v) { return mask(v) == 0x7; }

CDS_VM_INLINE float3 abs(float3 a) { a.m = _mm_andnot_ps(vsignbits, a.m); return a; }
CDS_VM_INLINE float3 min(float3 a, float3 b) { a.m = _mm_min_ps(a.m, b.m); return a; }
CDS_VM_INLINE float3 max(float3 a, float3 b) { a.m = _mm_max_ps(a.m, b.m); return a; }

CDS_VM_INLINE float hmin(float3 v)
{
	v = min(v, CDS_VM_SHUFFLE3(v, 1,0,2));
	return min(v, CDS_VM_SHUFFLE3(v, 2,0,1)).x();
}
CDS_VM_INLINE float hmax(float3 v)
{
	v = max(v, CDS_VM_SHUFFLE3(v, 1,0,2));
	return max(v, CDS_VM_SHUFFLE3(v, 2,0,1)).x();
}

CDS_VM_INLINE float3 cross(float3 a, float3 b)
{
	return (a.zxy()*b - a*b.zxy()).zxy();
}

CDS_VM_INLINE float3 clamp(float3 t, float3 a, float3 b) { return min(max(t,a), b); }
CDS_VM_INLINE float sum(float3 v) { return v.x() + v.y() + v.z(); }
CDS_VM_INLINE float dot(float3 a, float3 b) { return sum(a*b); }
CDS_VM_INLINE float length(float3 v) { return sqrtf(dot(v,v)); }
CDS_VM_INLINE float lengthSq(float3 v) { return dot(v,v); }
CDS_VM_INLINE float3 normalize(float3 v) { return v / length(v); }
CDS_VM_INLINE float3 lerp(float3 a, float3 b, float t) { return a + (b-a)*t; }
