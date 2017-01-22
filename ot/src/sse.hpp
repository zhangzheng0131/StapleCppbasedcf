/*******************************************************************************
NEON and SSE supported
by ihpdep
*******************************************************************************/
#ifndef _SSE_HPP_
#define _SSE_HPP_
#define __USE_SPEEDUP


#ifdef __USE_SPEEDUP
#define M128 __m128
#define M128i __m128i

#include <emmintrin.h> // SSE2:<e*.h>, SSE3:<p*.h>, SSE4:<s*.h>


#define RETf inline M128
#define RETi inline M128i

// set, load and store values
RETf SET( const float &x ) { return _mm_set1_ps(x); }
RETf SET( float x, float y, float z, float w ) { return _mm_set_ps(x,y,z,w); }
RETi SET( const int &x ) { return _mm_set1_epi32(x); }
RETf LD( const float &x ) { return _mm_load_ps(&x); }
RETf LDu( const float &x ) { return _mm_loadu_ps(&x); }
RETf STR( float &x, const M128 y ) { _mm_store_ps(&x,y); return y; }
RETf STR1( float &x, const M128 y ) { _mm_store_ss(&x,y); return y; }
RETf STRu( float &x, const M128 y ) { _mm_storeu_ps(&x,y); return y; }
RETf STR( float &x, const float y ) { return STR(x,SET(y)); }

// arithmetic operators
RETi ADD( const M128i x, const M128i y ) { return _mm_add_epi32(x,y); }
RETf ADD( const M128 x, const M128 y ) { return _mm_add_ps(x,y); }
RETf ADD( const M128 x, const M128 y, const M128 z ) {
  return ADD(ADD(x,y),z); }
RETf ADD( const M128 a, const M128 b, const M128 c, const M128 &d ) {
  return ADD(ADD(ADD(a,b),c),d); }
RETf SUB( const M128 x, const M128 y ) { return _mm_sub_ps(x,y); }
RETf MUL( const M128 x, const M128 y ) { return _mm_mul_ps(x,y); }
RETf MUL( const M128 x, const float y ) { return MUL(x,SET(y)); }
RETf MUL( const float x, const M128 y ) { return MUL(SET(x),y); }
RETf INC( M128 &x, const M128 y ) { return x = ADD(x,y); }
RETf INC( float &x, const M128 y ) { M128 t=ADD(LD(x),y); return STR(x,t); }
RETf DEC( M128 &x, const M128 y ) { return x = SUB(x,y); }
RETf DEC( float &x, const M128 y ) { M128 t=SUB(LD(x),y); return STR(x,t); }
RETf MIN( const M128 x, const M128 y ) { return _mm_min_ps(x,y); }
RETf RCP( const M128 x ) { return _mm_rcp_ps(x); }
RETf RCPSQRT( const M128 x ) { return _mm_rsqrt_ps(x); }

// logical operators
RETf AND( const M128 x, const M128 y ) { return _mm_and_ps(x,y); }
RETi AND( const M128i x, const M128i y ) { return _mm_and_si128(x,y); }
RETf ANDNOT( const M128 x, const M128 y ) { return _mm_andnot_ps(x,y); }
RETf OR( const M128 x, const M128 y ) { return _mm_or_ps(x,y); }
RETf XOR( const M128 x, const M128 y ) { return _mm_xor_ps(x,y); }

// comparison operators
RETf CMPGT( const M128 x, const M128 y ) { return _mm_cmpgt_ps(x,y); }
RETf CMPLT( const M128 x, const M128 y ) { return _mm_cmplt_ps(x,y); }
RETi CMPGT( const M128i x, const M128i y ) { return _mm_cmpgt_epi32(x,y); }
RETi CMPLT( const M128i x, const M128i y ) { return _mm_cmplt_epi32(x,y); }

// conversion operators
RETf CVT( const M128i x ) { return _mm_cvtepi32_ps(x); }
RETi CVT( const M128 x ) { return _mm_cvttps_epi32(x); }

#undef RETf
#undef RETi

#else

struct vec4
{
	union MyUnion
	{
		float data[4];
		unsigned datai[4];
	} _p;
	
	vec4() { for (int i = 0; i < 4;++i)	_p.datai[i] = 0; }
	vec4(float x){_p.data[0] = x; _p.data[1] = x; _p.data[2] = x; _p.data[3] = x;}
	vec4(unsigned x) { _p.datai[0] = x; _p.datai[1] = x; _p.datai[2] = x; _p.datai[3] = x; }
	vec4(float x, float y, float z, float w){_p.data[0] = x; _p.data[1] = y; _p.data[2] = z; _p.data[3] = w;}
	vec4(unsigned x, unsigned y, unsigned z, unsigned w) { _p.datai[0] = x; _p.datai[1] = y; _p.datai[2] = z; _p.datai[3] = w; }
	
};
struct vec4i
{
// 	int _data[4];
// 	vec4i() {}
// 	vec4i(int x){_data[0] = x; _data[1] = x; _data[2] = x; _data[3] = x;}
// 	vec4i(int x, int y, int z, int w){_data[0] = x; _data[1] = y; _data[2] = z; _data[3] = w;}
	union MyUnion2
	{
		unsigned data[4];
		float dataf[4];
	} _p;

	vec4i() { for (int i = 0; i < 4; ++i)	_p.data[i] = 0; }
	vec4i(int x) { _p.data[0] = x; _p.data[1] = x; _p.data[2] = x; _p.data[3] = x; }
	vec4i(int x, int y, int z, int w) { _p.data[0] = x; _p.data[1] = y; _p.data[2] = z; _p.data[3] = w; }
};


#define __VEC(a,r) a._p.data[r]
#define __VECi(a,r) a._p.datai[r]
#define _SV4(s) __VEC(a,0) s __VEC(b,0), __VEC(a,1) s __VEC(b,1), __VEC(a,2) s __VEC(b,2), __VEC(a,3) s __VEC(b,3)
#define _SV4i(s)  __VECi(a,0) s __VECi(b,0), __VECi(a,1) s __VECi(b,1), __VECi(a,2) s __VECi(b,2), __VECi(a,3) s __VECi(b,3)
#define _FV4(s) s(__VEC(a,0),__VEC(b,0)),s( __VEC(a,1), __VEC(b,1)), s( __VEC(a,2), __VEC(b,2)), s(__VEC(a,3), __VEC(b,3)) 
#define _CV4(s) (__VEC(a,0) s __VEC(b,0))?(unsigned)0xffffffff : (unsigned)0x0 , (__VEC(a,1) s __VEC(b,1))?(unsigned)0xffffffff : (unsigned)0x0,( __VEC(a,2) s __VEC(b,2))?(unsigned)0xffffffff : (unsigned)0x0,( __VEC(a,3) s __VEC(b,3))?(unsigned)0xffffffff : (unsigned)0x0
#define _CV4i(s) (__VECi(a,0) s __VECi(b,0))?0xffffffff : 0x0 , (__VECi(a,1) s __VECi(b,1))?0xffffffff : 0x0,( __VECi(a,2) s __VECi(b,2))?0xffffffff : 0x0,( __VECi(a,3) s __VECi(b,3))?0xffffffff : 0x0
template<typename T> inline
T _myadd(T a, T b) { return T(_SV4( + )); }

template<typename T> inline
T _mysub(T a, T b) { return T(_SV4( - )); }

template<typename T> inline
T _mymul(T a, T b) { return T(_SV4( * )); }

template<typename T> inline
T _mymin(T a, T b) { return T(_FV4( min )); }


// #undef _SV4
// #undef __VEC

#define M128 vec4
#define M128i vec4i

#define RETf inline M128
#define RETi inline M128i

// set, load and store values
RETf SET(const float &x) { return vec4(x); }//_mm_set1_ps(x)
RETf SET(float x, float y, float z, float w) { return vec4(x, y, z, w); }//_mm_set_ps(x,y,z,w)
RETi SET(const int &x) { return vec4i(x); }//_mm_set1_epi32(x);
RETf LD(const float &x) { return vec4((&x)[0], (&x)[1], (&x)[2], (&x)[3]); }//_mm_load_ps(&x);
RETf LDu(const float &x) { return vec4((&x)[0], (&x)[1], (&x)[2], (&x)[3]);}//_mm_loadu_ps(&x)
RETf STR(float &x, const M128 y) { float* p = &x; p[0] = __VEC(y, 0); p[1] = __VEC(y, 1); p[2] = __VEC(y, 2); p[3] = __VEC(y, 3); return y; } //_mm_store_ps(&x, y)
RETf STR1(float &x, const M128 y) { x = __VEC(y,0);  return y; }//_mm_store_ss(&x, y);
RETf STRu(float &x, const M128 y) { float* p = &x; p[0] = __VEC(y, 0); p[1] = __VEC(y, 1); p[2] = __VEC(y, 2); p[3] = __VEC(y, 3); return y; }//_mm_storeu_ps(&x, y);
RETf STR(float &x, const float y) { return STR(x, SET(y)); }

// arithmetic operators
RETi ADD(const M128i x, const M128i y) { return _myadd(x, y); }//_mm_add_epi32(x,y);
RETf ADD(const M128 x, const M128 y) { return _myadd(x, y); }//_mm_add_ps(x,y);
RETf ADD(const M128 x, const M128 y, const M128 z) {
	return ADD(ADD(x, y), z);
}
RETf ADD(const M128 a, const M128 b, const M128 c, const M128 &d) {
	return ADD(ADD(ADD(a, b), c), d);
}
RETf SUB(const M128 x, const M128 y) { return  _mysub(x, y);}//_mm_sub_ps(x,y);
RETf MUL(const M128 x, const M128 y) { return _mymul(x, y); }//_mm_mul_ps(x,y);
RETf MUL(const M128 x, const float y) { return MUL(x, SET(y)); }
RETf MUL(const float x, const M128 y) { return MUL(SET(x), y); }
RETf INC(M128 &x, const M128 y) { return x = ADD(x, y); }
RETf INC(float &x, const M128 y) { M128 t = ADD(LD(x), y); return STR(x, t); }
RETf DEC(M128 &x, const M128 y) { return x = SUB(x, y); }
RETf DEC(float &x, const M128 y) { M128 t = SUB(LD(x), y); return STR(x, t); }
RETf MIN(const M128 x, const M128 y) { return _mymin(x, y); }//_mm_min_ps(x,y); 
RETf RCP(const M128 a) { return vec4(1/ __VEC(a, 0), 1 / __VEC(a, 1), 1 / __VEC(a, 2), 1 / __VEC(a, 3));} // _mm_rcp_ps(x)
RETf RCPSQRT(const M128 a) { return vec4(1 /sqrtf(__VEC(a, 0)), 1 / sqrtf(__VEC(a, 1)), 1 / sqrtf(__VEC(a, 2)), 1 / sqrtf(__VEC(a, 3))); }//_mm_rsqrt_ps(x)

// logical operators
RETf AND(const M128 a, const M128 b) { return vec4(__VECi(a, 0) & __VECi(b, 0), __VECi(a, 1) & __VECi(b, 1), __VECi(a, 2) & __VECi(b, 2), __VECi(a, 3) & __VECi(b, 3)); }//_mm_and_ps(x, y)
RETi AND(const M128i a, const M128i b) { return vec4i(_SV4( & )); }//_mm_and_si128(x, y)
RETf ANDNOT(const M128 a, const M128 b) { return vec4(~__VECi(a, 0) & __VECi(b, 0), ~__VECi(a, 1) & __VECi(b, 1), ~__VECi(a, 2) & __VECi(b, 2), ~__VECi(a, 3) & __VECi(b, 3)); }//_mm_andnot_ps(x, y)
RETf OR(const M128 a, const M128 b) { return vec4(_SV4i( | )); }//_mm_or_ps(x, y);
RETf XOR(const M128 a, const M128 b) { return vec4(_SV4i( ^ ));}//_mm_xor_ps(x, y);

//comparison operators
RETf CMPGT(const M128 a, const M128 b) { return vec4( _CV4( > )); }//_mm_cmpgt_ps
RETf CMPLT(const M128 a, const M128 b) {	return  vec4(_CV4( < ));}//vec4(_CV4( < )); }//_mm_cmplt_ps(x, y);
RETi CMPGT(const M128i a, const M128i b) { return vec4i(_CV4( > )); }//_mm_cmpgt_epi32(x, y)
RETi CMPLT(const M128i a, const M128i b) { return vec4i(_CV4( < )); }//_mm_cmplt_epi32(x, y)

//conversion operators
RETf CVT(const M128i a) { return vec4((float)__VEC(a, 0), (float)__VEC(a, 1), (float)__VEC(a, 2), (float)__VEC(a, 3)); }//_mm_cvtepi32_ps(x)
RETi CVT(const M128 a) { return vec4i((int)__VEC(a, 0), (int)__VEC(a, 1), (int)__VEC(a, 2), (int)__VEC(a, 3)); }//_mm_cvttps_epi32(x)

#undef RETf
#undef RETi


#endif

#endif
