// IEEE 754 binary64 arithmetic in software for Metal (which has no double type).
// Values travel as their bit patterns (ulong). All four IEEE rounding modes
// (rm: 0 nearest-even, 1 toward zero, 2 toward -inf, 3 toward +inf), subnormals,
// signed zeros, infinities and NaNs (results are the default quiet NaN).
// Algorithms follow Berkeley SoftFloat 3 (J. Hauser); division and square root
// use exact bit-serial integer algorithms.
#define D_SIGN 0x8000000000000000ul
#define D_INF 0x7FF0000000000000ul
#define D_QNAN 0x7FF8000000000000ul
#define D_FRAC 0x000FFFFFFFFFFFFFul

static inline ulong __dpack(uint s, int e, ulong sig) { return ((ulong)s << 63) + ((ulong)(uint)e << 52) + sig; }
static inline ulong __dsrjam(ulong a, uint d) {
  if (d == 0u) return a;
  return d < 63u ? ((a >> d) | (ulong)((a << (64u - d)) != 0ul)) : (ulong)(a != 0ul);
}
static inline ulong __dnorm_sub(ulong sig, thread int* e) {
  int sh = (int)clz(sig) - 11;
  *e = 1 - sh;
  return sig << (uint)sh;
}
// sig: leading 1 at bit 62, 10 rounding bits; e: biased exponent - 1
static inline ulong __dround_pack(uint s, int e, ulong sig, uint rm) {
  ulong inc = rm == 0u ? 0x200ul : ((rm == 2u && s != 0u) || (rm == 3u && s == 0u)) ? 0x3FFul : 0ul;
  uint rb = (uint)(sig & 0x3FFul);
  if ((uint)e >= 0x7FDu) {
    if (e < 0) { sig = __dsrjam(sig, (uint)(-e)); e = 0; rb = (uint)(sig & 0x3FFul); }
    else if (e > 0x7FD || sig + inc >= 0x8000000000000000ul) /* overflow: infinity, or the largest finite value */
      return (((ulong)s << 63) | D_INF) - (inc == 0ul ? 1ul : 0ul);
  }
  sig = (sig + inc) >> 10;
  if (rm == 0u && rb == 0x200u) sig &= ~1ul;
  if (sig == 0ul) e = 0;
  return __dpack(s, e, sig);
}
static inline ulong __dnorm_round_pack(uint s, int e, ulong sig, uint rm) {
  int sh = (int)clz(sig) - 1;
  e -= sh;
  if (sh >= 10 && (uint)e < 0x7FDu) return __dpack(s, sig != 0ul ? e : 0, sig << (uint)(sh - 10));
  return __dround_pack(s, e, sig << (uint)sh, rm);
}
static inline ulong __dzero(uint rm) { return rm == 2u ? D_SIGN : 0ul; } /* exact x - x */

static inline ulong __dadd_mags(ulong a, ulong b, uint s, uint rm) {
  int ea = (int)((a >> 52) & 0x7FFul), eb = (int)((b >> 52) & 0x7FFul);
  ulong sa = a & D_FRAC, sb = b & D_FRAC;
  int d = ea - eb, ez;
  ulong sz;
  if (d == 0) {
    if (ea == 0) return a + sb;
    if (ea == 0x7FF) return (sa | sb) != 0ul ? D_QNAN : a;
    ez = ea;
    sz = (0x0020000000000000ul + sa + sb) << 9;
  } else {
    sa <<= 9; sb <<= 9;
    if (d < 0) {
      if (eb == 0x7FF) return sb != 0ul ? D_QNAN : (((ulong)s << 63) | D_INF);
      ez = eb;
      if (ea != 0) sa += 0x2000000000000000ul; else sa <<= 1;
      sa = __dsrjam(sa, (uint)(-d));
    } else {
      if (ea == 0x7FF) return sa != 0ul ? D_QNAN : a;
      ez = ea;
      if (eb != 0) sb += 0x2000000000000000ul; else sb <<= 1;
      sb = __dsrjam(sb, (uint)d);
    }
    sz = 0x2000000000000000ul + sa + sb;
    if (sz < 0x4000000000000000ul) { ez--; sz <<= 1; }
  }
  return __dround_pack(s, ez, sz, rm);
}
static inline ulong __dsub_mags(ulong a, ulong b, uint s, uint rm) {
  int ea = (int)((a >> 52) & 0x7FFul), eb = (int)((b >> 52) & 0x7FFul);
  ulong sa = a & D_FRAC, sb = b & D_FRAC;
  int d = ea - eb;
  if (d == 0) {
    if (ea == 0x7FF) return D_QNAN;
    long sd = (long)sa - (long)sb;
    if (sd == 0l) return __dzero(rm);
    if (ea != 0) ea--;
    if (sd < 0l) { s ^= 1u; sd = -sd; }
    int sh = (int)clz((ulong)sd) - 11;
    int ez = ea - sh;
    if (ez < 0) { sh = ea; ez = 0; }
    return __dpack(s, ez, (ulong)sd << (uint)sh);
  }
  sa <<= 10; sb <<= 10;
  int ez;
  ulong sz;
  if (d < 0) {
    s ^= 1u;
    if (eb == 0x7FF) return sb != 0ul ? D_QNAN : (((ulong)s << 63) | D_INF);
    sa += ea != 0 ? 0x4000000000000000ul : sa;
    sa = __dsrjam(sa, (uint)(-d));
    sb |= 0x4000000000000000ul;
    ez = eb;
    sz = sb - sa;
  } else {
    if (ea == 0x7FF) return sa != 0ul ? D_QNAN : a;
    sb += eb != 0 ? 0x4000000000000000ul : sb;
    sb = __dsrjam(sb, (uint)d);
    sa |= 0x4000000000000000ul;
    ez = ea;
    sz = sa - sb;
  }
  return __dnorm_round_pack(s, ez - 1, sz, rm);
}
static inline ulong __dadd(ulong a, ulong b, uint rm) {
  uint sa = (uint)(a >> 63), sb = (uint)(b >> 63);
  return sa == sb ? __dadd_mags(a, b, sa, rm) : __dsub_mags(a, b, sa, rm);
}
static inline ulong __dsub(ulong a, ulong b, uint rm) { return __dadd(a, b ^ D_SIGN, rm); }

static inline ulong __dmul(ulong a, ulong b, uint rm) {
  uint s = (uint)((a ^ b) >> 63);
  int ea = (int)((a >> 52) & 0x7FFul), eb = (int)((b >> 52) & 0x7FFul);
  ulong sa = a & D_FRAC, sb = b & D_FRAC;
  if (ea == 0x7FF) {
    if (sa != 0ul || (eb == 0x7FF && sb != 0ul)) return D_QNAN;
    if (eb == 0 && sb == 0ul) return D_QNAN;
    return ((ulong)s << 63) | D_INF;
  }
  if (eb == 0x7FF) {
    if (sb != 0ul) return D_QNAN;
    if (ea == 0 && sa == 0ul) return D_QNAN;
    return ((ulong)s << 63) | D_INF;
  }
  if (ea == 0) { if (sa == 0ul) return (ulong)s << 63; sa = __dnorm_sub(sa, &ea); }
  if (eb == 0) { if (sb == 0ul) return (ulong)s << 63; sb = __dnorm_sub(sb, &eb); }
  int ez = ea + eb - 0x3FF;
  sa = (sa | 0x0010000000000000ul) << 10;
  sb = (sb | 0x0010000000000000ul) << 11;
  ulong hi = mulhi(sa, sb), lo = sa * sb;
  ulong sz = hi | (ulong)(lo != 0ul);
  if (sz < 0x4000000000000000ul) { ez--; sz <<= 1; }
  return __dround_pack(s, ez, sz, rm);
}

static inline ulong __ddiv(ulong a, ulong b, uint rm) {
  uint s = (uint)((a ^ b) >> 63);
  int ea = (int)((a >> 52) & 0x7FFul), eb = (int)((b >> 52) & 0x7FFul);
  ulong sa = a & D_FRAC, sb = b & D_FRAC;
  if (ea == 0x7FF) {
    if (sa != 0ul || eb == 0x7FF) return D_QNAN;
    return ((ulong)s << 63) | D_INF;
  }
  if (eb == 0x7FF) return sb != 0ul ? D_QNAN : ((ulong)s << 63);
  if (eb == 0) {
    if (sb == 0ul) return (ea == 0 && sa == 0ul) ? D_QNAN : (((ulong)s << 63) | D_INF);
    sb = __dnorm_sub(sb, &eb);
  }
  if (ea == 0) { if (sa == 0ul) return (ulong)s << 63; sa = __dnorm_sub(sa, &ea); }
  int ez = ea - eb + 0x3FE;
  sa |= 0x0010000000000000ul;
  sb |= 0x0010000000000000ul;
  if (sa < sb) { ez--; sa <<= 1; }
  ulong q = 0ul, rem = sa;
  for (uint i = 0u; i < 63u; i++) {
    q <<= 1;
    if (rem >= sb) { rem -= sb; q |= 1ul; }
    rem <<= 1;
  }
  return __dround_pack(s, ez, q | (ulong)(rem != 0ul), rm);
}

static inline ulong __dsqrt(ulong a, uint rm) {
  uint s = (uint)(a >> 63);
  int ea = (int)((a >> 52) & 0x7FFul);
  ulong sa = a & D_FRAC;
  if (ea == 0x7FF) return (sa != 0ul || s != 0u) ? D_QNAN : a;
  if (s != 0u) return (ea == 0 && sa == 0ul) ? a : D_QNAN;
  if (ea == 0) { if (sa == 0ul) return a; sa = __dnorm_sub(sa, &ea); }
  int e = ea - 1023;
  ulong m = sa | 0x0010000000000000ul;
  if ((e & 1) != 0) { m <<= 1; e -= 1; }
  // root = floor(sqrt(m << 60)): 57 bits, leading bit 56
  ulong root = 0ul, rem = 0ul;
  for (int i = 56; i >= 0; i--) {
    int j = 2 * i - 60;
    rem = (rem << 2) | (j >= 0 ? ((m >> (uint)j) & 3ul) : 0ul);
    ulong trial = (root << 2) | 1ul;
    root <<= 1;
    if (rem >= trial) { rem -= trial; root |= 1ul; }
  }
  return __dround_pack(0u, (e >> 1) + 0x3FE, (root << 6) | (ulong)(rem != 0ul), rm);
}

struct __u128 { ulong hi, lo; };
static inline __u128 __u128_make(ulong h, ulong l) { __u128 r; r.hi = h; r.lo = l; return r; }
static inline __u128 __u128_shl(__u128 x, uint d) {
  if (d == 0u) return x;
  if (d >= 128u) return __u128_make(0ul, 0ul);
  if (d >= 64u) return __u128_make(x.lo << (d - 64u), 0ul);
  return __u128_make((x.hi << d) | (x.lo >> (64u - d)), x.lo << d);
}
static inline __u128 __u128_shr_jam(__u128 x, uint d) {
  if (d == 0u) return x;
  if (d >= 128u) return __u128_make(0ul, (ulong)((x.hi | x.lo) != 0ul));
  ulong lost;
  __u128 r;
  if (d >= 64u) {
    lost = d == 64u ? x.lo : (x.lo | (x.hi << (128u - d)));
    r = __u128_make(0ul, d == 64u ? x.hi : (x.hi >> (d - 64u)));
  } else {
    lost = x.lo << (64u - d);
    r = __u128_make(x.hi >> d, (x.lo >> d) | (x.hi << (64u - d)));
  }
  r.lo |= (ulong)(lost != 0ul);
  return r;
}
static inline __u128 __u128_add(__u128 a, __u128 b) { ulong lo = a.lo + b.lo; return __u128_make(a.hi + b.hi + (ulong)(lo < a.lo), lo); }
static inline __u128 __u128_sub(__u128 a, __u128 b) { return __u128_make(a.hi - b.hi - (ulong)(a.lo < b.lo), a.lo - b.lo); }
static inline bool __u128_lt(__u128 a, __u128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
static inline int __u128_top(__u128 x) { return x.hi != 0ul ? 127 - (int)clz(x.hi) : 63 - (int)clz(x.lo); }

// fused multiply-add with a single rounding: exact 106-bit product, 128-bit alignment
static inline ulong __dfma(ulong a, ulong b, ulong c, uint rm) {
  uint sp = (uint)((a ^ b) >> 63), sc = (uint)(c >> 63);
  int ea = (int)((a >> 52) & 0x7FFul), eb = (int)((b >> 52) & 0x7FFul), ec = (int)((c >> 52) & 0x7FFul);
  ulong fa = a & D_FRAC, fb = b & D_FRAC, fc = c & D_FRAC;
  if ((ea == 0x7FF && fa != 0ul) || (eb == 0x7FF && fb != 0ul) || (ec == 0x7FF && fc != 0ul)) return D_QNAN;
  bool za = ea == 0 && fa == 0ul, zb = eb == 0 && fb == 0ul;
  if (ea == 0x7FF || eb == 0x7FF) {
    if (za || zb) return D_QNAN;
    if (ec == 0x7FF && sc != sp) return D_QNAN;
    return ((ulong)sp << 63) | D_INF;
  }
  if (ec == 0x7FF) return c;
  if (za || zb) return __dadd((ulong)sp << 63, c, rm);
  if (ec == 0 && fc == 0ul) return __dmul(a, b, rm);
  if (ea == 0) fa = __dnorm_sub(fa, &ea);
  if (eb == 0) fb = __dnorm_sub(fb, &eb);
  if (ec == 0) fc = __dnorm_sub(fc, &ec);
  fa |= 0x0010000000000000ul; fb |= 0x0010000000000000ul; fc |= 0x0010000000000000ul;
  __u128 P = __u128_make(mulhi(fa, fb), fa * fb);
  int lp = __u128_top(P);
  P = __u128_shl(P, (uint)(125 - lp));
  int ep = (ea - 1023) + (eb - 1023) + (lp - 104);
  int ecu = ec - 1023;
  __u128 C = __u128_shl(__u128_make(0ul, fc), 73u);
  __u128 X, Y;
  int ex;
  uint sx, sy;
  if (ep > ecu || (ep == ecu && !__u128_lt(P, C))) { X = P; sx = sp; ex = ep; Y = __u128_shr_jam(C, (uint)min(ep - ecu, 200)); sy = sc; }
  else { X = C; sx = sc; ex = ecu; Y = __u128_shr_jam(P, (uint)min(ecu - ep, 200)); sy = sp; }
  __u128 S = sx == sy ? __u128_add(X, Y) : __u128_sub(X, Y);
  if (S.hi == 0ul && S.lo == 0ul) return __dzero(rm);
  int ls = __u128_top(S);
  ulong sig = ls >= 62 ? __u128_shr_jam(S, (uint)(ls - 62)).lo : (S.lo << (uint)(62 - ls));
  return __dround_pack(sx, ex + ls - 125 + 0x3FE, sig, rm);
}

// --- comparisons, sign operations, classification ---
static inline bool __disnan(ulong a) { return (a & ~D_SIGN) > D_INF; }
// -1 less, 0 equal, 1 greater, 2 unordered
static inline int __dcmp(ulong a, ulong b) {
  if (__disnan(a) || __disnan(b)) return 2;
  long ka = (a & D_SIGN) != 0ul ? -(long)(a & ~D_SIGN) : (long)a;
  long kb = (b & D_SIGN) != 0ul ? -(long)(b & ~D_SIGN) : (long)b;
  return ka < kb ? -1 : (ka > kb ? 1 : 0);
}
static inline ulong __dtotal_key(ulong a) { return (a & D_SIGN) != 0ul ? ~a : (a | D_SIGN); }
static inline ulong __dmin(ulong a, ulong b) {
  if (__disnan(a)) return __disnan(b) ? D_QNAN : b;
  if (__disnan(b)) return a;
  return __dtotal_key(a) < __dtotal_key(b) ? a : b;
}
static inline ulong __dmax(ulong a, ulong b) {
  if (__disnan(a)) return __disnan(b) ? D_QNAN : b;
  if (__disnan(b)) return a;
  return __dtotal_key(a) > __dtotal_key(b) ? a : b;
}

// --- conversions ---
static inline ulong __i2d(long v, uint rm) {
  uint s = v < 0l ? 1u : 0u;
  ulong m = s != 0u ? (ulong)0 - (ulong)v : (ulong)v;
  if ((m & 0x7FFFFFFFFFFFFFFFul) == 0ul) return s != 0u ? __dpack(1u, 0x43E, 0ul) : 0ul;
  return __dnorm_round_pack(s, 0x43C, m, rm);
}
static inline ulong __u2d(ulong m, uint rm) {
  if (m == 0ul) return 0ul;
  if ((m & D_SIGN) != 0ul) return __dround_pack(0u, 0x43D, (m >> 1) | (m & 1ul), rm);
  return __dnorm_round_pack(0u, 0x43C, m, rm);
}
// binary64 -> binary32 with an explicit rounding mode
static inline float __d2f_r(ulong b, uint rm) {
  uint sign = (uint)(b >> 32) & 0x80000000u;
  int e = (int)((b >> 52) & 0x7FFul);
  ulong f = b & D_FRAC;
  if (e == 0x7FF) return as_type<float>(sign | 0x7F800000u | (f != 0ul ? 0x400000u : 0u));
  if (e == 0 && f == 0ul) return as_type<float>(sign);
  bool away = (rm == 2u && sign != 0u) || (rm == 3u && sign == 0u);   /* directed rounding away from zero */
  uint maxf = sign | 0x7F7FFFFFu, inf = sign | 0x7F800000u;
  if (e == 0) return as_type<float>(sign | (away ? 1u : 0u));        /* double subnormal: below float range */
  ulong m = f | 0x0010000000000000ul;
  int ue = e - 1023;
  if (ue > 127) return as_type<float>((rm == 0u || away) ? inf : maxf);
  uint sh = ue >= -126 ? 29u : (uint)(29 + (-126 - ue));
  if (sh > 63u) return as_type<float>(sign | (away ? 1u : 0u));
  ulong r = m >> sh, rem = m & ((1ul << sh) - 1ul), hv = 1ul << (sh - 1u);
  bool up = rm == 0u ? (rem > hv || (rem == hv && (r & 1ul) != 0ul)) : (away && rem != 0ul);
  if (up) r++;
  if (ue >= -126) {
    if (r == 0x1000000ul) { r >>= 1; ue++; if (ue > 127) return as_type<float>((rm == 0u || away) ? inf : maxf); }
    return as_type<float>(sign | ((uint)(ue + 127) << 23) | ((uint)r & 0x7FFFFFu));
  }
  return as_type<float>(sign | (uint)r);
}
// round to an integral double; mode 0 nearest-even, 1 toward zero, 2 down, 3 up
static inline ulong __dround(ulong x, uint mode) {
  uint s = (uint)(x >> 63);
  int e = (int)((x >> 52) & 0x7FFul) - 1023;
  if (e >= 52) return x;
  ulong mag = x & ~D_SIGN;
  if (mag == 0ul) return x;
  if (e < 0) {
    bool up = mode == 0u ? (e == -1 && mag > 0x3FE0000000000000ul) : mode == 2u ? s != 0u : mode == 3u ? s == 0u : false;
    return ((ulong)s << 63) | (up ? 0x3FF0000000000000ul : 0ul);
  }
  uint fb = (uint)(52 - e);
  ulong mask = (1ul << fb) - 1ul, r = mag & mask;
  if (r == 0ul) return x;
  ulong base = x & ~mask;
  bool up;
  if (mode == 0u) {
    ulong hv = 1ul << (fb - 1u);
    ulong odd = e == 0 ? 1ul : ((base >> fb) & 1ul);
    up = r > hv || (r == hv && odd != 0ul);
  } else up = mode == 2u ? s != 0u : mode == 3u ? s == 0u : false;
  return up ? base + (1ul << fb) : base;
}
// integral double -> integer, saturating (PTX cvt semantics; NaN -> 0)
static inline long __d2s64(ulong x) {
  if (__disnan(x)) return 0l;
  uint s = (uint)(x >> 63);
  int e = (int)((x >> 52) & 0x7FFul) - 1023;
  if (e < 0) return 0l;
  if (e >= 63) return s != 0u ? (long)0x8000000000000000ul : 0x7FFFFFFFFFFFFFFFl;
  ulong m = (x & D_FRAC) | 0x0010000000000000ul;
  ulong v = e >= 52 ? m << (uint)(e - 52) : m >> (uint)(52 - e);
  return s != 0u ? -(long)v : (long)v;
}
static inline ulong __d2u64(ulong x) {
  if (__disnan(x) || (x >> 63) != 0ul) return 0ul;
  int e = (int)((x >> 52) & 0x7FFul) - 1023;
  if (e < 0) return 0ul;
  if (e >= 64) return 0xFFFFFFFFFFFFFFFFul;
  ulong m = (x & D_FRAC) | 0x0010000000000000ul;
  return e >= 52 ? m << (uint)(e - 52) : m >> (uint)(52 - e);
}
