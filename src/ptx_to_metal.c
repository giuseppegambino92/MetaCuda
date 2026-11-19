/*
 * MetaCuda — PTX → Metal Shading Language translator.
 *
 * Pipeline:   PTX text ─► tokens ─► module/function/instruction IR
 *                     ─► basic blocks + CFG ─► loop-aware reverse post-order
 *                     ─► MSL (one `kernel` per `.entry`)
 *
 * Control flow: PTX is unstructured (labels + predicated branches) and MSL has
 * no goto, so every function is emitted as a dispatch loop over basic blocks.
 * For kernels that use barriers or warp intrinsics the dispatcher is
 * "min-PC" scheduled: every SIMD-group executes the lowest-numbered pending
 * block first (blocks are numbered in loop-aware RPO), which reproduces SIMT
 * reconvergence at post-dominators — the property __syncthreads() and
 * __shfl_*_sync() rely on.
 *
 * Types: registers keep the storage class of their PTX declaration
 * (.b32 → uint, .f32 → float, …). Every instruction reads its operands *as its
 * own type* (bit-casts between same-sized int/float, value-casts between ints)
 * which is exactly PTX's "bit-size types are compatible with any type of the
 * same size" rule.
 *
 * f64: Metal has no double. f64 values are computed in fp32 but stored in
 * memory as genuine IEEE binary64 (converted on load/store), so data layouts
 * stay compatible. A warning is recorded when this happens.
 */
#include "ptx_to_metal.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ───────────────────────────── string builder ───────────────────────────── */

typedef struct { char *p; size_t n, cap; } sb_t;

static void sb_grow(sb_t *s, size_t add) {
    if (s->n + add + 1 > s->cap) {
        size_t c = s->cap ? s->cap * 2 : 8192;
        while (c < s->n + add + 1) c *= 2;
        s->p = (char *)realloc(s->p, c);
        s->cap = c;
    }
}
static void sb_puts(sb_t *s, const char *str) {
    size_t l = strlen(str);
    sb_grow(s, l);
    memcpy(s->p + s->n, str, l);
    s->n += l;
    s->p[s->n] = 0;
}
static void sb_printf(sb_t *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void sb_printf(sb_t *s, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[2048];
    int l = vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    if (l < 0) return;
    if ((size_t)l < sizeof tmp) { sb_puts(s, tmp); return; }
    sb_grow(s, (size_t)l);
    va_start(ap, fmt);
    vsnprintf(s->p + s->n, (size_t)l + 1, fmt, ap);
    va_end(ap);
    s->n += (size_t)l;
}

#define EXL 4096 /* max length of a single generated expression */

/* ───────────────────────────── value types ───────────────────────────── */

enum {
    VT_PRED = 0, VT_U8, VT_S8, VT_U16, VT_S16, VT_U32, VT_S32, VT_U64, VT_S64,
    VT_F16, VT_BF16, VT_F32, VT_F64, VT_COUNT
};
static const char *VT_C[VT_COUNT] = {"bool", "uchar", "char", "ushort", "short", "uint", "int",
                                     "ulong", "long", "half", "ushort", "float", "float"};
static const int VT_SIZE[VT_COUNT] = {1, 1, 1, 2, 2, 4, 4, 8, 8, 2, 2, 4, 8};

static int vt_isint(int v) { return v >= VT_U8 && v <= VT_S64; }
static int vt_issigned(int v) { return v == VT_S8 || v == VT_S16 || v == VT_S32 || v == VT_S64; }
/* float *storage* (C type is a floating type). BF16 is stored as ushort bits. */
static int vt_fltstore(int v) { return v == VT_F16 || v == VT_F32 || v == VT_F64; }
static int vt_isfloat(int v) { return vt_fltstore(v) || v == VT_BF16; }
static int vt_unsigned_of_size(int bytes) {
    switch (bytes) { case 1: return VT_U8; case 2: return VT_U16; case 4: return VT_U32; default: return VT_U64; }
}
static int vt_signed_of_size(int bytes) {
    switch (bytes) { case 1: return VT_S8; case 2: return VT_S16; case 4: return VT_S32; default: return VT_S64; }
}
static int vt_uns(int v) { return vt_isint(v) ? vt_unsigned_of_size(VT_SIZE[v]) : v; }

/* PTX type name ("f32", ".u64", ...) → VT, or -1. *is_x2 set for f16x2/bf16x2. */
static int vt_parse(const char *s, int *is_x2) {
    if (*s == '.') s++;
    if (is_x2) *is_x2 = 0;
    static const struct { const char *n; int v; } T[] = {
        {"pred", VT_PRED}, {"b8", VT_U8},   {"u8", VT_U8},   {"s8", VT_S8},   {"b16", VT_U16},
        {"u16", VT_U16},   {"s16", VT_S16}, {"f16", VT_F16}, {"bf16", VT_BF16}, {"b32", VT_U32},
        {"u32", VT_U32},   {"s32", VT_S32}, {"f32", VT_F32}, {"tf32", VT_U32}, {"b64", VT_U64},
        {"u64", VT_U64},   {"s64", VT_S64}, {"f64", VT_F64}};
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (!strcmp(s, T[i].n)) return T[i].v;
    if (!strcmp(s, "f16x2")) { if (is_x2) *is_x2 = 1; return VT_U32; }
    if (!strcmp(s, "bf16x2")) { if (is_x2) *is_x2 = 2; return VT_U32; }
    return -1;
}

/* Storage VT of a register declared with PTX type `v`. */
static int vt_reg_storage(int v) {
    if (v == VT_PRED || v == VT_F16 || v == VT_F32 || v == VT_F64) return v;
    if (v == VT_BF16) return VT_U16;
    return vt_unsigned_of_size(VT_SIZE[v]);
}

/* Cast expression `e` of type `from` into type `to` (PTX register semantics). */
static void castx(char *out, size_t n, const char *e, int from, int to) {
    if (from == to || (from == VT_BF16 && to == VT_U16) || (from == VT_U16 && to == VT_BF16)) {
        snprintf(out, n, "%s", e);
        return;
    }
    if (to == VT_PRED) { snprintf(out, n, "((%s) != 0)", e); return; }
    if (from == VT_PRED) { snprintf(out, n, "((%s)(%s))", VT_C[to], e); return; }
    int fi = vt_fltstore(from), ti = vt_fltstore(to);
    if (fi == ti) { snprintf(out, n, "((%s)(%s))", VT_C[to], e); return; }
    char t[EXL];
    if (fi) { /* float storage → integer storage: bit-cast */
        int u = vt_unsigned_of_size(VT_SIZE[from]);
        if (from == VT_F64) snprintf(t, sizeof t, "__d_bits(%s)", e);
        else snprintf(t, sizeof t, "as_type<%s>(%s)", VT_C[u], e);
        if (u == to) snprintf(out, n, "%s", t);
        else snprintf(out, n, "((%s)(%s))", VT_C[to], t);
    } else { /* integer → float storage: bit-cast */
        int u = vt_unsigned_of_size(VT_SIZE[to]);
        if (u == from) snprintf(t, sizeof t, "%s", e);
        else snprintf(t, sizeof t, "((%s)(%s))", VT_C[u], e);
        if (to == VT_F64) snprintf(out, n, "__d_from_bits(%s)", t);
        else snprintf(out, n, "as_type<%s>(%s)", VT_C[to], t);
    }
}

/* ───────────────────────────── tokenizer ───────────────────────────── */

enum { TK_EOF, TK_ID, TK_NUM, TK_STR, TK_P };
typedef struct { int k; const char *s; int len; int line; } tok_t;

static int is_id0(int c) { return isalpha(c) || c == '_' || c == '$' || c == '%' || c == '.'; }
static int is_id1(int c) { return isalnum(c) || c == '_' || c == '$' || c == '.'; }

static tok_t *tokenize(const char *s, size_t n, int *count) {
    size_t cap = 4096, nt = 0;
    tok_t *t = (tok_t *)malloc(cap * sizeof *t);
    int line = 1;
    size_t i = 0;
    while (i < n && s[i]) {
        char c = s[i];
        if (c == '\n') { line++; i++; continue; }
        if (isspace((unsigned char)c)) { i++; continue; }
        if (c == '/' && i + 1 < n && s[i + 1] == '/') { while (i < n && s[i] && s[i] != '\n') i++; continue; }
        if (c == '/' && i + 1 < n && s[i + 1] == '*') {
            i += 2;
            while (i + 1 < n && !(s[i] == '*' && s[i + 1] == '/')) { if (s[i] == '\n') line++; i++; }
            i += 2;
            continue;
        }
        if (nt + 1 >= cap) { cap *= 2; t = (tok_t *)realloc(t, cap * sizeof *t); }
        tok_t *k = &t[nt++];
        k->s = s + i;
        k->line = line;
        size_t j = i + 1;
        if (c == '"') {
            while (j < n && s[j] && s[j] != '"') { if (s[j] == '\\') j++; j++; }
            j++;
            k->k = TK_STR;
        } else if (isdigit((unsigned char)c)) {
            while (j < n && (isalnum((unsigned char)s[j]) || s[j] == '.')) j++;
            k->k = TK_NUM;
        } else if (is_id0((unsigned char)c)) {
            for (;;) {
                if (j < n && is_id1((unsigned char)s[j])) j++;
                else if (j + 1 < n && s[j] == ':' && s[j + 1] == ':') j += 2;
                else break;
            }
            k->k = TK_ID;
        } else {
            k->k = TK_P;
        }
        k->len = (int)(j - i);
        i = j;
    }
    t[nt].k = TK_EOF;
    t[nt].s = "";
    t[nt].len = 0;
    t[nt].line = line;
    *count = (int)nt;
    return t;
}

static int teq(const tok_t *t, const char *s) { return (int)strlen(s) == t->len && !memcmp(t->s, s, (size_t)t->len); }
static void tcpy(char *d, size_t n, const tok_t *t) {
    size_t l = (size_t)t->len < n - 1 ? (size_t)t->len : n - 1;
    memcpy(d, t->s, l);
    d[l] = 0;
}

/* Parse a PTX numeric literal. kind: 0 int, 1 f32 bits, 2 f64 bits. */
static int parse_num(const char *s, int *kind, uint64_t *bits) {
    if ((s[0] == '0') && (s[1] == 'f' || s[1] == 'F') && isxdigit((unsigned char)s[2])) {
        *kind = 1; *bits = strtoull(s + 2, NULL, 16); return 0;
    }
    if ((s[0] == '0') && (s[1] == 'd' || s[1] == 'D') && isxdigit((unsigned char)s[2])) {
        *kind = 2; *bits = strtoull(s + 2, NULL, 16); return 0;
    }
    if (strchr(s, '.') || ((strchr(s, 'e') || strchr(s, 'E')) && !(s[0] == '0' && (s[1] == 'x' || s[1] == 'X')))) {
        double d = strtod(s, NULL);
        *kind = 2; memcpy(bits, &d, 8); return 0;
    }
    *kind = 0;
    if (s[0] == '0' && (s[1] == 'b' || s[1] == 'B')) *bits = strtoull(s + 2, NULL, 2);
    else *bits = strtoull(s, NULL, 0);
    return 0;
}

/* ───────────────────────────── IR ───────────────────────────── */

enum { OP_REG, OP_IMM, OP_ADDR, OP_VEC, OP_SYM, OP_SINK };
enum { SP_NONE, SP_GLOBAL, SP_CONST, SP_SHARED, SP_LOCAL, SP_PARAM, SP_GENERIC };

typedef struct operand {
    int kind;
    char name[MC_MAX_NAME];  /* register / symbol / address base          */
    char name2[MC_MAX_NAME]; /* second destination of "%r|%p"             */
    int not_;                /* "!%p"                                      */
    int imm_kind;            /* 0 int, 1 f32 bits, 2 f64 bits              */
    uint64_t bits;
    int64_t off;             /* address / symbol offset                   */
    int has_base;            /* OP_ADDR: name[] valid                     */
    struct operand *elems;   /* OP_VEC                                    */
    int nelems;
} operand_t;

#define MAX_OPS 8
#define MAX_PARTS 16
typedef struct {
    char pred[MC_MAX_NAME];
    int pred_not;
    char opcode[MC_MAX_NAME];
    char partbuf[MC_MAX_NAME];
    char *parts[MAX_PARTS];
    int np;
    operand_t ops[MAX_OPS];
    int nops;
    int line;
} instr_t;

typedef struct {
    int is_label;
    char label[MC_MAX_NAME];
    instr_t ins;
} stmt_t;

typedef struct {
    char name[MC_MAX_NAME];
    int space;
    uint32_t align;
    uint64_t size;
    uint64_t offset;
    int is_extern; /* extern .shared (dynamic) */
    int elem_vt;
} var_t;

typedef struct {
    char prefix[MC_MAX_NAME];
    int count;  /* 0 → plain (non-parameterized) register */
    int vt;     /* declared PTX type */
} regdecl_t;

typedef struct {
    char name[MC_MAX_NAME];
    int is_entry;
    int is_proto;
    mc_param_t *params;
    int nparams;
    uint32_t param_bytes;
    regdecl_t *regs;
    int nregs, capregs;
    var_t *vars;
    int nvars, capvars;
    stmt_t *stmts;
    int nstmts, capstmts;
    uint32_t reqntid[3];
} func_t;

typedef struct {
    tok_t *t;
    int nt, i;
    char err[1024];
    sb_t warn;
    var_t *mvars; /* module scope variables */
    int nmvars, capmvars;
    func_t *funcs;
    int nfuncs, capfuncs;
    uint8_t *ginit;
    uint64_t gbytes;
    int warned_f64, warned_printf, warned_atom64, warned_namedbar;
} P;

#define PERR(p, ...) do { if (!(p)->err[0]) snprintf((p)->err, sizeof (p)->err, __VA_ARGS__); } while (0)

static tok_t *pk(P *p) { return &p->t[p->i]; }
static tok_t *pk1(P *p) { return &p->t[p->i + 1 <= p->nt ? p->i + 1 : p->nt]; }
static int at_eof(P *p) { return p->t[p->i].k == TK_EOF; }
static void adv(P *p) { if (!at_eof(p)) p->i++; }
static int accept(P *p, const char *s) { if (teq(pk(p), s)) { adv(p); return 1; } return 0; }
static int expect(P *p, const char *s) {
    if (accept(p, s)) return 1;
    char got[64];
    tcpy(got, sizeof got, pk(p));
    PERR(p, "PTX parse error (line %d): expected '%s' but found '%s'", pk(p)->line, s, got);
    return 0;
}
static void skip_line(P *p) {
    int l = pk(p)->line;
    while (!at_eof(p) && pk(p)->line == l) adv(p);
}
static void skip_until_semicolon(P *p) {
    while (!at_eof(p) && !teq(pk(p), ";")) adv(p);
    accept(p, ";");
}
static void warn(P *p, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char tmp[512];
    vsnprintf(tmp, sizeof tmp, fmt, ap);
    va_end(ap);
    sb_puts(&p->warn, tmp);
    sb_puts(&p->warn, "\n");
}

static void split_parts(instr_t *in) {
    snprintf(in->partbuf, sizeof in->partbuf, "%s", in->opcode);
    in->np = 0;
    char *s = in->partbuf;
    while (*s && in->np < MAX_PARTS) {
        in->parts[in->np++] = s;
        /* split on '.' but not inside "::" qualifiers (e.g. "shared::cta") */
        while (*s && *s != '.') s++;
        if (*s == '.') *s++ = 0;
    }
}

/* ───────────────────────────── parser ───────────────────────────── */

static int parse_int_tok(P *p, int64_t *v) {
    int neg = 0;
    if (accept(p, "-")) neg = 1;
    if (pk(p)->k != TK_NUM) { PERR(p, "PTX parse error (line %d): expected number", pk(p)->line); return 0; }
    char b[64];
    tcpy(b, sizeof b, pk(p));
    int kind;
    uint64_t bits;
    parse_num(b, &kind, &bits);
    adv(p);
    *v = neg ? -(int64_t)bits : (int64_t)bits;
    return 1;
}

static int parse_operand(P *p, operand_t *op);

static int parse_operand(P *p, operand_t *op) {
    memset(op, 0, sizeof *op);
    tok_t *t = pk(p);
    if (teq(t, "[")) {
        adv(p);
        op->kind = OP_ADDR;
        if (pk(p)->k == TK_ID) {
            tcpy(op->name, sizeof op->name, pk(p));
            op->has_base = 1;
            adv(p);
        } else if (pk(p)->k == TK_NUM) {
            int64_t v;
            if (!parse_int_tok(p, &v)) return 0;
            op->off = v;
        }
        while (teq(pk(p), "+") || teq(pk(p), "-")) {
            int minus = teq(pk(p), "-");
            adv(p);
            int64_t v;
            if (!parse_int_tok(p, &v)) return 0;
            op->off += minus ? -v : v;
        }
        return expect(p, "]");
    }
    if (teq(t, "{") || teq(t, "(")) {
        const char *close = teq(t, "{") ? "}" : ")";
        adv(p);
        op->kind = OP_VEC;
        int cap = 8;
        op->elems = (operand_t *)calloc((size_t)cap, sizeof(operand_t));
        while (!teq(pk(p), close) && !at_eof(p)) {
            if (op->nelems == cap) { cap *= 2; op->elems = (operand_t *)realloc(op->elems, (size_t)cap * sizeof(operand_t)); }
            if (!parse_operand(p, &op->elems[op->nelems++])) return 0;
            if (!accept(p, ",")) break;
        }
        return expect(p, close);
    }
    if (teq(t, "!")) {
        adv(p);
        if (!parse_operand(p, op)) return 0;
        op->not_ = 1;
        return 1;
    }
    if (teq(t, "-") || t->k == TK_NUM) {
        int neg = accept(p, "-");
        if (pk(p)->k != TK_NUM) { PERR(p, "PTX parse error (line %d): bad immediate", pk(p)->line); return 0; }
        char b[64];
        tcpy(b, sizeof b, pk(p));
        adv(p);
        op->kind = OP_IMM;
        parse_num(b, &op->imm_kind, &op->bits);
        if (neg) {
            if (op->imm_kind == 0) op->bits = (uint64_t)(-(int64_t)op->bits);
            else if (op->imm_kind == 1) op->bits ^= 0x80000000ull;
            else op->bits ^= 0x8000000000000000ull;
        }
        return 1;
    }
    if (t->k == TK_ID) {
        tcpy(op->name, sizeof op->name, t);
        adv(p);
        if (!strcmp(op->name, "_")) { op->kind = OP_SINK; return 1; }
        op->kind = op->name[0] == '%' ? OP_REG : OP_SYM;
        if (teq(pk(p), "|")) {
            adv(p);
            tcpy(op->name2, sizeof op->name2, pk(p));
            adv(p);
        }
        if (op->kind == OP_SYM && teq(pk(p), "+") && pk1(p)->k == TK_NUM) {
            adv(p);
            int64_t v;
            if (!parse_int_tok(p, &v)) return 0;
            op->off = v;
        }
        /* generic(sym) / function-call style wrappers */
        if (op->kind == OP_SYM && teq(pk(p), "(") && !strcmp(op->name, "generic")) {
            adv(p);
            tcpy(op->name, sizeof op->name, pk(p));
            adv(p);
            expect(p, ")");
        }
        return 1;
    }
    char got[64];
    tcpy(got, sizeof got, t);
    PERR(p, "PTX parse error (line %d): unexpected operand '%s'", t->line, got);
    return 0;
}

static void free_operand(operand_t *op) {
    if (op->kind == OP_VEC) {
        for (int i = 0; i < op->nelems; i++) free_operand(&op->elems[i]);
        free(op->elems);
    }
}

static var_t *add_var(var_t **arr, int *n, int *cap) {
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 16; *arr = (var_t *)realloc(*arr, (size_t)*cap * sizeof(var_t)); }
    var_t *v = &(*arr)[(*n)++];
    memset(v, 0, sizeof *v);
    return v;
}

static uint64_t align_up(uint64_t v, uint64_t a) { return a ? (v + a - 1) / a * a : v; }

/* Writes one initializer element into the module global image. */
static void put_init(P *p, var_t *v, uint64_t idx, int kind, uint64_t bits) {
    int es = VT_SIZE[v->elem_vt];
    if (v->elem_vt == VT_PRED) es = 1;
    uint64_t at = v->offset + idx * (uint64_t)es;
    if (at + (uint64_t)es > p->gbytes) return;
    uint8_t *dst = p->ginit + at;
    uint64_t val = bits;
    if (v->elem_vt == VT_F32) {
        if (kind == 0) { float f = (float)(int64_t)bits; uint32_t u; memcpy(&u, &f, 4); val = u; }
        else if (kind == 2) { double d; memcpy(&d, &bits, 8); float f = (float)d; uint32_t u; memcpy(&u, &f, 4); val = u; }
    } else if (v->elem_vt == VT_F64) {
        if (kind == 0) { double d = (double)(int64_t)bits; memcpy(&val, &d, 8); }
        else if (kind == 1) { uint32_t u = (uint32_t)bits; float f; memcpy(&f, &u, 4); double d = f; memcpy(&val, &d, 8); }
    }
    memcpy(dst, &val, (size_t)es); /* little endian */
}

static int parse_initializer(P *p, var_t *v, uint64_t *idx) {
    if (accept(p, "{")) {
        while (!teq(pk(p), "}") && !at_eof(p)) {
            if (!parse_initializer(p, v, idx)) return 0;
            if (!accept(p, ",")) break;
        }
        return expect(p, "}");
    }
    int neg = accept(p, "-");
    if (pk(p)->k == TK_NUM) {
        char b[64];
        tcpy(b, sizeof b, pk(p));
        adv(p);
        int kind;
        uint64_t bits;
        parse_num(b, &kind, &bits);
        if (neg) {
            if (kind == 0) bits = (uint64_t)(-(int64_t)bits);
            else if (kind == 1) bits ^= 0x80000000ull;
            else bits ^= 0x8000000000000000ull;
        }
        if (v->space == SP_GLOBAL || v->space == SP_CONST) put_init(p, v, *idx, kind, bits);
        (*idx)++;
        return 1;
    }
    if (pk(p)->k == TK_ID) { /* address-of initializers: generic(sym), sym+off */
        char nm[MC_MAX_NAME];
        tcpy(nm, sizeof nm, pk(p));
        adv(p);
        if (accept(p, "(")) { adv(p); expect(p, ")"); }
        if (accept(p, "+")) { int64_t d; parse_int_tok(p, &d); }
        warn(p, "initializer of '%s' takes the address of '%s'; initialised to 0", v->name, nm);
        (*idx)++;
        return 1;
    }
    PERR(p, "PTX parse error (line %d): bad initializer", pk(p)->line);
    return 0;
}

/* Parses `.global/.const/.shared/.local/.param [...] name[dims] [= init];` */
static int parse_var(P *p, int is_extern, var_t **arr, int *n, int *cap, int module_scope) {
    int space;
    if (teq(pk(p), ".global")) space = SP_GLOBAL;
    else if (teq(pk(p), ".const")) space = SP_CONST;
    else if (teq(pk(p), ".shared")) space = SP_SHARED;
    else if (teq(pk(p), ".local")) space = SP_LOCAL;
    else space = SP_PARAM;
    adv(p);
    uint32_t align = 0;
    int vt = -1, vec = 1;
    for (;;) {
        tok_t *t = pk(p);
        if (teq(t, ".align")) { adv(p); int64_t a; if (!parse_int_tok(p, &a)) return 0; align = (uint32_t)a; continue; }
        if (teq(t, ".v2")) { vec = 2; adv(p); continue; }
        if (teq(t, ".v4")) { vec = 4; adv(p); continue; }
        if (teq(t, ".ptr") || teq(t, ".attribute") || teq(t, ".managed") || teq(t, ".unified")) {
            adv(p);
            if (accept(p, "(")) { while (!teq(pk(p), ")") && !at_eof(p)) adv(p); expect(p, ")"); }
            continue;
        }
        if (t->k == TK_ID && t->s[0] == '.') {
            char b[32];
            tcpy(b, sizeof b, t);
            int x2;
            int v = vt_parse(b, &x2);
            if (v >= 0) { vt = v; if (x2) vec *= 1; adv(p); continue; }
            if (teq(t, ".shared::cta") ) { space = SP_SHARED; adv(p); continue; }
        }
        break;
    }
    if (vt < 0) { PERR(p, "PTX parse error (line %d): variable without type", pk(p)->line); return 0; }
    for (;;) {
        if (pk(p)->k != TK_ID) { PERR(p, "PTX parse error (line %d): expected variable name", pk(p)->line); return 0; }
        var_t *v = add_var(arr, n, cap);
        tcpy(v->name, sizeof v->name, pk(p));
        adv(p);
        v->space = space;
        v->elem_vt = vt;
        int es = (vt == VT_PRED ? 1 : VT_SIZE[vt]) * vec;
        v->align = align ? align : (uint32_t)es;
        uint64_t count = 1;
        int unsized = 0;
        while (accept(p, "[")) {
            if (accept(p, "]")) { unsized = 1; continue; }
            int64_t d;
            if (!parse_int_tok(p, &d)) return 0;
            count *= (uint64_t)d;
            if (!expect(p, "]")) return 0;
        }
        v->size = unsized ? 0 : count * (uint64_t)es;
        v->is_extern = (space == SP_SHARED) && (is_extern || unsized);
        if (module_scope && (space == SP_GLOBAL || space == SP_CONST)) {
            uint64_t off = align_up(p->gbytes, v->align < 16 ? 16 : v->align);
            v->offset = off;
            uint64_t nb = off + (v->size ? v->size : 0);
            p->ginit = (uint8_t *)realloc(p->ginit, nb ? nb : 1);
            memset(p->ginit + p->gbytes, 0, nb - p->gbytes);
            p->gbytes = nb;
        }
        if (accept(p, "=")) {
            uint64_t idx = 0;
            if (!parse_initializer(p, v, &idx)) return 0;
        }
        if (accept(p, ",")) continue;
        break;
    }
    return expect(p, ";");
}

static func_t *add_func(P *p) {
    if (p->nfuncs == p->capfuncs) {
        p->capfuncs = p->capfuncs ? p->capfuncs * 2 : 8;
        p->funcs = (func_t *)realloc(p->funcs, (size_t)p->capfuncs * sizeof(func_t));
    }
    func_t *f = &p->funcs[p->nfuncs++];
    memset(f, 0, sizeof *f);
    return f;
}

static int parse_params(P *p, func_t *f, int record) {
    if (!expect(p, "(")) return 0;
    uint32_t off = 0;
    int cap = 0;
    while (!teq(pk(p), ")") && !at_eof(p)) {
        if (!accept(p, ".param") && !accept(p, ".reg")) { PERR(p, "PTX parse error (line %d): expected .param", pk(p)->line); return 0; }
        uint32_t align = 0;
        int vt = -1, after_ptr = 0;
        for (;;) {
            tok_t *t = pk(p);
            if (teq(t, ".align")) {
                adv(p);
                int64_t a;
                if (!parse_int_tok(p, &a)) return 0;
                if (!after_ptr) align = (uint32_t)a; /* after .ptr it's the pointee alignment */
                continue;
            }
            if (teq(t, ".ptr")) { after_ptr = 1; adv(p); continue; }
            if (t->k == TK_ID && t->s[0] == '.') {
                char b[32];
                tcpy(b, sizeof b, t);
                int v = vt_parse(b, NULL);
                if (v >= 0) { vt = v; adv(p); continue; }
                /* .global/.shared/.const/.local after .ptr */
                adv(p);
                continue;
            }
            break;
        }
        if (vt < 0) { PERR(p, "PTX parse error (line %d): param without type", pk(p)->line); return 0; }
        char name[MC_MAX_NAME];
        tcpy(name, sizeof name, pk(p));
        adv(p);
        uint32_t es = (uint32_t)VT_SIZE[vt], count = 1;
        while (accept(p, "[")) {
            int64_t d;
            if (!parse_int_tok(p, &d)) return 0;
            count *= (uint32_t)d;
            expect(p, "]");
        }
        if (!align) align = es;
        off = (uint32_t)align_up(off, align);
        if (record) {
            if (f->nparams == cap) { cap = cap ? cap * 2 : 8; f->params = (mc_param_t *)realloc(f->params, (size_t)cap * sizeof(mc_param_t)); }
            mc_param_t *pp = &f->params[f->nparams++];
            memset(pp, 0, sizeof *pp);
            snprintf(pp->name, sizeof pp->name, "%s", name);
            pp->offset = off;
            pp->size = es * count;
            pp->align = align;
        }
        off += es * count;
        if (!accept(p, ",")) break;
    }
    if (record) f->param_bytes = off;
    return expect(p, ")");
}

static regdecl_t *add_reg(func_t *f) {
    if (f->nregs == f->capregs) {
        f->capregs = f->capregs ? f->capregs * 2 : 16;
        f->regs = (regdecl_t *)realloc(f->regs, (size_t)f->capregs * sizeof(regdecl_t));
    }
    regdecl_t *r = &f->regs[f->nregs++];
    memset(r, 0, sizeof *r);
    return r;
}

static stmt_t *add_stmt(func_t *f) {
    if (f->nstmts == f->capstmts) {
        f->capstmts = f->capstmts ? f->capstmts * 2 : 64;
        f->stmts = (stmt_t *)realloc(f->stmts, (size_t)f->capstmts * sizeof(stmt_t));
    }
    stmt_t *s = &f->stmts[f->nstmts++];
    memset(s, 0, sizeof *s);
    return s;
}

static int parse_body(P *p, func_t *f) {
    int depth = 1;
    while (depth > 0) {
        if (at_eof(p)) { PERR(p, "PTX parse error: unexpected end of file in body of %s", f->name); return 0; }
        tok_t *t = pk(p);
        if (teq(t, "{")) { depth++; adv(p); continue; }
        if (teq(t, "}")) { depth--; adv(p); continue; }
        if (teq(t, ";")) { adv(p); continue; }
        if (teq(t, ".loc") || teq(t, ".file")) { skip_line(p); continue; }
        if (teq(t, ".pragma")) { skip_until_semicolon(p); continue; }
        if (teq(t, ".reg")) {
            adv(p);
            int vt = -1;
            while (pk(p)->k == TK_ID && pk(p)->s[0] == '.') {
                char b[32];
                tcpy(b, sizeof b, pk(p));
                if (!strcmp(b, ".v2") || !strcmp(b, ".v4")) { PERR(p, "PTX line %d: vector registers are not supported", pk(p)->line); return 0; }
                int v = vt_parse(b, NULL);
                if (v >= 0) vt = v;
                adv(p);
            }
            if (vt < 0) { PERR(p, "PTX parse error (line %d): .reg without type", pk(p)->line); return 0; }
            for (;;) {
                regdecl_t *r = add_reg(f);
                tcpy(r->prefix, sizeof r->prefix, pk(p));
                r->vt = vt;
                adv(p);
                if (accept(p, "<")) {
                    int64_t c;
                    if (!parse_int_tok(p, &c)) return 0;
                    r->count = (int)c;
                    if (!expect(p, ">")) return 0;
                }
                if (!accept(p, ",")) break;
            }
            if (!expect(p, ";")) return 0;
            continue;
        }
        if (teq(t, ".shared") || teq(t, ".local") || teq(t, ".param") || teq(t, ".const") || teq(t, ".global")) {
            if (!parse_var(p, 0, &f->vars, &f->nvars, &f->capvars, 0)) return 0;
            continue;
        }
        if (t->k == TK_ID && teq(pk1(p), ":") && t->s[0] != '.') {
            stmt_t *s = add_stmt(f);
            s->is_label = 1;
            tcpy(s->label, sizeof s->label, t);
            adv(p);
            adv(p);
            continue;
        }
        /* instruction */
        stmt_t *s = add_stmt(f);
        instr_t *in = &s->ins;
        in->line = t->line;
        if (accept(p, "@")) {
            if (accept(p, "!")) in->pred_not = 1;
            tcpy(in->pred, sizeof in->pred, pk(p));
            adv(p);
        }
        if (pk(p)->k != TK_ID) {
            char got[64];
            tcpy(got, sizeof got, pk(p));
            PERR(p, "PTX parse error (line %d): expected instruction, found '%s'", pk(p)->line, got);
            return 0;
        }
        tcpy(in->opcode, sizeof in->opcode, pk(p));
        adv(p);
        split_parts(in);
        while (!teq(pk(p), ";") && !at_eof(p)) {
            if (in->nops == MAX_OPS) { PERR(p, "PTX line %d: too many operands", in->line); return 0; }
            if (!parse_operand(p, &in->ops[in->nops++])) return 0;
            if (!accept(p, ",")) break;
        }
        if (!expect(p, ";")) return 0;
    }
    return 1;
}

static int parse_function(P *p, int is_entry) {
    adv(p); /* .entry / .func */
    func_t *f = add_func(p);
    f->is_entry = is_entry;
    if (!is_entry && teq(pk(p), "(")) { /* return value list */
        func_t dummy;
        memset(&dummy, 0, sizeof dummy);
        if (!parse_params(p, &dummy, 0)) return 0;
    }
    tcpy(f->name, sizeof f->name, pk(p));
    adv(p);
    if (teq(pk(p), "(")) {
        if (!parse_params(p, f, 1)) return 0;
    }
    for (;;) {
        tok_t *t = pk(p);
        if (teq(t, "{") || teq(t, ";") || at_eof(p)) break;
        if (teq(t, ".maxntid") || teq(t, ".reqntid")) {
            int isreq = teq(t, ".reqntid");
            adv(p);
            for (int d = 0; d < 3; d++) {
                int64_t v;
                if (!parse_int_tok(p, &v)) return 0;
                if (isreq || !f->reqntid[d]) f->reqntid[d] = (uint32_t)v;
                if (!accept(p, ",")) break;
            }
            continue;
        }
        if (teq(t, ".pragma")) { skip_until_semicolon(p); continue; }
        if (teq(t, ".minnctapersm") || teq(t, ".maxnreg") || teq(t, ".maxclusterrank") || teq(t, ".reqnctapercluster")) {
            adv(p);
            while (pk(p)->k == TK_NUM || teq(pk(p), ",")) adv(p);
            continue;
        }
        adv(p); /* .noreturn, .explicitcluster, ... */
    }
    if (accept(p, ";")) { f->is_proto = 1; return 1; }
    if (!expect(p, "{")) return 0;
    return parse_body(p, f);
}

static int parse_module(P *p) {
    while (!at_eof(p) && !p->err[0]) {
        tok_t *t = pk(p);
        if (teq(t, ".version") || teq(t, ".target") || teq(t, ".address_size") || teq(t, ".file") || teq(t, ".loc")) {
            skip_line(p);
            continue;
        }
        if (teq(t, ".section")) {
            while (!at_eof(p) && !teq(pk(p), "{")) adv(p);
            int depth = 0;
            do {
                if (teq(pk(p), "{")) depth++;
                else if (teq(pk(p), "}")) depth--;
                adv(p);
            } while (depth > 0 && !at_eof(p));
            continue;
        }
        if (teq(t, ".pragma")) { skip_until_semicolon(p); continue; }
        if (teq(t, ";")) { adv(p); continue; }
        int is_extern = 0;
        while (teq(pk(p), ".visible") || teq(pk(p), ".extern") || teq(pk(p), ".weak") || teq(pk(p), ".common")) {
            if (teq(pk(p), ".extern")) is_extern = 1;
            adv(p);
        }
        t = pk(p);
        if (teq(t, ".entry")) { if (!parse_function(p, 1)) return 0; continue; }
        if (teq(t, ".func")) { if (!parse_function(p, 0)) return 0; continue; }
        if (teq(t, ".global") || teq(t, ".const") || teq(t, ".shared") || teq(t, ".local")) {
            if (!parse_var(p, is_extern, &p->mvars, &p->nmvars, &p->capmvars, 1)) return 0;
            continue;
        }
        if (teq(t, ".alias")) { skip_until_semicolon(p); continue; }
        char got[64];
        tcpy(got, sizeof got, t);
        PERR(p, "PTX parse error (line %d): unexpected '%s' at module scope", t->line, got);
        return 0;
    }
    return p->err[0] ? 0 : 1;
}

/* ───────────────────────────── emitter ───────────────────────────── */

#define TAG_SHARED 0xFFFFull
#define TAG_LOCAL  0xFFFEull
#define TAG_PARAM  0xFFFDull

typedef struct {
    P *p;
    func_t *f;
    mc_kernel_t *ki;
    sb_t *o;
    int uses_smem, uses_generic_tags, uses_bar_red, convergent;
    uint32_t smem_static, smem_dyn_off, red_off, local_bytes;
    int tmp; /* unique temp counter */
} K;

typedef struct { var_t *v; int module; } symref_t;

static symref_t find_sym(K *k, const char *name) {
    symref_t r = {0, 0};
    for (int i = 0; i < k->f->nvars; i++)
        if (!strcmp(k->f->vars[i].name, name)) { r.v = &k->f->vars[i]; return r; }
    for (int i = 0; i < k->p->nmvars; i++)
        if (!strcmp(k->p->mvars[i].name, name)) { r.v = &k->p->mvars[i]; r.module = 1; return r; }
    return r;
}
static mc_param_t *find_param(K *k, const char *name) {
    for (int i = 0; i < k->f->nparams; i++)
        if (!strcmp(k->f->params[i].name, name)) return &k->f->params[i];
    return NULL;
}

/* Register lookup: returns declared PTX VT or -1. */
static int reg_vt(K *k, const char *name) {
    for (int i = 0; i < k->f->nregs; i++) {
        regdecl_t *r = &k->f->regs[i];
        if (r->count == 0) {
            if (!strcmp(r->prefix, name)) return r->vt;
        } else {
            size_t pl = strlen(r->prefix);
            if (!strncmp(name, r->prefix, pl) && name[pl] && isdigit((unsigned char)name[pl])) {
                const char *q = name + pl;
                while (isdigit((unsigned char)*q)) q++;
                if (!*q && atoi(name + pl) < r->count) return r->vt;
            }
        }
    }
    return -1;
}

static void sanitize(char *out, size_t n, const char *in, const char *prefix) {
    size_t j = 0;
    for (const char *q = prefix; *q && j + 1 < n; q++) out[j++] = *q;
    for (const char *q = in; *q && j + 2 < n; q++) {
        char c = *q;
        if (c == '%') continue;
        if (isalnum((unsigned char)c) || c == '_') out[j++] = c;
        else if (c == '$') { out[j++] = '_'; out[j++] = 'S'; }
        else out[j++] = '_';
    }
    out[j] = 0;
}
static void reg_cname(char *out, size_t n, const char *name) { sanitize(out, n, name, "v_"); }

/* Special registers → (expression, vt). Returns 0 if not special. */
static int special_reg(K *k, const char *n, char *out, size_t on, int *vt) {
    *vt = VT_U32;
    static const char *dims[] = {"tid", "ntid", "ctaid", "nctaid"};
    for (int d = 0; d < 4; d++) {
        char b[32];
        for (int c = 0; c < 3; c++) {
            snprintf(b, sizeof b, "%%%s.%c", dims[d], "xyz"[c]);
            if (!strcmp(n, b)) { snprintf(out, on, "__%s.%c", dims[d], "xyz"[c]); return 1; }
        }
    }
    if (!strcmp(n, "%laneid")) { snprintf(out, on, "__lane"); return 1; }
    if (!strcmp(n, "%warpid")) { snprintf(out, on, "__warpid"); return 1; }
    if (!strcmp(n, "%nwarpid")) { snprintf(out, on, "((__ntid.x*__ntid.y*__ntid.z+31u)/32u)"); return 1; }
    if (!strcmp(n, "%smid")) { snprintf(out, on, "0u"); return 1; }
    if (!strcmp(n, "%nsmid")) { snprintf(out, on, "MC_NUM_SM"); return 1; }
    if (!strcmp(n, "%gridid")) { *vt = VT_U64; snprintf(out, on, "0ul"); return 1; }
    if (!strcmp(n, "%lanemask_eq")) { snprintf(out, on, "(1u << __lane)"); return 1; }
    if (!strcmp(n, "%lanemask_lt")) { snprintf(out, on, "((1u << __lane) - 1u)"); return 1; }
    if (!strcmp(n, "%lanemask_le")) { snprintf(out, on, "((2u << __lane) - 1u)"); return 1; }
    if (!strcmp(n, "%lanemask_gt")) { snprintf(out, on, "(~((2u << __lane) - 1u))"); return 1; }
    if (!strcmp(n, "%lanemask_ge")) { snprintf(out, on, "(~((1u << __lane) - 1u))"); return 1; }
    if (!strcmp(n, "%clock") || !strcmp(n, "%clock_hi")) { snprintf(out, on, "0u"); return 1; }
    if (!strcmp(n, "%clock64") || !strcmp(n, "%globaltimer")) { *vt = VT_U64; snprintf(out, on, "0ul"); return 1; }
    if (!strcmp(n, "%globaltimer_lo") || !strcmp(n, "%globaltimer_hi")) { snprintf(out, on, "0u"); return 1; }
    if (!strcmp(n, "%dynamic_smem_size")) { snprintf(out, on, "__dynsmem"); return 1; }
    if (!strcmp(n, "%total_smem_size")) { snprintf(out, on, "(%uu + __dynsmem)", k->smem_dyn_off); return 1; }
    if (!strncmp(n, "%envreg", 7) || !strncmp(n, "%pm", 3)) { snprintf(out, on, "0u"); return 1; }
    return 0;
}

/* Immediate → expression of type vt. */
static void imm_expr(char *out, size_t n, const operand_t *op, int vt) {
    uint64_t b = op->bits;
    if (vt == VT_PRED) { snprintf(out, n, "%s", b ? "true" : "false"); return; }
    if (op->imm_kind == 0) {
        if (vt_isfloat(vt)) {
            int64_t sv = (int64_t)b;
            if (vt == VT_BF16) snprintf(out, n, "__f2bf((float)(%lld))", (long long)sv);
            else snprintf(out, n, "((%s)(%lld.0f))", VT_C[vt], (long long)sv);
            return;
        }
    } else {
        /* f32 / f64 literal */
        uint32_t f32bits;
        if (op->imm_kind == 1) f32bits = (uint32_t)b;
        else { double d; memcpy(&d, &b, 8); float f = (float)d; memcpy(&f32bits, &f, 4); }
        if (vt == VT_F32 || vt == VT_F64) { snprintf(out, n, "as_type<float>(0x%08xu)", f32bits); return; }
        if (vt == VT_F16) { snprintf(out, n, "((half)as_type<float>(0x%08xu))", f32bits); return; }
        if (vt == VT_BF16) { snprintf(out, n, "__f2bf(as_type<float>(0x%08xu))", f32bits); return; }
        /* bit pattern into an integer register */
        if (op->imm_kind == 1) b = f32bits; /* 0f: 32-bit pattern */
    }
    switch (VT_SIZE[vt]) {
    case 1: snprintf(out, n, "((%s)0x%xu)", VT_C[vt], (unsigned)(b & 0xff)); break;
    case 2: snprintf(out, n, "((%s)0x%xu)", VT_C[vt], (unsigned)(b & 0xffff)); break;
    case 4: snprintf(out, n, vt_issigned(vt) ? "((int)0x%xu)" : "0x%xu", (unsigned)(b & 0xffffffffu)); break;
    default: snprintf(out, n, vt_issigned(vt) ? "((long)0x%llxul)" : "0x%llxul", (unsigned long long)b); break;
    }
}

static int sym_value(K *k, const operand_t *op, char *out, size_t n); /* fwd */

/* Read operand as type vt (storage semantics). */
static int rd(K *k, const operand_t *op, int vt, char *out, size_t n) {
    char t[EXL], c[EXL];
    switch (op->kind) {
    case OP_REG: {
        int rv = reg_vt(k, op->name);
        if (rv >= 0) {
            reg_cname(c, sizeof c, op->name);
            castx(t, sizeof t, c, vt_reg_storage(rv), vt);
        } else {
            int svt;
            if (!special_reg(k, op->name, c, sizeof c, &svt)) {
                PERR(k->p, "PTX: unknown register '%s' in %s", op->name, k->f->name);
                return 0;
            }
            castx(t, sizeof t, c, svt, vt);
        }
        if (op->not_) snprintf(out, n, "(!%s)", t);
        else snprintf(out, n, "%s", t);
        return 1;
    }
    case OP_IMM: imm_expr(out, n, op, vt); return 1;
    case OP_SYM: {
        if (!sym_value(k, op, c, sizeof c)) return 0;
        castx(out, n, c, VT_U64, vt);
        return 1;
    }
    case OP_SINK: snprintf(out, n, "0"); return 1;
    default: PERR(k->p, "PTX: unexpected operand kind in %s", k->f->name); return 0;
    }
}
/* Read operand numerically (bf16 is widened to float). */
static int rdn(K *k, const operand_t *op, int vt, char *out, size_t n) {
    if (vt != VT_BF16) return rd(k, op, vt, out, n);
    char t[EXL];
    if (!rd(k, op, VT_U16, t, sizeof t)) return 0;
    snprintf(out, n, "__bf2f(%s)", t);
    return 1;
}
static int vt_compute(int vt) { return vt == VT_BF16 ? VT_F32 : vt; }

/* Emit `dst = expr` where expr has type vt. */
static int wr_named(K *k, const char *name, const char *expr, int vt) {
    if (!strcmp(name, "_")) { sb_printf(k->o, "(void)(%s);\n", expr); return 1; }
    int rv = reg_vt(k, name);
    if (rv < 0) { PERR(k->p, "PTX: unknown destination register '%s' in %s", name, k->f->name); return 0; }
    char c[EXL], t[EXL];
    reg_cname(c, sizeof c, name);
    castx(t, sizeof t, expr, vt, vt_reg_storage(rv));
    sb_printf(k->o, "%s = %s;\n", c, t);
    return 1;
}
static int wr(K *k, const operand_t *d, const char *expr, int vt) {
    if (d->kind == OP_SINK) return 1;
    if (d->kind != OP_REG) { PERR(k->p, "PTX: destination is not a register in %s", k->f->name); return 0; }
    return wr_named(k, d->name, expr, vt);
}
/* write numeric (compute-type) result into type vt */
static int wrn(K *k, const operand_t *d, const char *expr, int vt) {
    if (vt != VT_BF16) return wr(k, d, expr, vt);
    char t[EXL];
    snprintf(t, sizeof t, "__f2bf(%s)", expr);
    return wr(k, d, t, VT_U16);
}

static int sym_value(K *k, const operand_t *op, char *out, size_t n) {
    symref_t s = find_sym(k, op->name);
    long long off = (long long)op->off;
    if (s.v) {
        switch (s.v->space) {
        case SP_SHARED:
            k->uses_smem = 1;
            snprintf(out, n, "((ulong)%lluul)", (unsigned long long)((s.v->is_extern ? k->smem_dyn_off : s.v->offset) + (uint64_t)off));
            return 1;
        case SP_LOCAL:
            snprintf(out, n, "((ulong)%lluul)", (unsigned long long)(s.v->offset + (uint64_t)off));
            return 1;
        case SP_GLOBAL:
        case SP_CONST:
            snprintf(out, n, "(__gbase + %lluul)", (unsigned long long)(s.v->offset + (uint64_t)off));
            return 1;
        case SP_PARAM:
            PERR(k->p, "PTX: taking the address of call parameter '%s' is not supported", op->name);
            return 0;
        }
    }
    mc_param_t *pp = find_param(k, op->name);
    if (pp) { snprintf(out, n, "((ulong)%uu)", pp->offset + (uint32_t)off); return 1; }
    PERR(k->p, "PTX: unknown symbol '%s' in %s (function pointers / external symbols are not supported)", op->name, k->f->name);
    return 0;
}

/* Computes the address expression (ulong) for an OP_ADDR operand. *space is
 * refined when the base is a symbol of a known state space. */
static int addr_expr(K *k, const operand_t *op, int *space, char *out, size_t n) {
    char b[EXL];
    if (!op->has_base) { snprintf(out, n, "((ulong)%lldl)", (long long)op->off); return 1; }
    if (op->name[0] == '%') {
        operand_t r;
        memset(&r, 0, sizeof r);
        r.kind = OP_REG;
        snprintf(r.name, sizeof r.name, "%s", op->name);
        if (!rd(k, &r, VT_U64, b, sizeof b)) return 0;
        if (op->off) snprintf(out, n, "(%s + (ulong)(%lldl))", b, (long long)op->off);
        else snprintf(out, n, "%s", b);
        return 1;
    }
    /* symbol base */
    symref_t s = find_sym(k, op->name);
    mc_param_t *pp = s.v ? NULL : find_param(k, op->name);
    if (s.v) {
        if (s.v->space == SP_PARAM) { /* call-site .param scratch */
            char cn[EXL];
            sanitize(cn, sizeof cn, s.v->name, "__cp_");
            if (*space == SP_PARAM || *space == SP_NONE) *space = -SP_PARAM; /* marker: local param array */
            snprintf(out, n, "%lldl", (long long)op->off);
            (void)cn;
            return 1;
        }
        if (*space == SP_GENERIC || *space == SP_NONE) {
            if (s.v->space == SP_SHARED) *space = SP_SHARED;
            else if (s.v->space == SP_LOCAL) *space = SP_LOCAL;
            else *space = SP_GLOBAL;
        }
    } else if (pp) {
        if (*space == SP_GENERIC || *space == SP_NONE) *space = SP_PARAM;
    }
    operand_t so;
    memset(&so, 0, sizeof so);
    so.kind = OP_SYM;
    snprintf(so.name, sizeof so.name, "%s", op->name);
    so.off = op->off;
    return sym_value(k, &so, out, n);
}

/* memory C type for a value type (f64 travels as its IEEE bits) */
static const char *mem_ctype(int vt) {
    if (vt == VT_F64) return "ulong";
    if (vt == VT_PRED) return "uchar";
    return VT_C[vt];
}

static int space_of(instr_t *in) {
    for (int i = 1; i < in->np; i++) {
        const char *s = in->parts[i];
        if (!strcmp(s, "global")) return SP_GLOBAL;
        if (!strncmp(s, "shared", 6)) return SP_SHARED;
        if (!strcmp(s, "local")) return SP_LOCAL;
        if (!strcmp(s, "param")) return SP_PARAM;
        if (!strcmp(s, "const")) return SP_CONST;
    }
    return SP_GENERIC;
}
static int has_part(instr_t *in, const char *s) {
    for (int i = 1; i < in->np; i++)
        if (!strcmp(in->parts[i], s)) return 1;
    return 0;
}
static int vec_width(instr_t *in) {
    for (int i = 1; i < in->np; i++) {
        if (!strcmp(in->parts[i], "v2")) return 2;
        if (!strcmp(in->parts[i], "v4")) return 4;
        if (!strcmp(in->parts[i], "v8")) return 8;
    }
    return 1;
}
/* last type part (or the n-th type part when nth>=0) */
static int type_part(instr_t *in, int nth, int *x2) {
    int found = 0, res = -1;
    for (int i = 1; i < in->np; i++) {
        int x;
        int v = vt_parse(in->parts[i], &x);
        if (v >= 0) {
            if (nth < 0 || found == nth) { res = v; if (x2) *x2 = x; }
            found++;
            if (nth >= 0 && found > nth) break;
        }
    }
    return res;
}

/* Build an lvalue/rvalue for memory at `a` (ulong expr) in `space` with C type `ct`. */
static void deref(K *k, char *out, size_t n, int space, const char *ct, const char *a, int vol) {
    const char *v = vol ? "volatile " : "";
    switch (space) {
    case SP_SHARED: k->uses_smem = 1; snprintf(out, n, "(*(%sthreadgroup %s*)(__smem + (uint)(%s)))", v, ct, a); break;
    case SP_LOCAL: snprintf(out, n, "(*(%sthread %s*)(__ldepot + (uint)(%s)))", v, ct, a); break;
    case SP_PARAM: snprintf(out, n, "(*(constant %s*)(__params + (uint)(%s)))", ct, a); break;
    default: snprintf(out, n, "(*(%sdevice %s*)(%s))", v, ct, a); break;
    }
}

/* Generic-address load expression (runtime dispatch on the address tag). */
static void generic_load(K *k, char *out, size_t n, const char *ct, const char *a, int vol) {
    char g[EXL], s[EXL], l[EXL], pa[EXL];
    deref(k, g, sizeof g, SP_GLOBAL, ct, a, vol);
    if (!k->uses_generic_tags) { snprintf(out, n, "%s", g); return; }
    deref(k, s, sizeof s, SP_SHARED, ct, a, vol);
    deref(k, l, sizeof l, SP_LOCAL, ct, a, vol);
    deref(k, pa, sizeof pa, SP_PARAM, ct, a, 0);
    snprintf(out, n, "((((%s) >> 48) == 0xFFFFul) ? %s : (((%s) >> 48) == 0xFFFEul) ? %s : (((%s) >> 48) == 0xFFFDul) ? %s : %s)",
             a, s, a, l, a, pa, g);
}
static void generic_store(K *k, const char *ct, const char *a, const char *val, int vol) {
    char g[EXL], s[EXL], l[EXL];
    deref(k, g, sizeof g, SP_GLOBAL, ct, a, vol);
    if (!k->uses_generic_tags) { sb_printf(k->o, "%s = %s;\n", g, val); return; }
    deref(k, s, sizeof s, SP_SHARED, ct, a, vol);
    deref(k, l, sizeof l, SP_LOCAL, ct, a, vol);
    sb_printf(k->o, "{ ulong __ga = %s; if ((__ga >> 48) == 0xFFFFul) ", a);
    char s2[EXL], l2[EXL], g2[EXL];
    deref(k, s2, sizeof s2, SP_SHARED, ct, "__ga", vol);
    deref(k, l2, sizeof l2, SP_LOCAL, ct, "__ga", vol);
    deref(k, g2, sizeof g2, SP_GLOBAL, ct, "__ga", vol);
    sb_printf(k->o, "%s = %s; else if ((__ga >> 48) == 0xFFFEul) %s = %s; else %s = %s; }\n", s2, val, l2, val, g2, val);
    (void)s; (void)l;
}

static int emit_ldst(K *k, instr_t *in, int is_store) {
    int vt = type_part(in, -1, NULL);
    if (vt < 0) { PERR(k->p, "PTX line %d: %s without type", in->line, in->opcode); return 0; }
    if (has_part(in, "b128")) { PERR(k->p, "PTX line %d: 128-bit loads are not supported", in->line); return 0; }
    int space = space_of(in);
    int vol = has_part(in, "volatile") || has_part(in, "relaxed") || has_part(in, "acquire") || has_part(in, "release");
    int vw = vec_width(in);
    if (vw > 4) { PERR(k->p, "PTX line %d: .v8 accesses are not supported", in->line); return 0; }
    operand_t *mop = is_store ? &in->ops[0] : &in->ops[1];
    operand_t *vop = is_store ? &in->ops[1] : &in->ops[0];
    if (mop->kind != OP_ADDR) { PERR(k->p, "PTX line %d: expected [address]", in->line); return 0; }
    char a[EXL];
    int sp = space;
    if (!addr_expr(k, mop, &sp, a, sizeof a)) return 0;
    if (vt == VT_F64 && !k->p->warned_f64) { k->p->warned_f64 = 1; warn(k->p, "f64 arithmetic is emulated in fp32 (Metal has no double); memory layout stays binary64"); }
    const char *ct = mem_ctype(vt);
    int es = VT_SIZE[vt == VT_PRED ? VT_U8 : vt];
    int local_param = (sp == -SP_PARAM);
    char cpname[EXL] = "";
    if (local_param) sanitize(cpname, sizeof cpname, mop->name, "__cp_");
    for (int e = 0; e < vw; e++) {
        operand_t *val = vop->kind == OP_VEC ? &vop->elems[e] : vop;
        if (vop->kind == OP_VEC && e >= vop->nelems) break;
        char ae[EXL], m[EXL];
        if (e) snprintf(ae, sizeof ae, "(%s + %dul)", a, e * es);
        else snprintf(ae, sizeof ae, "%s", a);
        int isgen = 0;
        if (local_param) snprintf(m, sizeof m, "(*(thread %s*)(%s + (uint)(%s)))", ct, cpname, ae);
        else if (sp == SP_GENERIC) isgen = 1;
        else deref(k, m, sizeof m, sp, ct, ae, vol);
        if (is_store) {
            char v[EXL], vv[EXL];
            if (!rd(k, val, vt, v, sizeof v)) return 0;
            if (vt == VT_F64) snprintf(vv, sizeof vv, "__d_bits(%s)", v);
            else if (vt == VT_PRED) snprintf(vv, sizeof vv, "(uchar)(%s)", v);
            else snprintf(vv, sizeof vv, "%s", v);
            if (isgen) generic_store(k, ct, ae, vv, vol);
            else sb_printf(k->o, "%s = %s;\n", m, vv);
        } else {
            if (val->kind == OP_SINK) continue;
            char g[EXL], v[EXL];
            if (isgen) generic_load(k, g, sizeof g, ct, ae, vol);
            else snprintf(g, sizeof g, "%s", m);
            if (vt == VT_F64) snprintf(v, sizeof v, "__d_from_bits(%s)", g);
            else snprintf(v, sizeof v, "%s", g);
            if (!wr(k, val, v, vt)) return 0;
        }
    }
    return 1;
}

static const char *cmp_int(const char *c, int sgn) {
    (void)sgn;
    if (!strcmp(c, "eq")) return "==";
    if (!strcmp(c, "ne")) return "!=";
    if (!strcmp(c, "lt") || !strcmp(c, "lo")) return "<";
    if (!strcmp(c, "le") || !strcmp(c, "ls")) return "<=";
    if (!strcmp(c, "gt") || !strcmp(c, "hi")) return ">";
    if (!strcmp(c, "ge") || !strcmp(c, "hs")) return ">=";
    return NULL;
}

/* float comparison with PTX ordered/unordered semantics */
static int cmp_float(char *out, size_t n, const char *c, const char *a, const char *b) {
    if (!strcmp(c, "eq")) snprintf(out, n, "(%s == %s)", a, b);
    else if (!strcmp(c, "ne")) snprintf(out, n, "((%s < %s) || (%s > %s))", a, b, a, b);
    else if (!strcmp(c, "lt")) snprintf(out, n, "(%s < %s)", a, b);
    else if (!strcmp(c, "le")) snprintf(out, n, "(%s <= %s)", a, b);
    else if (!strcmp(c, "gt")) snprintf(out, n, "(%s > %s)", a, b);
    else if (!strcmp(c, "ge")) snprintf(out, n, "(%s >= %s)", a, b);
    else if (!strcmp(c, "equ")) snprintf(out, n, "(!((%s < %s) || (%s > %s)))", a, b, a, b);
    else if (!strcmp(c, "neu")) snprintf(out, n, "(%s != %s)", a, b);
    else if (!strcmp(c, "ltu")) snprintf(out, n, "(!(%s >= %s))", a, b);
    else if (!strcmp(c, "leu")) snprintf(out, n, "(!(%s > %s))", a, b);
    else if (!strcmp(c, "gtu")) snprintf(out, n, "(!(%s <= %s))", a, b);
    else if (!strcmp(c, "geu")) snprintf(out, n, "(!(%s < %s))", a, b);
    else if (!strcmp(c, "num")) snprintf(out, n, "(!isnan(%s) && !isnan(%s))", a, b);
    else if (!strcmp(c, "nan")) snprintf(out, n, "(isnan(%s) || isnan(%s))", a, b);
    else return 0;
    return 1;
}

/* Builds the comparison expression for setp/set. */
static int compare_expr(K *k, instr_t *in, int vt, const operand_t *A, const operand_t *B, char *out, size_t n) {
    const char *c = in->parts[1];
    char a[EXL], b[EXL];
    if (vt_isfloat(vt)) {
        if (!rdn(k, A, vt, a, sizeof a) || !rdn(k, B, vt, b, sizeof b)) return 0;
        if (!cmp_float(out, n, c, a, b)) { PERR(k->p, "PTX line %d: bad float comparison '%s'", in->line, c); return 0; }
        return 1;
    }
    int t = vt;
    if (!strcmp(c, "lo") || !strcmp(c, "ls") || !strcmp(c, "hi") || !strcmp(c, "hs")) t = vt_uns(vt);
    if (vt == VT_PRED) t = VT_PRED;
    if (!rd(k, A, t, a, sizeof a) || !rd(k, B, t, b, sizeof b)) return 0;
    const char *op = cmp_int(c, vt_issigned(t));
    if (!op) { PERR(k->p, "PTX line %d: bad integer comparison '%s'", in->line, c); return 0; }
    snprintf(out, n, "(%s %s %s)", a, op, b);
    return 1;
}

static const char *bool_op(instr_t *in) {
    for (int i = 2; i < in->np; i++) {
        if (!strcmp(in->parts[i], "and")) return "&&";
        if (!strcmp(in->parts[i], "or")) return "||";
        if (!strcmp(in->parts[i], "xor")) return "!=";
    }
    return NULL;
}

static const char *rounding_fn(instr_t *in) {
    for (int i = 1; i < in->np; i++) {
        const char *s = in->parts[i];
        if (!strcmp(s, "rni")) return "rint";
        if (!strcmp(s, "rzi")) return "trunc";
        if (!strcmp(s, "rmi")) return "floor";
        if (!strcmp(s, "rpi")) return "ceil";
    }
    return NULL;
}

/* float→int with PTX saturation semantics */
static void f2i(char *out, size_t n, const char *f, int dt) {
    switch (dt) {
    case VT_S32: snprintf(out, n, "__f2s32(%s)", f); break;
    case VT_U32: snprintf(out, n, "__f2u32(%s)", f); break;
    case VT_S64: snprintf(out, n, "__f2s64(%s)", f); break;
    case VT_U64: snprintf(out, n, "__f2u64(%s)", f); break;
    case VT_S16: snprintf(out, n, "((short)clamp(__f2s32(%s), -32768, 32767))", f); break;
    case VT_U16: snprintf(out, n, "((ushort)min(__f2u32(%s), 65535u))", f); break;
    case VT_S8: snprintf(out, n, "((char)clamp(__f2s32(%s), -128, 127))", f); break;
    default: snprintf(out, n, "((uchar)min(__f2u32(%s), 255u))", f); break;
    }
}

static int emit_cvt(K *k, instr_t *in) {
    int x2d = 0;
    int dt = type_part(in, 0, &x2d), st = type_part(in, 1, NULL);
    if (dt < 0 || st < 0) { PERR(k->p, "PTX line %d: malformed cvt", in->line); return 0; }
    char a[EXL], r[EXL];
    if (x2d) { /* cvt.rn.f16x2.f32 d, a, b  → d = {cvt(a) hi, cvt(b) lo} */
        char b[EXL];
        if (in->nops < 3 || !rd(k, &in->ops[1], VT_F32, a, sizeof a) || !rd(k, &in->ops[2], VT_F32, b, sizeof b)) return 0;
        if (x2d == 1) snprintf(r, sizeof r, "(((uint)as_type<ushort>((half)(%s)) << 16) | (uint)as_type<ushort>((half)(%s)))", a, b);
        else snprintf(r, sizeof r, "(((uint)__f2bf(%s) << 16) | (uint)__f2bf(%s))", a, b);
        return wr(k, &in->ops[0], r, VT_U32);
    }
    if (!rdn(k, &in->ops[1], st, a, sizeof a)) return 0;
    int sat = has_part(in, "sat");
    const char *rf = rounding_fn(in);
    if (vt_isfloat(st) && vt_isint(dt)) {
        char t[EXL];
        snprintf(t, sizeof t, "%s(%s)", rf ? rf : "rint", a);
        f2i(r, sizeof r, t, dt);
        return wr(k, &in->ops[0], r, dt);
    }
    if (vt_isfloat(st) && vt_isfloat(dt)) {
        char t[EXL];
        if (rf) snprintf(t, sizeof t, "%s(%s)", rf, a);
        else snprintf(t, sizeof t, "%s", a);
        if (sat) { char u[EXL]; snprintf(u, sizeof u, "saturate((float)(%s))", t); snprintf(t, sizeof t, "%s", u); }
        if (dt == VT_BF16) { snprintf(r, sizeof r, "__f2bf((float)(%s))", t); return wr(k, &in->ops[0], r, VT_U16); }
        snprintf(r, sizeof r, "((%s)(%s))", VT_C[dt], t);
        return wr(k, &in->ops[0], r, dt);
    }
    if (vt_isint(st) && vt_isfloat(dt)) {
        if (dt == VT_BF16) { snprintf(r, sizeof r, "__f2bf((float)(%s))", a); return wr(k, &in->ops[0], r, VT_U16); }
        if (sat) snprintf(r, sizeof r, "saturate((%s)(%s))", VT_C[dt], a);
        else snprintf(r, sizeof r, "((%s)(%s))", VT_C[dt], a);
        return wr(k, &in->ops[0], r, dt);
    }
    /* int → int */
    if (sat && VT_SIZE[dt] < 8) {
        int64_t lo = 0, hi = 0;
        switch (dt) {
        case VT_S8: lo = -128; hi = 127; break;
        case VT_U8: lo = 0; hi = 255; break;
        case VT_S16: lo = -32768; hi = 32767; break;
        case VT_U16: lo = 0; hi = 65535; break;
        case VT_S32: lo = -2147483648LL; hi = 2147483647LL; break;
        case VT_U32: lo = 0; hi = 4294967295LL; break;
        }
        if (vt_issigned(st)) snprintf(r, sizeof r, "((%s)clamp((long)(%s), (long)(%lld), (long)(%lld)))", VT_C[dt], a, (long long)lo, (long long)hi);
        else snprintf(r, sizeof r, "((%s)min((ulong)(%s), (ulong)%lldul))", VT_C[dt], a, (long long)hi);
    } else {
        snprintf(r, sizeof r, "((%s)(%s))", VT_C[dt], a);
    }
    return wr(k, &in->ops[0], r, dt);
}

static void mem_scope_fence(K *k, instr_t *in, int before) {
    int acq = has_part(in, "acquire") || has_part(in, "acq_rel");
    int rel = has_part(in, "release") || has_part(in, "acq_rel");
    if ((before && rel) || (!before && acq))
        sb_puts(k->o, "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);\n");
}

/* Atomic op on `space` at address `a`. Emits statements; result in `__old<N>`. */
static int emit_atomic(K *k, instr_t *in, int has_result) {
    int sp = space_of(in);
    int vt = type_part(in, -1, NULL);
    const char *op = NULL;
    static const char *ops[] = {"add", "inc", "dec", "exch", "cas", "min", "max", "and", "or", "xor"};
    for (int i = 1; i < in->np && !op; i++)
        for (size_t j = 0; j < sizeof ops / sizeof ops[0]; j++)
            if (!strcmp(in->parts[i], ops[j])) { op = ops[j]; break; }
    if (!op || vt < 0) { PERR(k->p, "PTX line %d: unsupported atomic '%s'", in->line, in->opcode); return 0; }
    int x2 = 0;
    type_part(in, -1, &x2);
    if (x2 || vt == VT_F16 || vt == VT_BF16) { PERR(k->p, "PTX line %d: 16-bit float atomics are not supported", in->line); return 0; }
    int base = has_result ? 1 : 0;
    operand_t *A = &in->ops[base];
    char a[EXL], b[EXL], c[EXL] = "";
    if (A->kind != OP_ADDR) { PERR(k->p, "PTX line %d: atomic needs [address]", in->line); return 0; }
    int s2 = sp;
    if (!addr_expr(k, A, &s2, a, sizeof a)) return 0;
    sp = s2;
    /* A generic address may point into shared memory (after cvta.shared): when
     * the kernel creates such addresses, dispatch on the address tag at runtime. */
    int variants = (sp == SP_GENERIC && k->uses_generic_tags) ? 2 : 1;
    if (sp == SP_GENERIC || sp == SP_CONST) sp = SP_GLOBAL;
    int bt = vt;
    if (bt == VT_PRED) bt = VT_U32;
    if (!rd(k, &in->ops[base + 1], bt, b, sizeof b)) return 0;
    if (!strcmp(op, "cas") && !rd(k, &in->ops[base + 2], bt, c, sizeof c)) return 0;
    int sz = VT_SIZE[bt];
    int id = k->tmp++;
    const char *ct = VT_C[bt];
    mem_scope_fence(k, in, 1);
    sb_printf(k->o, "%s __old%d;\n", ct, id);
    for (int vi = 0; vi < variants; vi++) {
    if (variants == 2) {
        sp = vi == 0 ? SP_SHARED : SP_GLOBAL;
        if (vi == 0) sb_printf(k->o, "if (((%s) >> 48) == 0xFFFFul) ", a);
        else sb_puts(k->o, "else ");
    }
    const char *as = sp == SP_SHARED ? "threadgroup" : "device";
    char ptr[EXL];
    if (sp == SP_SHARED) { k->uses_smem = 1; snprintf(ptr, sizeof ptr, "(__smem + (uint)(%s))", a); }
    else snprintf(ptr, sizeof ptr, "(%s)", a);
    sb_puts(k->o, "{\n");
    if (sz == 8) {
        if (!strcmp(op, "add")) {
            if (!k->p->warned_atom64) { k->p->warned_atom64 = 1; warn(k->p, "64-bit atomic add emulated with two 32-bit atomics (sum is exact; returned old value may be torn)"); }
            sb_printf(k->o, "%s atomic_uint* __lo = (%s atomic_uint*)%s; ulong __v = (ulong)(%s);\n", as, as, ptr, b);
            sb_printf(k->o,
                      "uint __ol = atomic_fetch_add_explicit(__lo, (uint)__v, memory_order_relaxed);\n"
                      "uint __cy = (uint)(((ulong)__ol + (ulong)(uint)__v) >> 32);\n"
                      "uint __oh = atomic_fetch_add_explicit(__lo + 1, (uint)(__v >> 32) + __cy, memory_order_relaxed);\n"
                      "__old%d = (%s)(((ulong)__oh << 32) | (ulong)__ol);\n",
                      id, ct);
        } else if ((!strcmp(op, "min") || !strcmp(op, "max")) && sp == SP_GLOBAL && !vt_issigned(bt) && !vt_isfloat(bt)) {
            sb_printf(k->o, "atomic_%s_explicit((device atomic_ulong*)%s, (ulong)(%s), memory_order_relaxed); __old%d = 0;\n",
                      !strcmp(op, "min") ? "min" : "max", ptr, b, id);
            if (has_result) { PERR(k->p, "PTX line %d: 64-bit atomic min/max with a result is not supported on Metal", in->line); return 0; }
        } else if (variants == 2 && sp == SP_SHARED) {
            /* shared-memory flavour of a generic atomic Metal cannot do: only reachable if
             * the program really aims it at shared memory */
            warn(k->p, "PTX line %d: 64-bit atomic '%s' through a generic pointer is only supported on device memory", in->line, op);
            sb_printf(k->o, "__old%d = 0;\n", id);
        } else {
            PERR(k->p, "PTX line %d: 64-bit atomic '%s' is not supported by Metal on this GPU", in->line, op);
            return 0;
        }
    } else if (vt == VT_F32) {
        if (!strcmp(op, "add") && sp == SP_GLOBAL) {
            sb_printf(k->o, "__old%d = atomic_fetch_add_explicit((device atomic_float*)%s, %s, memory_order_relaxed);\n", id, ptr, b);
        } else if (!strcmp(op, "exch")) {
            sb_printf(k->o, "__old%d = as_type<float>(atomic_exchange_explicit((%s atomic_uint*)%s, as_type<uint>(%s), memory_order_relaxed));\n", id, as, ptr, b);
        } else if (!strcmp(op, "add") || !strcmp(op, "min") || !strcmp(op, "max")) {
            const char *f = !strcmp(op, "add") ? "__o + __b" : !strcmp(op, "min") ? "fmin(__o, __b)" : "fmax(__o, __b)";
            sb_printf(k->o,
                      "%s atomic_uint* __p = (%s atomic_uint*)%s; float __b = %s;\n"
                      "uint __e = atomic_load_explicit(__p, memory_order_relaxed);\n"
                      "for (;;) { float __o = as_type<float>(__e); if (atomic_compare_exchange_weak_explicit(__p, &__e, as_type<uint>(%s), memory_order_relaxed, memory_order_relaxed)) break; }\n"
                      "__old%d = as_type<float>(__e);\n",
                      as, as, ptr, b, f, id);
        } else if (!strcmp(op, "cas")) {
            sb_printf(k->o,
                      "%s atomic_uint* __p = (%s atomic_uint*)%s; uint __cmp = as_type<uint>(%s), __e = __cmp;\n"
                      "while (!atomic_compare_exchange_weak_explicit(__p, &__e, as_type<uint>(%s), memory_order_relaxed, memory_order_relaxed) && __e == __cmp) {}\n"
                      "__old%d = as_type<float>(__e);\n",
                      as, as, ptr, b, c, id);
        } else { PERR(k->p, "PTX line %d: unsupported f32 atomic '%s'", in->line, op); return 0; }
    } else {
        const char *at = vt_issigned(bt) ? "atomic_int" : "atomic_uint";
        const char *it = vt_issigned(bt) ? "int" : "uint";
        if (!strcmp(op, "inc") || !strcmp(op, "dec")) {
            const char *f = !strcmp(op, "inc") ? "(__o >= __b) ? 0u : __o + 1u" : "((__o == 0u) || (__o > __b)) ? __b : __o - 1u";
            sb_printf(k->o,
                      "%s atomic_uint* __p = (%s atomic_uint*)%s; uint __b = (uint)(%s);\n"
                      "uint __e = atomic_load_explicit(__p, memory_order_relaxed);\n"
                      "for (;;) { uint __o = __e; if (atomic_compare_exchange_weak_explicit(__p, &__e, %s, memory_order_relaxed, memory_order_relaxed)) break; }\n"
                      "__old%d = (%s)__e;\n",
                      as, as, ptr, b, f, id, ct);
        } else if (!strcmp(op, "cas")) {
            sb_printf(k->o,
                      "%s atomic_uint* __p = (%s atomic_uint*)%s; uint __cmp = (uint)(%s), __e = __cmp;\n"
                      "while (!atomic_compare_exchange_weak_explicit(__p, &__e, (uint)(%s), memory_order_relaxed, memory_order_relaxed) && __e == __cmp) {}\n"
                      "__old%d = (%s)__e;\n",
                      as, as, ptr, b, c, id, ct);
        } else {
            const char *fn = !strcmp(op, "add") ? "atomic_fetch_add_explicit"
                             : !strcmp(op, "exch") ? "atomic_exchange_explicit"
                             : !strcmp(op, "min") ? "atomic_fetch_min_explicit"
                             : !strcmp(op, "max") ? "atomic_fetch_max_explicit"
                             : !strcmp(op, "and") ? "atomic_fetch_and_explicit"
                             : !strcmp(op, "or") ? "atomic_fetch_or_explicit"
                                                 : "atomic_fetch_xor_explicit";
            sb_printf(k->o, "__old%d = (%s)%s((%s %s*)%s, (%s)(%s), memory_order_relaxed);\n", id, ct, fn, as, at, ptr, it, b);
        }
    }
    sb_puts(k->o, "}\n");
    }
    mem_scope_fence(k, in, 0);
    if (has_result) {
        char r[64];
        snprintf(r, sizeof r, "__old%d", id);
        return wr(k, &in->ops[0], r, bt);
    }
    sb_printf(k->o, "(void)__old%d;\n", id);
    return 1;
}

static int is_branch(instr_t *in) { return !strcmp(in->parts[0], "bra") || !strcmp(in->parts[0], "brx"); }
static int is_ret(instr_t *in) { return !strcmp(in->parts[0], "ret") || !strcmp(in->parts[0], "exit") || !strcmp(in->parts[0], "trap"); }

static int emit_f16x2(K *k, instr_t *in, const char *base, int x2) {
    const char *vt2 = x2 == 1 ? "half2" : "float2";
    char a[EXL], b[EXL], c[EXL], r[EXL];
    #define LD2(dst, op) do { char _t[EXL]; if (!rd(k, op, VT_U32, _t, sizeof _t)) return 0; \
        if (x2 == 1) snprintf(dst, sizeof dst, "as_type<half2>(%s)", _t); \
        else snprintf(dst, sizeof dst, "__bf2_unpack(%s)", _t); } while (0)
    LD2(a, &in->ops[1]);
    if (in->nops > 2) LD2(b, &in->ops[2]);
    if (in->nops > 3) LD2(c, &in->ops[3]);
    #undef LD2
    char e[EXL];
    if (!strcmp(base, "add")) snprintf(e, sizeof e, "(%s + %s)", a, b);
    else if (!strcmp(base, "sub")) snprintf(e, sizeof e, "(%s - %s)", a, b);
    else if (!strcmp(base, "mul")) snprintf(e, sizeof e, "(%s * %s)", a, b);
    else if (!strcmp(base, "fma") || !strcmp(base, "mad")) snprintf(e, sizeof e, "fma(%s, %s, %s)", a, b, c);
    else if (!strcmp(base, "neg")) snprintf(e, sizeof e, "(-%s)", a);
    else if (!strcmp(base, "abs")) snprintf(e, sizeof e, "abs(%s)", a);
    else if (!strcmp(base, "min")) snprintf(e, sizeof e, "fmin(%s, %s)", a, b);
    else if (!strcmp(base, "max")) snprintf(e, sizeof e, "fmax(%s, %s)", a, b);
    else { PERR(k->p, "PTX line %d: unsupported packed op '%s'", in->line, in->opcode); return 0; }
    if (has_part(in, "relu")) { char t[EXL]; snprintf(t, sizeof t, "max(%s, %s(0))", e, vt2); snprintf(e, sizeof e, "%s", t); }
    if (x2 == 1) snprintf(r, sizeof r, "as_type<uint>((half2)(%s))", e);
    else snprintf(r, sizeof r, "__bf2_pack((float2)(%s))", e);
    return wr(k, &in->ops[0], r, VT_U32);
}

/* Emit one non-terminator instruction. */
static int emit_instr(K *k, instr_t *in) {
    const char *op = in->parts[0];
    sb_t *o = k->o;
    int x2 = 0;
    int vt = type_part(in, -1, &x2);
    char a[EXL], b[EXL], c[EXL], r[EXL];

    if (in->pred[0]) {
        operand_t pr;
        memset(&pr, 0, sizeof pr);
        pr.kind = OP_REG;
        snprintf(pr.name, sizeof pr.name, "%s", in->pred);
        if (!rd(k, &pr, VT_PRED, a, sizeof a)) return 0;
        sb_printf(o, "if (%s%s) {\n", in->pred_not ? "!" : "", a);
    }
    int ok = 1;
#define R(i, t, buf) do { if (!rd(k, &in->ops[i], (t), buf, sizeof buf)) return 0; } while (0)
#define RN(i, t, buf) do { if (!rdn(k, &in->ops[i], (t), buf, sizeof buf)) return 0; } while (0)
#define NEED(nn) do { if (in->nops < (nn)) { PERR(k->p, "PTX line %d: '%s' needs %d operands", in->line, in->opcode, nn); return 0; } } while (0)

    if (!strcmp(op, "mov")) {
        NEED(2);
        if (in->ops[0].kind == OP_VEC) { /* unpack */
            int n = in->ops[0].nelems, es = VT_SIZE[vt] / n;
            int ut = vt_unsigned_of_size(VT_SIZE[vt]), et = vt_unsigned_of_size(es);
            R(1, ut, a);
            int id = k->tmp++;
            sb_printf(o, "{ %s __u%d = %s;\n", VT_C[ut], id, a);
            for (int e = 0; e < n; e++) {
                snprintf(r, sizeof r, "((%s)(__u%d >> %d))", VT_C[et], id, e * es * 8);
                if (in->ops[0].elems[e].kind != OP_SINK && !wr(k, &in->ops[0].elems[e], r, et)) return 0;
            }
            sb_puts(o, "}\n");
        } else if (in->ops[1].kind == OP_VEC) { /* pack */
            int n = in->ops[1].nelems, es = VT_SIZE[vt] / n;
            int ut = vt_unsigned_of_size(VT_SIZE[vt]), et = vt_unsigned_of_size(es);
            sb_t e = {0};
            sb_puts(&e, "(");
            for (int i = 0; i < n; i++) {
                if (!rd(k, &in->ops[1].elems[i], et, a, sizeof a)) return 0;
                sb_printf(&e, "%s((%s)(%s) << %d)", i ? " | " : "", VT_C[ut], a, i * es * 8);
            }
            sb_puts(&e, ")");
            ok = wr(k, &in->ops[0], e.p, ut);
            free(e.p);
        } else {
            R(1, vt, a);
            ok = wr(k, &in->ops[0], a, vt);
        }
    } else if (!strcmp(op, "ld") || !strcmp(op, "ldu")) {
        NEED(2);
        ok = emit_ldst(k, in, 0);
    } else if (!strcmp(op, "st")) {
        NEED(2);
        ok = emit_ldst(k, in, 1);
    } else if (!strcmp(op, "cvta")) {
        NEED(2);
        int to = has_part(in, "to");
        int sp = space_of(in);
        int t = vt < 0 ? VT_U64 : vt;
        R(1, VT_U64, a);
        if (sp == SP_GLOBAL || sp == SP_CONST || sp == SP_GENERIC) snprintf(r, sizeof r, "%s", a);
        else if (to) snprintf(r, sizeof r, "((%s) & 0xFFFFFFFFul)", a);
        else {
            unsigned long long tag = sp == SP_SHARED ? TAG_SHARED : sp == SP_LOCAL ? TAG_LOCAL : TAG_PARAM;
            snprintf(r, sizeof r, "((%s) | 0x%llxul)", a, tag << 48);
        }
        ok = wr(k, &in->ops[0], r, VT_U64);
        (void)t;
    } else if (!strcmp(op, "cvt")) {
        NEED(2);
        ok = emit_cvt(k, in);
    } else if (x2 && (!strcmp(op, "add") || !strcmp(op, "sub") || !strcmp(op, "mul") || !strcmp(op, "fma") || !strcmp(op, "neg") ||
                      !strcmp(op, "abs") || !strcmp(op, "min") || !strcmp(op, "max") || !strcmp(op, "mad"))) {
        ok = emit_f16x2(k, in, op, x2);
    } else if (!strcmp(op, "add") || !strcmp(op, "sub") || !strcmp(op, "addc") || !strcmp(op, "subc")) {
        NEED(3);
        int sub = op[0] == 's';
        if (vt_isfloat(vt)) {
            RN(1, vt, a); RN(2, vt, b);
            snprintf(r, sizeof r, "(%s %c %s)", a, sub ? '-' : '+', b);
            if (has_part(in, "sat")) { char t[EXL]; snprintf(t, sizeof t, "saturate(%s)", r); snprintf(r, sizeof r, "%s", t); }
            ok = wrn(k, &in->ops[0], r, vt);
        } else if (has_part(in, "cc") || !strcmp(op, "addc") || !strcmp(op, "subc")) {
            int carry_in = (!strcmp(op, "addc") || !strcmp(op, "subc"));
            int ut = vt_uns(vt);
            int wide = VT_SIZE[ut] == 8;
            R(1, ut, a); R(2, ut, b);
            int id = k->tmp++;
            if (!wide) {
                if (!sub) sb_printf(o, "ulong __t%d = (ulong)%s + (ulong)%s%s;\n", id, a, b, carry_in ? " + (ulong)__cc" : "");
                else sb_printf(o, "ulong __t%d = (ulong)%s - (ulong)%s%s;\n", id, a, b, carry_in ? " - (ulong)__cc" : "");
                snprintf(r, sizeof r, "((uint)__t%d)", id);
                if (has_part(in, "cc")) sb_printf(o, "__cc = (uint)((__t%d >> 32) & 1ul);\n", id);
                ok = wr(k, &in->ops[0], r, VT_U32);
            } else {
                sb_printf(o, "ulong __a%d = %s, __b%d = %s; ulong __ci%d = %s;\n", id, a, id, b, id, carry_in ? "(ulong)__cc" : "0ul");
                if (!sub) sb_printf(o, "ulong __t%d = __a%d + __b%d + __ci%d; uint __c%d = (__t%d < __a%d || (__ci%d != 0ul && __t%d == __a%d)) ? 1u : 0u;\n", id, id, id, id, id, id, id, id, id, id);
                else sb_printf(o, "ulong __t%d = __a%d - __b%d - __ci%d; uint __c%d = (__a%d < __b%d || (__ci%d != 0ul && __a%d == __b%d)) ? 1u : 0u;\n", id, id, id, id, id, id, id, id, id, id);
                if (has_part(in, "cc")) sb_printf(o, "__cc = __c%d;\n", id);
                snprintf(r, sizeof r, "__t%d", id);
                ok = wr(k, &in->ops[0], r, VT_U64);
            }
        } else if (has_part(in, "sat") && vt == VT_S32) {
            R(1, VT_S32, a); R(2, VT_S32, b);
            snprintf(r, sizeof r, "%s(%s, %s)", sub ? "subsat" : "addsat", a, b);
            ok = wr(k, &in->ops[0], r, VT_S32);
        } else {
            int ut = vt_uns(vt);
            R(1, ut, a); R(2, ut, b);
            snprintf(r, sizeof r, "((%s)(%s %c %s))", VT_C[ut], a, sub ? '-' : '+', b);
            ok = wr(k, &in->ops[0], r, ut);
        }
    } else if (!strcmp(op, "mul") || !strcmp(op, "mul24")) {
        NEED(3);
        if (vt_isfloat(vt)) {
            RN(1, vt, a); RN(2, vt, b);
            snprintf(r, sizeof r, "(%s * %s)", a, b);
            if (has_part(in, "sat")) { char t[EXL]; snprintf(t, sizeof t, "saturate(%s)", r); snprintf(r, sizeof r, "%s", t); }
            ok = wrn(k, &in->ops[0], r, vt);
        } else if (!strcmp(op, "mul24")) {
            /* exact PTX semantics: 48-bit product of the (sign/zero-extended) low 24 bits */
            R(1, VT_U32, a); R(2, VT_U32, b);
            snprintf(r, sizeof r, "__mul24_%s_%s(%s, %s)", vt_issigned(vt) ? "s" : "u", has_part(in, "hi") ? "hi" : "lo", a, b);
            ok = wr(k, &in->ops[0], r, VT_U32);
        } else if (has_part(in, "wide")) {
            int wt = vt_issigned(vt) ? vt_signed_of_size(VT_SIZE[vt] * 2) : vt_unsigned_of_size(VT_SIZE[vt] * 2);
            R(1, vt, a); R(2, vt, b);
            snprintf(r, sizeof r, "((%s)(%s) * (%s)(%s))", VT_C[wt], a, VT_C[wt], b);
            ok = wr(k, &in->ops[0], r, wt);
        } else if (has_part(in, "hi")) {
            R(1, vt, a); R(2, vt, b);
            snprintf(r, sizeof r, "mulhi(%s, %s)", a, b);
            ok = wr(k, &in->ops[0], r, vt);
        } else {
            int ut = vt_uns(vt);
            R(1, ut, a); R(2, ut, b);
            snprintf(r, sizeof r, "((%s)(%s * %s))", VT_C[ut], a, b);
            ok = wr(k, &in->ops[0], r, ut);
        }
    } else if (!strcmp(op, "mad") || !strcmp(op, "fma") || !strcmp(op, "mad24")) {
        NEED(4);
        if (vt_isfloat(vt)) {
            RN(1, vt, a); RN(2, vt, b); RN(3, vt, c);
            snprintf(r, sizeof r, "fma(%s, %s, %s)", a, b, c);
            if (has_part(in, "sat")) { char t[EXL]; snprintf(t, sizeof t, "saturate(%s)", r); snprintf(r, sizeof r, "%s", t); }
            ok = wrn(k, &in->ops[0], r, vt);
        } else if (has_part(in, "wide")) {
            int wt = vt_issigned(vt) ? vt_signed_of_size(VT_SIZE[vt] * 2) : vt_unsigned_of_size(VT_SIZE[vt] * 2);
            R(1, vt, a); R(2, vt, b); R(3, wt, c);
            snprintf(r, sizeof r, "((%s)((%s)(%s) * (%s)(%s) + %s))", VT_C[wt], VT_C[wt], a, VT_C[wt], b, c);
            ok = wr(k, &in->ops[0], r, wt);
        } else if (has_part(in, "hi")) {
            R(1, vt, a); R(2, vt, b); R(3, vt, c);
            snprintf(r, sizeof r, "((%s)(mulhi(%s, %s) + %s))", VT_C[vt], a, b, c);
            ok = wr(k, &in->ops[0], r, vt);
        } else if (!strcmp(op, "mad24")) {
            R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c);
            snprintf(r, sizeof r, "((uint)(__mul24_%s_%s(%s, %s) + %s))", vt_issigned(vt) ? "s" : "u", has_part(in, "hi") ? "hi" : "lo", a, b, c);
            ok = wr(k, &in->ops[0], r, VT_U32);
        } else {
            int ut = vt_uns(vt);
            R(1, ut, a); R(2, ut, b); R(3, ut, c);
            snprintf(r, sizeof r, "((%s)(%s * %s + %s))", VT_C[ut], a, b, c);
            ok = wr(k, &in->ops[0], r, ut);
        }
    } else if (!strcmp(op, "div")) {
        NEED(3);
        if (vt_isfloat(vt)) {
            RN(1, vt, a); RN(2, vt, b);
            if (vt == VT_F32 && (has_part(in, "approx") || has_part(in, "full"))) snprintf(r, sizeof r, "fast::divide(%s, %s)", a, b);
            else if (vt == VT_F32 || vt == VT_F64) snprintf(r, sizeof r, "precise::divide(%s, %s)", a, b);
            else snprintf(r, sizeof r, "(%s / %s)", a, b);
            ok = wrn(k, &in->ops[0], r, vt);
        } else {
            R(1, vt, a); R(2, vt, b);
            snprintf(r, sizeof r, "((%s) != 0 ? (%s)(%s / %s) : (%s)(-1))", b, VT_C[vt], a, b, VT_C[vt]);
            ok = wr(k, &in->ops[0], r, vt);
        }
    } else if (!strcmp(op, "rem")) {
        NEED(3);
        R(1, vt, a); R(2, vt, b);
        snprintf(r, sizeof r, "((%s) != 0 ? (%s)(%s %% %s) : (%s))", b, VT_C[vt], a, b, a);
        ok = wr(k, &in->ops[0], r, vt);
    } else if (!strcmp(op, "min") || !strcmp(op, "max")) {
        NEED(3);
        int mx = op[1] == 'a';
        if (vt_isfloat(vt)) {
            RN(1, vt, a); RN(2, vt, b);
            if (has_part(in, "NaN")) snprintf(r, sizeof r, "((isnan(%s) || isnan(%s)) ? NAN : %s(%s, %s))", a, b, mx ? "fmax" : "fmin", a, b);
            else snprintf(r, sizeof r, "%s(%s, %s)", mx ? "fmax" : "fmin", a, b);
            ok = wrn(k, &in->ops[0], r, vt);
        } else {
            R(1, vt, a); R(2, vt, b);
            snprintf(r, sizeof r, "%s(%s, %s)", mx ? "max" : "min", a, b);
            if (has_part(in, "relu")) { char t[EXL]; snprintf(t, sizeof t, "max(%s, (%s)0)", r, VT_C[vt]); snprintf(r, sizeof r, "%s", t); }
            ok = wr(k, &in->ops[0], r, vt);
        }
    } else if (!strcmp(op, "abs") || !strcmp(op, "neg")) {
        NEED(2);
        int ab = op[0] == 'a';
        if (vt_isfloat(vt)) {
            RN(1, vt, a);
            snprintf(r, sizeof r, ab ? "fabs(%s)" : "(-(%s))", a);
            ok = wrn(k, &in->ops[0], r, vt);
        } else {
            if (ab) { R(1, vt_signed_of_size(VT_SIZE[vt]), a); snprintf(r, sizeof r, "((%s)abs(%s))", VT_C[vt_uns(vt)], a); }
            else { R(1, vt_uns(vt), a); snprintf(r, sizeof r, "((%s)(0 - %s))", VT_C[vt_uns(vt)], a); }
            ok = wr(k, &in->ops[0], r, vt_uns(vt));
        }
    } else if (!strcmp(op, "and") || !strcmp(op, "or") || !strcmp(op, "xor")) {
        NEED(3);
        if (vt == VT_PRED) {
            R(1, VT_PRED, a); R(2, VT_PRED, b);
            snprintf(r, sizeof r, "(%s %s %s)", a, op[0] == 'a' ? "&&" : op[0] == 'o' ? "||" : "!=", b);
            ok = wr(k, &in->ops[0], r, VT_PRED);
        } else {
            int ut = vt_uns(vt);
            R(1, ut, a); R(2, ut, b);
            snprintf(r, sizeof r, "((%s)(%s %c %s))", VT_C[ut], a, op[0] == 'a' ? '&' : op[0] == 'o' ? '|' : '^', b);
            ok = wr(k, &in->ops[0], r, ut);
        }
    } else if (!strcmp(op, "not") || !strcmp(op, "cnot")) {
        NEED(2);
        if (vt == VT_PRED) { R(1, VT_PRED, a); snprintf(r, sizeof r, "(!%s)", a); ok = wr(k, &in->ops[0], r, VT_PRED); }
        else {
            int ut = vt_uns(vt);
            R(1, ut, a);
            if (op[0] == 'c') snprintf(r, sizeof r, "((%s)((%s) == 0 ? 1 : 0))", VT_C[ut], a);
            else snprintf(r, sizeof r, "((%s)(~%s))", VT_C[ut], a);
            ok = wr(k, &in->ops[0], r, ut);
        }
    } else if (!strcmp(op, "shl") || !strcmp(op, "shr")) {
        NEED(3);
        int bits = VT_SIZE[vt] * 8;
        R(2, VT_U32, b);
        if (op[2] == 'l' || !vt_issigned(vt)) {
            int ut = vt_uns(vt);
            R(1, ut, a);
            snprintf(r, sizeof r, "((%s) < %du ? (%s)(%s %s (%s)) : (%s)0)", b, bits, VT_C[ut], a, op[2] == 'l' ? "<<" : ">>", b, VT_C[ut]);
            ok = wr(k, &in->ops[0], r, ut);
        } else {
            R(1, vt, a);
            snprintf(r, sizeof r, "((%s)(%s >> min((%s), %du)))", VT_C[vt], a, b, bits - 1);
            ok = wr(k, &in->ops[0], r, vt);
        }
    } else if (!strcmp(op, "shf")) {
        NEED(4);
        R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c);
        snprintf(r, sizeof r, "__shf(%s, %s, %s, %d, %d)", a, b, c, has_part(in, "l") ? 1 : 0, has_part(in, "clamp") ? 1 : 0);
        ok = wr(k, &in->ops[0], r, VT_U32);
    } else if (!strcmp(op, "popc") || !strcmp(op, "clz") || !strcmp(op, "brev") || !strcmp(op, "bfind")) {
        NEED(2);
        if (!strcmp(op, "bfind")) {
            R(1, vt, a);
            snprintf(r, sizeof r, "__bfind%s(%s, %d)", VT_SIZE[vt] == 8 ? "64" : "32", a, has_part(in, "shiftamt"));
            ok = wr(k, &in->ops[0], r, VT_U32);
        } else {
            int ut = vt_uns(vt);
            R(1, ut, a);
            if (op[0] == 'b') { snprintf(r, sizeof r, "reverse_bits(%s)", a); ok = wr(k, &in->ops[0], r, ut); }
            else { snprintf(r, sizeof r, "((uint)%s(%s))", op[0] == 'p' ? "popcount" : "clz", a); ok = wr(k, &in->ops[0], r, VT_U32); }
        }
    } else if (!strcmp(op, "bfe")) {
        NEED(4);
        R(1, vt, a); R(2, VT_U32, b); R(3, VT_U32, c);
        if (VT_SIZE[vt] == 8) snprintf(r, sizeof r, "__bfe64(%s, %s, %s, %d)", a, b, c, vt_issigned(vt));
        else snprintf(r, sizeof r, "__bfe32(%s, %s, %s, %d)", a, b, c, vt_issigned(vt));
        ok = wr(k, &in->ops[0], r, vt_uns(vt));
    } else if (!strcmp(op, "bfi")) {
        NEED(5);
        char d[EXL];
        int ut = vt_uns(vt);
        R(1, ut, a); R(2, ut, b); R(3, VT_U32, c); R(4, VT_U32, d);
        snprintf(r, sizeof r, "__bfi%s(%s, %s, %s, %s)", VT_SIZE[vt] == 8 ? "64" : "32", a, b, c, d);
        ok = wr(k, &in->ops[0], r, ut);
    } else if (!strcmp(op, "prmt")) {
        NEED(4);
        R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c);
        int mode = 0;
        const char *modes[] = {"", "f4e", "b4e", "rc8", "ecl", "ecr", "rc16"};
        for (int i = 1; i < 7; i++) if (has_part(in, modes[i])) mode = i;
        snprintf(r, sizeof r, "__prmt(%s, %s, %s, %d)", a, b, c, mode);
        ok = wr(k, &in->ops[0], r, VT_U32);
    } else if (!strcmp(op, "lop3")) {
        NEED(5);
        char d[EXL];
        R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c); R(4, VT_U32, d);
        snprintf(r, sizeof r, "__lop3(%s, %s, %s, %s)", a, b, c, d);
        ok = wr(k, &in->ops[0], r, VT_U32);
    } else if (!strcmp(op, "dp4a") || !strcmp(op, "dp2a")) {
        NEED(4);
        int ta = type_part(in, 0, NULL), tb = type_part(in, 1, NULL);
        R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c);
        if (op[2] == '4') snprintf(r, sizeof r, "__dp4a(%s, %s, %s, %d, %d)", a, b, c, vt_issigned(ta), vt_issigned(tb));
        else snprintf(r, sizeof r, "__dp2a(%s, %s, %s, %d, %d, %d)", a, b, c, vt_issigned(ta), vt_issigned(tb), has_part(in, "hi"));
        ok = wr(k, &in->ops[0], r, VT_U32);
    } else if (!strcmp(op, "sad")) {
        NEED(4);
        R(1, vt, a); R(2, vt, b); R(3, vt_uns(vt), c);
        snprintf(r, sizeof r, "((%s)(absdiff(%s, %s) + %s))", VT_C[vt_uns(vt)], a, b, c);
        ok = wr(k, &in->ops[0], r, vt_uns(vt));
    } else if (!strcmp(op, "setp")) {
        NEED(3);
        int st = vt;
        if (!compare_expr(k, in, st, &in->ops[1], &in->ops[2], c, sizeof c)) return 0;
        const char *bo = bool_op(in);
        int id = k->tmp++;
        sb_printf(o, "{ bool __cmp%d = %s;\n", id, c);
        if (bo) {
            NEED(4);
            R(3, VT_PRED, b);
            snprintf(r, sizeof r, "(__cmp%d %s %s)", id, bo, b);
            if (!wr(k, &in->ops[0], r, VT_PRED)) return 0;
            if (in->ops[0].name2[0]) { snprintf(r, sizeof r, "((!__cmp%d) %s %s)", id, bo, b); if (!wr_named(k, in->ops[0].name2, r, VT_PRED)) return 0; }
        } else {
            snprintf(r, sizeof r, "__cmp%d", id);
            if (!wr(k, &in->ops[0], r, VT_PRED)) return 0;
            if (in->ops[0].name2[0]) { snprintf(r, sizeof r, "(!__cmp%d)", id); if (!wr_named(k, in->ops[0].name2, r, VT_PRED)) return 0; }
        }
        sb_puts(o, "}\n");
    } else if (!strcmp(op, "set")) {
        NEED(3);
        int dt = type_part(in, 0, NULL), st = type_part(in, 1, NULL);
        if (!compare_expr(k, in, st, &in->ops[1], &in->ops[2], c, sizeof c)) return 0;
        const char *bo = bool_op(in);
        if (bo) { NEED(4); R(3, VT_PRED, b); snprintf(a, sizeof a, "(%s %s %s)", c, bo, b); snprintf(c, sizeof c, "%s", a); }
        if (vt_isfloat(dt)) snprintf(r, sizeof r, "((%s) ? 1.0f : 0.0f)", c);
        else snprintf(r, sizeof r, "((%s) ? 0xFFFFFFFFu : 0u)", c);
        ok = wr(k, &in->ops[0], r, vt_isfloat(dt) ? VT_F32 : VT_U32);
    } else if (!strcmp(op, "selp")) {
        NEED(4);
        R(1, vt, a); R(2, vt, b); R(3, VT_PRED, c);
        snprintf(r, sizeof r, "((%s) ? %s : %s)", c, a, b);
        ok = wr(k, &in->ops[0], r, vt);
    } else if (!strcmp(op, "slct")) {
        NEED(4);
        int dt = type_part(in, 0, NULL), ct = type_part(in, 1, NULL);
        R(1, dt, a); R(2, dt, b); R(3, ct, c);
        snprintf(r, sizeof r, "((%s) >= 0 ? %s : %s)", c, a, b);
        ok = wr(k, &in->ops[0], r, dt);
    } else if (!strcmp(op, "rcp") || !strcmp(op, "sqrt") || !strcmp(op, "rsqrt") || !strcmp(op, "sin") || !strcmp(op, "cos") ||
               !strcmp(op, "lg2") || !strcmp(op, "ex2") || !strcmp(op, "tanh")) {
        NEED(2);
        RN(1, vt, a);
        int approx = has_part(in, "approx");
        int f32 = vt_compute(vt) == VT_F32 || vt == VT_F64;
        if (!strcmp(op, "rcp")) snprintf(r, sizeof r, approx && f32 ? "fast::divide(1.0f, %s)" : f32 ? "precise::divide(1.0f, %s)" : "(1.0h / %s)", a);
        else if (!strcmp(op, "sqrt")) snprintf(r, sizeof r, approx && f32 ? "fast::sqrt(%s)" : f32 ? "precise::sqrt(%s)" : "sqrt(%s)", a);
        else if (!strcmp(op, "rsqrt")) snprintf(r, sizeof r, f32 ? "fast::rsqrt(%s)" : "rsqrt(%s)", a);
        else if (!strcmp(op, "sin")) snprintf(r, sizeof r, f32 ? "fast::sin(%s)" : "sin(%s)", a);
        else if (!strcmp(op, "cos")) snprintf(r, sizeof r, f32 ? "fast::cos(%s)" : "cos(%s)", a);
        else if (!strcmp(op, "lg2")) snprintf(r, sizeof r, f32 ? "fast::log2(%s)" : "log2(%s)", a);
        else if (!strcmp(op, "ex2")) snprintf(r, sizeof r, f32 ? "fast::exp2(%s)" : "exp2(%s)", a);
        else snprintf(r, sizeof r, f32 ? "fast::tanh(%s)" : "tanh(%s)", a);
        ok = wrn(k, &in->ops[0], r, vt);
    } else if (!strcmp(op, "copysign")) {
        NEED(3);
        RN(1, vt, a); RN(2, vt, b);
        snprintf(r, sizeof r, "copysign(%s, %s)", b, a);
        ok = wrn(k, &in->ops[0], r, vt);
    } else if (!strcmp(op, "testp")) {
        NEED(2);
        RN(1, vt, a);
        const char *q = in->parts[1];
        if (!strcmp(q, "finite")) snprintf(r, sizeof r, "isfinite(%s)", a);
        else if (!strcmp(q, "infinite")) snprintf(r, sizeof r, "isinf(%s)", a);
        else if (!strcmp(q, "number")) snprintf(r, sizeof r, "(!isnan(%s))", a);
        else if (!strcmp(q, "notanumber")) snprintf(r, sizeof r, "isnan(%s)", a);
        else if (!strcmp(q, "normal")) snprintf(r, sizeof r, "isnormal(%s)", a);
        else snprintf(r, sizeof r, "(fpclassify(%s) == FP_SUBNORMAL)", a);
        ok = wr(k, &in->ops[0], r, VT_PRED);
    } else if (!strcmp(op, "bar") || !strcmp(op, "barrier")) {
        const char *q = in->np > 1 ? in->parts[1] : "sync";
        if (!strcmp(q, "cta")) q = in->np > 2 ? in->parts[2] : "sync";
        if (!strcmp(q, "warp")) {
            sb_puts(o, "simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n");
        } else if (!strcmp(q, "red")) {
            /* bar.red.{popc,and,or}.{u32,pred} d, a{, b}, {!}c */
            k->uses_bar_red = 1;
            k->uses_smem = 1;
            const char *rop = in->parts[2];
            operand_t *pc = &in->ops[in->nops - 1];
            R(in->nops - 1, VT_PRED, c);
            (void)pc;
            int id = k->tmp++;
            sb_printf(o,
                      "threadgroup atomic_uint* __rs%d = (threadgroup atomic_uint*)(__smem + %uu);\n"
                      "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
                      "if (__tid.x == 0 && __tid.y == 0 && __tid.z == 0) atomic_store_explicit(__rs%d, %su, memory_order_relaxed);\n"
                      "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n",
                      id, k->red_off, id, !strcmp(rop, "and") ? "1" : "0");
            if (!strcmp(rop, "popc")) sb_printf(o, "if (%s) atomic_fetch_add_explicit(__rs%d, 1u, memory_order_relaxed);\n", c, id);
            else if (!strcmp(rop, "and")) sb_printf(o, "if (!(%s)) atomic_store_explicit(__rs%d, 0u, memory_order_relaxed);\n", c, id);
            else sb_printf(o, "if (%s) atomic_store_explicit(__rs%d, 1u, memory_order_relaxed);\n", c, id);
            sb_printf(o, "threadgroup_barrier(mem_flags::mem_threadgroup);\nuint __rv%d = atomic_load_explicit(__rs%d, memory_order_relaxed);\nthreadgroup_barrier(mem_flags::mem_threadgroup);\n", id, id);
            snprintf(r, sizeof r, "__rv%d", id);
            ok = wr(k, &in->ops[0], r, VT_U32);
        } else if (!strcmp(q, "arrive")) {
            PERR(k->p, "PTX line %d: bar.arrive (split barriers) is not supported", in->line);
            return 0;
        } else {
            if (in->nops > 1 && !k->p->warned_namedbar) { k->p->warned_namedbar = 1; warn(k->p, "named barriers with a thread count are mapped to a full threadgroup barrier"); }
            sb_puts(o, "threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);\n");
        }
    } else if (!strcmp(op, "membar") || !strcmp(op, "fence")) {
        if (has_part(in, "proxy")) { /* proxy fences: no-op on Metal */ }
        else if (has_part(in, "cta")) sb_puts(o, "atomic_thread_fence(mem_flags::mem_threadgroup | mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);\n");
        else sb_puts(o, "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);\n");
    } else if (!strcmp(op, "atom")) {
        ok = emit_atomic(k, in, 1);
    } else if (!strcmp(op, "red")) {
        ok = emit_atomic(k, in, 0);
    } else if (!strcmp(op, "shfl")) {
        NEED(4);
        int mode = has_part(in, "up") ? 0 : has_part(in, "down") ? 1 : has_part(in, "bfly") ? 2 : 3;
        R(1, VT_U32, a); R(2, VT_U32, b); R(3, VT_U32, c);
        int id = k->tmp++;
        sb_printf(o, "{ bool __sp%d; uint __sv%d = __mc_shfl(%s, %s, %s, %du, __lane, &__sp%d);\n", id, id, a, b, c, mode, id);
        snprintf(r, sizeof r, "__sv%d", id);
        if (!wr(k, &in->ops[0], r, VT_U32)) return 0;
        if (in->ops[0].name2[0]) { snprintf(r, sizeof r, "__sp%d", id); if (!wr_named(k, in->ops[0].name2, r, VT_PRED)) return 0; }
        sb_puts(o, "}\n");
    } else if (!strcmp(op, "vote")) {
        NEED(2);
        R(1, VT_PRED, a);
        if (has_part(in, "ballot")) { snprintf(r, sizeof r, "((uint)(ulong)(simd_vote::vote_t)simd_ballot(%s))", a); ok = wr(k, &in->ops[0], r, VT_U32); }
        else if (has_part(in, "all")) { snprintf(r, sizeof r, "simd_all(%s)", a); ok = wr(k, &in->ops[0], r, VT_PRED); }
        else if (has_part(in, "any")) { snprintf(r, sizeof r, "simd_any(%s)", a); ok = wr(k, &in->ops[0], r, VT_PRED); }
        else { snprintf(r, sizeof r, "(simd_all(%s) || simd_all(!(%s)))", a, a); ok = wr(k, &in->ops[0], r, VT_PRED); }
    } else if (!strcmp(op, "activemask")) {
        NEED(1);
        ok = wr(k, &in->ops[0], "((uint)(ulong)(simd_vote::vote_t)simd_active_threads_mask())", VT_U32);
    } else if (!strcmp(op, "redux")) {
        NEED(2);
        int t = vt_issigned(vt) ? VT_S32 : VT_U32;
        R(1, t, a);
        const char *fn = has_part(in, "add") ? "simd_sum" : has_part(in, "min") ? "simd_min" : has_part(in, "max") ? "simd_max"
                       : has_part(in, "and") ? "simd_and" : has_part(in, "or") ? "simd_or" : "simd_xor";
        snprintf(r, sizeof r, "%s(%s)", fn, a);
        ok = wr(k, &in->ops[0], r, t);
    } else if (!strcmp(op, "elect")) {
        NEED(1);
        ok = wr(k, &in->ops[0], "simd_broadcast_first(__lane)", VT_U32);
        if (ok && in->ops[0].name2[0]) ok = wr_named(k, in->ops[0].name2, "simd_is_first()", VT_PRED);
    } else if (!strcmp(op, "cp")) {
        /* cp.async.ca.shared.global [dst], [src], cp-size{, src-size} : executed synchronously */
        if (has_part(in, "commit_group") || has_part(in, "wait_group") || has_part(in, "wait_all")) {
            /* copies are synchronous: nothing to wait for */
        } else if (has_part(in, "async") && !has_part(in, "bulk") && !has_part(in, "mbarrier")) {
            NEED(3);
            int sd = SP_SHARED, ss = SP_GLOBAL;
            if (!addr_expr(k, &in->ops[0], &sd, a, sizeof a) || !addr_expr(k, &in->ops[1], &ss, b, sizeof b)) return 0;
            R(2, VT_U32, c);
            char srcsz[EXL];
            if (in->nops > 3 && in->ops[3].kind != OP_ADDR) { if (!rd(k, &in->ops[3], VT_U32, srcsz, sizeof srcsz)) return 0; }
            else snprintf(srcsz, sizeof srcsz, "%s", c);
            k->uses_smem = 1;
            sb_printf(o, "{ threadgroup uchar* __cd = __smem + (uint)(%s); device const uchar* __cs = (device const uchar*)(%s); uint __cn = %s, __cz = %s;\n"
                         "  for (uint __i = 0; __i < __cn; __i++) __cd[__i] = __i < __cz ? __cs[__i] : (uchar)0; }\n",
                      a, b, c, srcsz);
        } else {
            PERR(k->p, "PTX line %d: '%s' is not supported", in->line, in->opcode);
            return 0;
        }
    } else if (!strcmp(op, "prefetch") || !strcmp(op, "prefetchu") || !strcmp(op, "discard") || !strcmp(op, "applypriority") ||
               !strcmp(op, "nanosleep") || !strcmp(op, "pmevent") || !strcmp(op, "brkpt")) {
        /* performance hints: no-op */
    } else if (!strcmp(op, "createpolicy")) {
        NEED(1);
        ok = wr(k, &in->ops[0], "0ul", VT_U64);
    } else if (!strcmp(op, "call")) {
        const char *callee = NULL;
        for (int i = 0; i < in->nops; i++) if (in->ops[i].kind == OP_SYM) { callee = in->ops[i].name; break; }
        if (callee && (!strcmp(callee, "vprintf") || !strcmp(callee, "__assertfail") || !strcmp(callee, "__assert_fail"))) {
            if (!k->p->warned_printf) { k->p->warned_printf = 1; warn(k->p, "device-side printf/assert calls are ignored"); }
        } else {
            PERR(k->p, "PTX line %d: call to device function '%s' is not supported (compile with inlining, e.g. -O3 / __forceinline__)", in->line, callee ? callee : "?");
            return 0;
        }
    } else if (!strcmp(op, "mma") || !strcmp(op, "wmma") || !strcmp(op, "ldmatrix") || !strcmp(op, "stmatrix") || !strcmp(op, "wgmma")) {
        PERR(k->p, "PTX line %d: tensor-core instruction '%s' is not supported yet", in->line, in->opcode);
        return 0;
    } else if (!strncmp(op, "tex", 3) || !strncmp(op, "tld4", 4) || !strncmp(op, "suld", 4) || !strncmp(op, "sust", 4) || !strcmp(op, "txq") || !strcmp(op, "suq")) {
        PERR(k->p, "PTX line %d: texture/surface instruction '%s' is not supported", in->line, in->opcode);
        return 0;
    } else if (is_ret(in)) {
        sb_puts(o, "return;\n");
    } else {
        PERR(k->p, "PTX line %d: unsupported instruction '%s'", in->line, in->opcode);
        return 0;
    }
#undef R
#undef RN
#undef NEED
    if (in->pred[0]) sb_puts(o, "}\n");
    return ok;
}

/* ───────────────────────────── CFG ───────────────────────────── */

typedef struct {
    int start, end; /* stmt range [start,end) — instructions only */
    int succ[2], nsucc;
    int term;       /* stmt index of terminator or -1 */
    int fall;       /* next block in program order or -1 */
    int id;         /* emission id (loop-aware RPO) */
    int reach;      /* reachable from the entry */
} block_t;

static int label_block(char (*labels)[MC_MAX_NAME], int *lblk, int nl, const char *name) {
    for (int i = 0; i < nl; i++) if (!strcmp(labels[i], name)) return lblk[i];
    return -1;
}

typedef struct {
    block_t *b;
    int nb;
    int *state; /* 0 unvisited 1 on stack 2 done */
    int *post;
    int npost;
    int (*backedge)[2];
    int nback, capback;
    unsigned char *inloop; /* [nloops][nb] */
    int nloops;
} cfg_t;

static void dfs1(cfg_t *g, int u) {
    g->state[u] = 1;
    for (int i = 0; i < g->b[u].nsucc; i++) {
        int v = g->b[u].succ[i];
        if (g->state[v] == 0) dfs1(g, v);
        else if (g->state[v] == 1) {
            if (g->nback == g->capback) { g->capback = g->capback ? g->capback * 2 : 8; g->backedge = realloc(g->backedge, (size_t)g->capback * sizeof *g->backedge); }
            g->backedge[g->nback][0] = u;
            g->backedge[g->nback][1] = v;
            g->nback++;
        }
    }
    g->state[u] = 2;
}

static int common_loops(cfg_t *g, int u, int v) {
    int c = 0;
    for (int l = 0; l < g->nloops; l++)
        if (g->inloop[(size_t)l * g->nb + u] && g->inloop[(size_t)l * g->nb + v]) c++;
    return c;
}

static void dfs2(cfg_t *g, int u) {
    g->state[u] = 1;
    int s[2], ns = g->b[u].nsucc;
    for (int i = 0; i < ns; i++) s[i] = g->b[u].succ[i];
    /* visit loop exits first so that in-loop blocks get lower RPO numbers */
    if (ns == 2 && common_loops(g, u, s[0]) > common_loops(g, u, s[1])) { int t = s[0]; s[0] = s[1]; s[1] = t; }
    for (int i = 0; i < ns; i++)
        if (!g->state[s[i]]) dfs2(g, s[i]);
    g->post[g->npost++] = u;
}

/* ───────────────────────────── prelude ───────────────────────────── */

static const char *PRELUDE =
    "// Generated by MetaCuda (" MC_TRANSLATOR_VERSION ") — PTX translated to Metal Shading Language.\n"
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "#ifndef MC_NUM_SM\n#define MC_NUM_SM 1u\n#endif\n"
    "#pragma clang diagnostic ignored \"-Wunused-variable\"\n"
    "#pragma clang diagnostic ignored \"-Wunused-but-set-variable\"\n"
    "#pragma clang diagnostic ignored \"-Wunused-function\"\n"
    "#pragma clang diagnostic ignored \"-Wunused-label\"\n"
    "#pragma clang diagnostic ignored \"-Wsometimes-uninitialized\"\n"
    "#pragma clang diagnostic ignored \"-Wimplicit-fallthrough\"\n"
    "#pragma clang diagnostic ignored \"-Wtautological-compare\"\n"
    "#pragma clang diagnostic ignored \"-Wtautological-constant-out-of-range-compare\"\n"
    "#pragma clang diagnostic ignored \"-Wconstant-conversion\"\n"
    "#pragma clang diagnostic ignored \"-Wparentheses-equality\"\n"
    "#pragma clang diagnostic ignored \"-Wshift-count-overflow\"\n"
    "// --- IEEE binary64 <-> fp32 (Metal has no double) ---\n"
    "static inline float __d_from_bits(ulong b) {\n"
    "  uint hi = (uint)(b >> 32), lo = (uint)b; uint sign = hi & 0x80000000u;\n"
    "  int e = (int)((hi >> 20) & 0x7FFu); uint mhi = hi & 0xFFFFFu;\n"
    "  if (e == 0x7FF) return as_type<float>(sign | 0x7F800000u | (((mhi | lo) != 0u) ? 0x400000u : 0u));\n"
    "  if (e == 0) return as_type<float>(sign);\n"
    "  int fe = e - 1023 + 127; ulong m = ((ulong)mhi << 32) | (ulong)lo;\n"
    "  uint m23 = (uint)(m >> 29); ulong rem = m & ((1ul << 29) - 1ul); ulong half_ = 1ul << 28;\n"
    "  if (rem > half_ || (rem == half_ && (m23 & 1u))) { m23++; if (m23 == 0x800000u) { m23 = 0u; fe++; } }\n"
    "  if (fe >= 255) return as_type<float>(sign | 0x7F800000u);\n"
    "  if (fe <= 0) { if (fe < -23) return as_type<float>(sign); uint full = m23 | 0x800000u; return as_type<float>(sign | (full >> (uint)(1 - fe))); }\n"
    "  return as_type<float>(sign | ((uint)fe << 23) | m23);\n"
    "}\n"
    "static inline ulong __d_bits(float f) {\n"
    "  uint u = as_type<uint>(f); ulong sign = (ulong)(u & 0x80000000u) << 32; uint e = (u >> 23) & 0xFFu; uint m = u & 0x7FFFFFu;\n"
    "  if (e == 255u) return sign | 0x7FF0000000000000ul | ((ulong)m << 29);\n"
    "  if (e == 0u) { if (m == 0u) return sign; uint p = 31u - clz(m); ulong mm = ((ulong)m << (52u - p)) & 0xFFFFFFFFFFFFFul;\n"
    "    return sign | ((ulong)(p + 1023u - 149u) << 52) | mm; }\n"
    "  return sign | ((ulong)(e - 127u + 1023u) << 52) | ((ulong)m << 29);\n"
    "}\n"
    "// --- bf16 helpers ---\n"
    "static inline float __bf2f(ushort b) { return as_type<float>((uint)b << 16); }\n"
    "static inline ushort __f2bf(float f) { uint u = as_type<uint>(f); if (isnan(f)) return (ushort)0x7FC0u;\n"
    "  u += 0x7FFFu + ((u >> 16) & 1u); return (ushort)(u >> 16); }\n"
    "static inline float2 __bf2_unpack(uint v) { return float2(__bf2f((ushort)v), __bf2f((ushort)(v >> 16))); }\n"
    "static inline uint __bf2_pack(float2 v) { return (uint)__f2bf(v.x) | ((uint)__f2bf(v.y) << 16); }\n"
    "// --- float -> int with PTX saturation semantics ---\n"
    "static inline int __f2s32(float f) { if (isnan(f)) return 0; if (f >= 2147483648.0f) return 2147483647; if (f <= -2147483648.0f) return (-2147483647 - 1); return (int)f; }\n"
    "static inline uint __f2u32(float f) { if (isnan(f) || f <= 0.0f) return 0u; if (f >= 4294967296.0f) return 0xFFFFFFFFu; return (uint)f; }\n"
    "static inline long __f2s64(float f) { if (isnan(f)) return 0; if (f >= 9223372036854775808.0f) return 0x7FFFFFFFFFFFFFFFl; if (f <= -9223372036854775808.0f) return (long)0x8000000000000000ul; return (long)f; }\n"
    "static inline ulong __f2u64(float f) { if (isnan(f) || f <= 0.0f) return 0ul; if (f >= 18446744073709551616.0f) return 0xFFFFFFFFFFFFFFFFul; return (ulong)f; }\n"
    "// --- bit manipulation ---\n"
    "static inline uint __bfe32(uint a, uint b, uint c, int sgn) {\n"
    "  uint pos = b & 0xFFu, len = c & 0xFFu; if (len == 0u) return 0u;\n"
    "  uint sbit = sgn ? ((a >> min(pos + len - 1u, 31u)) & 1u) : 0u;\n"
    "  if (pos >= 32u) return sbit ? 0xFFFFFFFFu : 0u;\n"
    "  uint v = a >> pos; if (len < 32u) { uint m = (1u << len) - 1u; v = sbit ? (v | ~m) : (v & m); }\n"
    "  return v; }\n"
    "static inline ulong __bfe64(ulong a, uint b, uint c, int sgn) {\n"
    "  uint pos = b & 0xFFu, len = c & 0xFFu; if (len == 0u) return 0ul;\n"
    "  ulong sbit = sgn ? ((a >> min(pos + len - 1u, 63u)) & 1ul) : 0ul;\n"
    "  if (pos >= 64u) return sbit ? 0xFFFFFFFFFFFFFFFFul : 0ul;\n"
    "  ulong v = a >> pos; if (len < 64u) { ulong m = (1ul << len) - 1ul; v = sbit ? (v | ~m) : (v & m); }\n"
    "  return v; }\n"
    "static inline uint __bfi32(uint a, uint b, uint c, uint d) { uint pos = c & 0xFFu, len = d & 0xFFu; if (len == 0u || pos >= 32u) return b;\n"
    "  uint m = (len >= 32u ? 0xFFFFFFFFu : ((1u << len) - 1u)) << pos; return (b & ~m) | ((a << pos) & m); }\n"
    "static inline ulong __bfi64(ulong a, ulong b, uint c, uint d) { uint pos = c & 0xFFu, len = d & 0xFFu; if (len == 0u || pos >= 64u) return b;\n"
    "  ulong m = (len >= 64u ? 0xFFFFFFFFFFFFFFFFul : ((1ul << len) - 1ul)) << pos; return (b & ~m) | ((a << pos) & m); }\n"
    "static inline uint __bfind32(uint a, int shiftamt) { if (a == 0u) return 0xFFFFFFFFu; uint p = 31u - clz(a); return shiftamt ? 31u - p : p; }\n"
    "static inline uint __bfind32(int a, int shiftamt) { uint u = a < 0 ? ~(uint)a : (uint)a; return __bfind32(u, shiftamt); }\n"
    "static inline uint __bfind64(ulong a, int shiftamt) { if (a == 0ul) return 0xFFFFFFFFu; uint p = 63u - (uint)clz(a); return shiftamt ? 63u - p : p; }\n"
    "static inline uint __bfind64(long a, int shiftamt) { ulong u = a < 0 ? ~(ulong)a : (ulong)a; return __bfind64(u, shiftamt); }\n"
    "static inline uint __prmt(uint a, uint b, uint c, int mode) {\n"
    "  ulong x = ((ulong)b << 32) | (ulong)a; uint r = 0u;\n"
    "  if (mode == 0) { for (uint i = 0u; i < 4u; i++) { uint s = (c >> (4u * i)) & 0xFu; uint byte = (uint)(x >> (8u * (s & 7u))) & 0xFFu;\n"
    "      if (s & 8u) byte = (byte & 0x80u) ? 0xFFu : 0u; r |= byte << (8u * i); } return r; }\n"
    "  uint sel = c & 3u; uint idx[4];\n"
    "  switch (mode) {\n"
    "    case 1: for (uint i = 0u; i < 4u; i++) idx[i] = (sel + i) & 7u; break;\n"
    "    case 2: for (uint i = 0u; i < 4u; i++) idx[i] = (sel - i) & 7u; break;\n"
    "    case 3: for (uint i = 0u; i < 4u; i++) idx[i] = sel; break;\n"
    "    case 4: for (uint i = 0u; i < 4u; i++) idx[i] = max(i, sel); break;\n"
    "    case 5: for (uint i = 0u; i < 4u; i++) idx[i] = min(i, sel); break;\n"
    "    default: { uint h = (sel & 1u) * 2u; idx[0] = h; idx[1] = h + 1u; idx[2] = h; idx[3] = h + 1u; } break;\n"
    "  }\n"
    "  for (uint i = 0u; i < 4u; i++) r |= ((uint)(x >> (8u * idx[i])) & 0xFFu) << (8u * i);\n"
    "  return r; }\n"
    "static inline uint __lop3(uint a, uint b, uint c, uint lut) { uint r = 0u;\n"
    "  for (uint k = 0u; k < 8u; k++) if (lut & (1u << k)) r |= ((k & 4u) ? a : ~a) & ((k & 2u) ? b : ~b) & ((k & 1u) ? c : ~c);\n"
    "  return r; }\n"
    "static inline uint __shf(uint lo, uint hi, uint c, int left, int clampm) {\n"
    "  uint n = clampm ? min(c, 32u) : (c & 31u); ulong x = ((ulong)hi << 32) | (ulong)lo;\n"
    "  return left ? (uint)((x << n) >> 32) : (uint)(x >> n); }\n"
    "static inline uint __dp4a(uint a, uint b, uint c, int sa, int sb) { int acc = (int)c;\n"
    "  for (uint i = 0u; i < 4u; i++) { int x = sa ? (int)(char)(a >> (8u * i)) : (int)((a >> (8u * i)) & 0xFFu);\n"
    "    int y = sb ? (int)(char)(b >> (8u * i)) : (int)((b >> (8u * i)) & 0xFFu); acc += x * y; } return (uint)acc; }\n"
    "static inline uint __dp2a(uint a, uint b, uint c, int sa, int sb, int hi) { int acc = (int)c;\n"
    "  for (uint i = 0u; i < 2u; i++) { int x = sa ? (int)(short)(a >> (16u * i)) : (int)((a >> (16u * i)) & 0xFFFFu);\n"
    "    uint bi = (hi ? 2u : 0u) + i; int y = sb ? (int)(char)(b >> (8u * bi)) : (int)((b >> (8u * bi)) & 0xFFu); acc += x * y; } return (uint)acc; }\n"
    "static inline long __m24s(uint a, uint b) { return (long)((int)(a << 8) >> 8) * (long)((int)(b << 8) >> 8); }\n"
    "static inline ulong __m24u(uint a, uint b) { return (ulong)(a & 0xFFFFFFu) * (ulong)(b & 0xFFFFFFu); }\n"
    "static inline uint __mul24_s_lo(uint a, uint b) { return (uint)__m24s(a, b); }\n"
    "static inline uint __mul24_s_hi(uint a, uint b) { return (uint)(__m24s(a, b) >> 16); }\n"
    "static inline uint __mul24_u_lo(uint a, uint b) { return (uint)__m24u(a, b); }\n"
    "static inline uint __mul24_u_hi(uint a, uint b) { return (uint)(__m24u(a, b) >> 16); }\n"
    "// --- warp shuffle with exact PTX shfl.sync semantics (lane clamp & segment mask) ---\n"
    "static inline uint __mc_shfl(uint v, uint b, uint c, uint mode, uint lane, thread bool* p) {\n"
    "  uint bval = b & 31u, cval = c & 31u, segmask = (c >> 8) & 31u;\n"
    "  int maxLane = (int)((lane & segmask) | (cval & ~segmask)); int minLane = (int)(lane & segmask);\n"
    "  int j; bool pv;\n"
    "  if (mode == 0u) { j = (int)lane - (int)bval; pv = j >= maxLane; }\n"
    "  else if (mode == 1u) { j = (int)(lane + bval); pv = j <= maxLane; }\n"
    "  else if (mode == 2u) { j = (int)(lane ^ bval); pv = j <= maxLane; }\n"
    "  else { j = minLane | (int)(bval & ~segmask); pv = j <= maxLane; }\n"
    "  if (!pv) j = (int)lane;\n"
    "  *p = pv; return simd_shuffle(v, (ushort)j); }\n";


/* ───────────────────────────── structurizer ─────────────────────────────
 *
 * Turns a reducible CFG into structured MSL (for / do-while / if) following
 * N. Ramsey, "Beyond Relooper: recursive translation of unstructured control
 * flow to structured control flow" (ICFP 2022), driven by the dominator tree:
 *   - a loop header becomes `for (;;) { ... }`; back edges become `continue`
 *   - a node with >= 2 forward predecessors ("merge node") is emitted after a
 *     `do { ... } while (false)` block of its immediate dominator; forward
 *     branches to it become `break`
 *   - any other forward target is emitted inline at its single predecessor.
 * Two adaptations:
 *   - C has no labelled break/continue (and MSL has no goto), so a jump that
 *     crosses several constructs sets `__jmp` and the constructs it crosses
 *     forward it outward;
 *   - SIMT convergence: blocks outside a loop are never inlined into the loop
 *     body (which the sequential algorithm allows). They are placed after the
 *     outermost loop they leave, so code after a loop — typically a
 *     __syncthreads() — runs once all threads of the SIMD-group left the loop.
 * Irreducible CFGs are rejected (the caller falls back to a dispatch loop).
 */

typedef struct { int kind, target, id, check; } sctx_t; /* kind 0: block followed by target; 1: loop headed by target */

typedef struct {
    K *k;
    func_t *f;
    block_t *B;
    int nb;
    int *idom, *fwd_in, *exit_of, *kid_start, *kids;
    unsigned char *is_header, *inloop; /* inloop[h * nb + v] */
    sctx_t *st;
    int sp, cap, next_id;
    char (*labels)[MC_MAX_NAME];
    int *lblk, nl;
} SE;

static int se_tree(SE *s, int x);

static int se_find(SE *s, int kind, int target) {
    for (int j = s->sp - 1; j >= 0; j--)
        if (s->st[j].kind == kind && s->st[j].target == target) return s->sp - 1 - j;
    return -1;
}
static void se_jump(SE *s, int depth) {
    sctx_t *t = &s->st[s->sp - 1 - depth];
    if (depth == 0) { sb_puts(s->k->o, t->kind == 1 ? "continue;\n" : "break;\n"); return; }
    for (int j = 0; j < depth; j++) s->st[s->sp - 1 - j].check = 1;
    sb_printf(s->k->o, "__jmp = %du; break;\n", t->id);
}
static void se_push(SE *s, int kind, int target) {
    if (s->sp == s->cap) { s->cap *= 2; s->st = (sctx_t *)realloc(s->st, (size_t)s->cap * sizeof(sctx_t)); }
    sctx_t c = {kind, target, ++s->next_id, 0};
    s->st[s->sp++] = c;
}
static void se_pop(SE *s) {
    sctx_t c = s->st[--s->sp];
    if (c.check && s->sp > 0) {
        sctx_t *o = &s->st[s->sp - 1];
        sb_printf(s->k->o, "if (__jmp != 0u) { if (__jmp == %du) { __jmp = 0u; %s } else break; }\n", o->id, o->kind == 1 ? "continue;" : "break;");
    }
}

static int se_branch(SE *s, int src, int tgt) {
    if (s->B[tgt].id <= s->B[src].id) { /* back edge */
        int d = se_find(s, 1, tgt);
        if (d < 0) { PERR(s->k->p, "structurizer: back edge without loop in %s", s->f->name); return 0; }
        se_jump(s, d);
        return 1;
    }
    if (s->fwd_in[tgt] >= 2 || s->exit_of[tgt] >= 0) {
        int d = se_find(s, 0, tgt);
        if (d < 0) { PERR(s->k->p, "structurizer: merge target not in scope in %s", s->f->name); return 0; }
        se_jump(s, d);
        return 1;
    }
    return se_tree(s, tgt);
}

static int term_pred(K *k, instr_t *in, char *pe, size_t n) {
    pe[0] = 0;
    if (!in->pred[0]) return 1;
    operand_t pr;
    memset(&pr, 0, sizeof pr);
    pr.kind = OP_REG;
    snprintf(pr.name, sizeof pr.name, "%s", in->pred);
    char t[EXL];
    if (!rd(k, &pr, VT_PRED, t, sizeof t)) return 0;
    snprintf(pe, n, "%s%s", in->pred_not ? "!" : "", t);
    return 1;
}

static int emit_block_body(K *k, func_t *f, block_t *bl) {
    int last = bl->term >= 0 ? bl->term : bl->end;
    for (int i = bl->start; i < last; i++) {
        if (f->stmts[i].is_label) continue;
        if (!emit_instr(k, &f->stmts[i].ins)) return 0;
    }
    return 1;
}

static int se_node(SE *s, int x) {
    block_t *bl = &s->B[x];
    sb_t *o = s->k->o;
    int any = 0;
    for (int l = 0; l < s->nl; l++)
        if (s->lblk[l] == x) { sb_printf(o, "%s %s", any ? "," : "//", s->labels[l]); any = 1; }
    if (any) sb_puts(o, "\n");
    if (!emit_block_body(s->k, s->f, bl)) return 0;
    if (bl->term < 0) {
        if (bl->fall >= 0) return se_branch(s, x, bl->fall);
        sb_puts(o, "return;\n");
        return 1;
    }
    instr_t *in = &s->f->stmts[bl->term].ins;
    char pe[EXL];
    if (!term_pred(s->k, in, pe, sizeof pe)) return 0;
    if (is_branch(in)) {
        int T = bl->succ[0];
        if (!pe[0]) return se_branch(s, x, T);
        int F = bl->fall;
        if (F == T) return se_branch(s, x, T);
        sb_printf(o, "if (%s) {\n", pe);
        if (!se_branch(s, x, T)) return 0;
        sb_puts(o, "} else {\n");
        if (F >= 0) { if (!se_branch(s, x, F)) return 0; }
        else sb_puts(o, "return;\n");
        sb_puts(o, "}\n");
        return 1;
    }
    /* ret / exit / trap */
    if (pe[0]) {
        sb_printf(o, "if (%s) return;\n", pe);
        if (bl->fall >= 0) return se_branch(s, x, bl->fall);
    }
    sb_puts(o, "return;\n");
    return 1;
}

/* nodeWithin: x's code wrapped in blocks for its merge children ys (sorted by rpo, largest first) */
static int se_within(SE *s, int x, int *ys, int ny) {
    if (ny == 0) return se_node(s, x);
    se_push(s, 0, ys[0]);
    sb_puts(s->k->o, "do {\n");
    int ok = se_within(s, x, ys + 1, ny - 1);
    sb_puts(s->k->o, "} while (false);\n");
    se_pop(s);
    return ok && se_tree(s, ys[0]);
}
/* loop x, followed by the blocks relocated after it (es sorted by rpo, largest first) */
static int se_loop(SE *s, int x, int *es, int ne, int *ys, int ny) {
    if (ne == 0) {
        se_push(s, 1, x);
        sb_puts(s->k->o, "for (;;) {\n");
        int ok = se_within(s, x, ys, ny);
        sb_puts(s->k->o, "}\n");
        se_pop(s);
        return ok;
    }
    se_push(s, 0, es[0]);
    sb_puts(s->k->o, "do {\n");
    int ok = se_loop(s, x, es + 1, ne - 1, ys, ny);
    sb_puts(s->k->o, "} while (false);\n");
    se_pop(s);
    return ok && se_tree(s, es[0]);
}

static int se_tree(SE *s, int x) {
    int nk = s->kid_start[x + 1] - s->kid_start[x];
    int *ys = (int *)malloc((size_t)(nk + 1) * sizeof(int)), ny = 0;
    for (int i = nk - 1; i >= 0; i--) { /* kids are sorted by rpo ascending */
        int c = s->kids[s->kid_start[x] + i];
        if (s->exit_of[c] < 0 && s->fwd_in[c] >= 2) ys[ny++] = c;
    }
    int ok;
    if (s->is_header[x]) {
        int *es = (int *)malloc((size_t)s->nb * sizeof(int)), ne = 0;
        for (int c = 0; c < s->nb; c++) if (s->exit_of[c] == x) es[ne++] = c;
        for (int i = 1; i < ne; i++) /* sort by rpo descending */
            for (int j = i; j > 0 && s->B[es[j]].id > s->B[es[j - 1]].id; j--) { int t = es[j]; es[j] = es[j - 1]; es[j - 1] = t; }
        ok = se_loop(s, x, es, ne, ys, ny);
        free(es);
    } else {
        ok = se_within(s, x, ys, ny);
    }
    free(ys);
    return ok;
}

/* Returns 1 if emitted structured code, 0 if the CFG is irreducible / too large, -1 on error. */
static int structurize(K *k, func_t *f, block_t *B, int nb, char (*labels)[MC_MAX_NAME], int *lblk, int nl) {
    if (nb > 4096) return 0;
    int ok = 0;
    int *ord = (int *)calloc((size_t)nb, sizeof(int)), nr = 0;
    int *pcount = (int *)calloc((size_t)nb + 1, sizeof(int)), *pstart = (int *)calloc((size_t)nb + 1, sizeof(int));
    int *preds = NULL;
    SE s;
    memset(&s, 0, sizeof s);
    s.k = k; s.f = f; s.B = B; s.nb = nb; s.labels = labels; s.lblk = lblk; s.nl = nl;
    s.idom = (int *)malloc((size_t)nb * sizeof(int));
    s.fwd_in = (int *)calloc((size_t)nb, sizeof(int));
    s.exit_of = (int *)malloc((size_t)nb * sizeof(int));
    s.kid_start = (int *)calloc((size_t)nb + 1, sizeof(int));
    s.kids = (int *)calloc((size_t)nb + 1, sizeof(int));
    s.is_header = (unsigned char *)calloc((size_t)nb, 1);
    s.inloop = (unsigned char *)calloc((size_t)nb * (size_t)nb, 1);
    int *stack = (int *)malloc((size_t)nb * sizeof(int));

    for (int i = 0; i < nb; i++) if (B[i].reach) ord[B[i].id] = i, nr++;
    /* predecessors (reachable only) */
    for (int u = 0; u < nb; u++) if (B[u].reach) for (int j = 0; j < B[u].nsucc; j++) pcount[B[u].succ[j]]++;
    for (int i = 0; i < nb; i++) pstart[i + 1] = pstart[i] + pcount[i];
    preds = (int *)malloc((size_t)(pstart[nb] + 1) * sizeof(int));
    memset(pcount, 0, (size_t)nb * sizeof(int));
    for (int u = 0; u < nb; u++) if (B[u].reach) for (int j = 0; j < B[u].nsucc; j++) { int v = B[u].succ[j]; preds[pstart[v] + pcount[v]++] = u; }
    /* dominators (Cooper, Harvey & Kennedy) */
    for (int i = 0; i < nb; i++) s.idom[i] = -1;
    int entry = ord[0];
    s.idom[entry] = entry;
    for (int changed = 1; changed;) {
        changed = 0;
        for (int r = 1; r < nr; r++) {
            int b = ord[r], nd = -1;
            for (int j = pstart[b]; j < pstart[b + 1]; j++) {
                int p = preds[j];
                if (s.idom[p] < 0) continue;
                if (nd < 0) { nd = p; continue; }
                int a = p, c = nd;
                while (a != c) {
                    while (B[a].id > B[c].id) a = s.idom[a];
                    while (B[c].id > B[a].id) c = s.idom[c];
                }
                nd = a;
            }
            if (nd != s.idom[b]) { s.idom[b] = nd; changed = 1; }
        }
    }
    /* back edges must target dominators (reducibility); collect natural loops */
    for (int u = 0; u < nb; u++) {
        if (!B[u].reach) continue;
        for (int j = 0; j < B[u].nsucc; j++) {
            int v = B[u].succ[j];
            if (B[u].id < B[v].id) { s.fwd_in[v]++; continue; }
            int d = u;
            while (d != v && d != entry) d = s.idom[d];
            if (d != v) goto done; /* irreducible */
            s.is_header[v] = 1;
            unsigned char *in = s.inloop + (size_t)v * nb;
            in[v] = 1;
            int sp = 0;
            if (!in[u]) { in[u] = 1; stack[sp++] = u; }
            while (sp) {
                int x = stack[--sp];
                for (int q = pstart[x]; q < pstart[x + 1]; q++) if (!in[preds[q]]) { in[preds[q]] = 1; stack[sp++] = preds[q]; }
            }
        }
    }
    /* blocks leaving a loop are emitted after the outermost loop they leave */
    for (int c = 0; c < nb; c++) {
        s.exit_of[c] = -1;
        if (!B[c].reach || c == entry) continue;
        int v = s.idom[c], best = -1;
        for (int h = 0; h < nb; h++)
            if (s.is_header[h] && s.inloop[(size_t)h * nb + v] && !s.inloop[(size_t)h * nb + c] && (best < 0 || B[h].id < B[best].id)) best = h;
        s.exit_of[c] = best;
    }
    /* dominator-tree children, sorted by rpo */
    for (int r = 1; r < nr; r++) s.kid_start[s.idom[ord[r]] + 1]++;
    for (int i = 0; i < nb; i++) s.kid_start[i + 1] += s.kid_start[i];
    {
        int *fill = (int *)calloc((size_t)nb, sizeof(int));
        for (int r = 1; r < nr; r++) { int b = ord[r], d = s.idom[b]; s.kids[s.kid_start[d] + fill[d]++] = b; }
        free(fill);
    }
    s.cap = 64;
    s.st = (sctx_t *)malloc((size_t)s.cap * sizeof(sctx_t));
    ok = se_tree(&s, entry) ? 1 : -1;
    free(s.st);
done:
    free(ord); free(pcount); free(pstart); free(preds); free(stack);
    free(s.idom); free(s.fwd_in); free(s.exit_of); free(s.kid_start); free(s.kids); free(s.is_header); free(s.inloop);
    return ok;
}

/* ───────────────────────────── kernel emission ───────────────────────────── */

static int kernel_is_convergent(func_t *f) {
    for (int i = 0; i < f->nstmts; i++) {
        if (f->stmts[i].is_label) continue;
        const char *op = f->stmts[i].ins.parts[0];
        if (!strcmp(op, "bar") || !strcmp(op, "barrier") || !strcmp(op, "shfl") || !strcmp(op, "vote") || !strcmp(op, "redux") ||
            !strcmp(op, "activemask") || !strcmp(op, "elect") || !strcmp(op, "match"))
            return 1;
    }
    return 0;
}

static int operand_mentions(const operand_t *op, const char *name) {
    if (op->kind == OP_VEC) {
        for (int i = 0; i < op->nelems; i++) if (operand_mentions(&op->elems[i], name)) return 1;
        return 0;
    }
    return (op->kind == OP_SYM || (op->kind == OP_ADDR && op->has_base)) && !strcmp(op->name, name);
}
static int func_references(func_t *f, const char *name) {
    for (int i = 0; i < f->nstmts; i++) {
        if (f->stmts[i].is_label) continue;
        for (int o = 0; o < f->stmts[i].ins.nops; o++)
            if (operand_mentions(&f->stmts[i].ins.ops[o], name)) return 1;
    }
    return 0;
}

static int emit_kernel(P *p, func_t *f, mc_kernel_t *ki, sb_t *out) {
    K k;
    memset(&k, 0, sizeof k);
    k.p = p;
    k.f = f;
    k.ki = ki;
    /* parts[] point into each instruction's own buffer; the statement array may
     * have been reallocated while parsing, so re-split now that it is stable. */
    for (int i = 0; i < f->nstmts; i++)
        if (!f->stmts[i].is_label) split_parts(&f->stmts[i].ins);

    /* ---- shared / local layout ---- */
    uint32_t soff = 0, loff = 0;
    int has_extern = 0;
    for (int i = 0; i < p->nmvars; i++) {
        var_t *v = &p->mvars[i];
        if (v->space != SP_SHARED) continue;
        if (!func_references(f, v->name)) continue; /* only lay out what this kernel uses */
        if (v->is_extern) { has_extern = 1; continue; }
        soff = (uint32_t)align_up(soff, v->align ? v->align : 4);
        v->offset = soff;
        soff += (uint32_t)v->size;
    }
    for (int i = 0; i < f->nvars; i++) {
        var_t *v = &f->vars[i];
        if (v->space == SP_SHARED) {
            if (v->is_extern) { has_extern = 1; continue; }
            soff = (uint32_t)align_up(soff, v->align ? v->align : 4);
            v->offset = soff;
            soff += (uint32_t)v->size;
        } else if (v->space == SP_LOCAL) {
            loff = (uint32_t)align_up(loff, v->align ? v->align : 4);
            v->offset = loff;
            loff += (uint32_t)v->size;
        }
    }
    (void)has_extern;
    k.smem_static = soff;
    k.red_off = (uint32_t)align_up(soff, 16);
    k.smem_dyn_off = k.red_off + 16; /* 16 bytes reserved for bar.red scratch */
    k.local_bytes = loff;
    k.convergent = kernel_is_convergent(f);

    /* generic-address tagging is needed if the kernel converts shared/local/param
     * addresses to generic ones */
    for (int i = 0; i < f->nstmts; i++) {
        if (f->stmts[i].is_label) continue;
        instr_t *in = &f->stmts[i].ins;
        if (!strcmp(in->parts[0], "cvta") && !has_part(in, "to")) {
            int sp = space_of(in);
            if (sp == SP_SHARED || sp == SP_LOCAL || sp == SP_PARAM) k.uses_generic_tags = 1;
        }
    }

    /* ---- basic blocks ---- */
    int cap = 16, nb = 0;
    block_t *B = (block_t *)calloc((size_t)cap, sizeof(block_t));
    char (*labels)[MC_MAX_NAME] = calloc((size_t)(f->nstmts + 1), MC_MAX_NAME);
    int *lblk = (int *)calloc((size_t)(f->nstmts + 1), sizeof(int));
    int nl = 0;
    int cur = -1;
    for (int i = 0; i < f->nstmts; i++) {
        stmt_t *s = &f->stmts[i];
        if (s->is_label) {
            if (cur < 0 || B[cur].end > B[cur].start || B[cur].term >= 0) {
                if (nb == cap) { cap *= 2; B = (block_t *)realloc(B, (size_t)cap * sizeof(block_t)); }
                cur = nb++;
                memset(&B[cur], 0, sizeof B[cur]);
                B[cur].start = B[cur].end = i + 1;
                B[cur].term = -1;
            } else {
                B[cur].start = B[cur].end = i + 1;
            }
            snprintf(labels[nl], MC_MAX_NAME, "%s", s->label);
            lblk[nl++] = cur;
            continue;
        }
        if (cur < 0 || B[cur].term >= 0) {
            if (nb == cap) { cap *= 2; B = (block_t *)realloc(B, (size_t)cap * sizeof(block_t)); }
            cur = nb++;
            memset(&B[cur], 0, sizeof B[cur]);
            B[cur].start = i;
            B[cur].term = -1;
        }
        B[cur].end = i + 1;
        if (is_branch(&s->ins) || (is_ret(&s->ins) && !s->ins.pred[0])) B[cur].term = i;
        else if (is_ret(&s->ins) && s->ins.pred[0]) B[cur].term = i;
    }
    if (nb == 0) { /* empty kernel */
        nb = 1;
        memset(&B[0], 0, sizeof B[0]);
        B[0].term = -1;
    }
    for (int i = 0; i < nb; i++) {
        B[i].fall = i + 1 < nb ? i + 1 : -1;
        B[i].nsucc = 0;
        int t = B[i].term;
        if (t >= 0) {
            instr_t *in = &f->stmts[t].ins;
            if (is_branch(in)) {
                if (!strcmp(in->parts[0], "brx")) { PERR(p, "PTX line %d: indirect branches (brx.idx) are not supported", in->line); goto fail; }
                const char *target = in->nops ? in->ops[0].name : "";
                int tb = label_block(labels, lblk, nl, target);
                if (tb < 0) { PERR(p, "PTX line %d: branch to unknown label '%s'", in->line, target); goto fail; }
                B[i].succ[B[i].nsucc++] = tb;
                if (in->pred[0] && B[i].fall >= 0 && B[i].fall != tb) B[i].succ[B[i].nsucc++] = B[i].fall;
            } else if (in->pred[0] && B[i].fall >= 0) {
                B[i].succ[B[i].nsucc++] = B[i].fall; /* predicated ret */
            }
        } else if (B[i].fall >= 0) {
            B[i].succ[B[i].nsucc++] = B[i].fall;
        }
    }

    /* ---- loop-aware reverse post-order ---- */
    {
        cfg_t g;
        memset(&g, 0, sizeof g);
        g.b = B;
        g.nb = nb;
        g.state = (int *)calloc((size_t)nb, sizeof(int));
        g.post = (int *)calloc((size_t)nb, sizeof(int));
        dfs1(&g, 0);
        g.nloops = g.nback;
        g.inloop = (unsigned char *)calloc((size_t)(g.nloops ? g.nloops : 1) * (size_t)nb, 1);
        int *stack = (int *)malloc((size_t)nb * sizeof(int) * 2 + 16);
        for (int l = 0; l < g.nback; l++) {
            int u = g.backedge[l][0], h = g.backedge[l][1];
            unsigned char *in = g.inloop + (size_t)l * nb;
            in[h] = 1;
            int sp = 0;
            if (!in[u]) { in[u] = 1; stack[sp++] = u; }
            while (sp) { /* reverse reachability to the header */
                int x = stack[--sp];
                for (int y = 0; y < nb; y++)
                    for (int s = 0; s < B[y].nsucc; s++)
                        if (B[y].succ[s] == x && !in[y]) { in[y] = 1; stack[sp++] = y; }
            }
        }
        free(stack);
        memset(g.state, 0, (size_t)nb * sizeof(int));
        dfs2(&g, 0);
        int id = 0;
        for (int i = g.npost - 1; i >= 0; i--) B[g.post[i]].id = id++;
        for (int i = 0; i < nb; i++) { B[i].reach = g.state[i] != 0; if (!g.state[i]) B[i].id = id++; /* unreachable */ }
        free(g.state); free(g.post); free(g.backedge); free(g.inloop);
    }

    /* ---- body ---- */
    sb_t body = {0};
    k.o = &body;
    int linear = (nb == 1) && !(B[0].term >= 0 && is_branch(&f->stmts[B[0].term].ins));
    int structured = 0;
    if (!linear && !getenv("METALCUDA_NO_STRUCTURIZE")) {
        structured = structurize(&k, f, B, nb, labels, lblk, nl);
        if (structured < 0) { free(body.p); goto fail; }
        if (!structured) {
            if (body.p) body.p[0] = 0;
            body.n = 0;
            warn(p, "kernel %s has an irreducible CFG; using the block dispatch loop", f->name);
        }
    }
    /* fallback: dispatch loop over basic blocks, emitted in id order */
    int *order = (int *)malloc((size_t)nb * sizeof(int));
    for (int i = 0; i < nb; i++) order[B[i].id] = i;
    for (int oi = 0; oi < nb && !structured; oi++) {
        int bi = order[oi];
        block_t *bl = &B[bi];
        if (!linear) {
            sb_printf(&body, "case %d: {", bl->id);
            for (int l = 0; l < nl; l++) if (lblk[l] == bi) sb_printf(&body, " // %s", labels[l]);
            sb_puts(&body, "\n");
        }
        int last = bl->term >= 0 ? bl->term : bl->end;
        for (int s = bl->start; s < last; s++) {
            if (f->stmts[s].is_label) continue;
            if (!emit_instr(&k, &f->stmts[s].ins)) goto fail_body;
        }
        /* terminator */
        if (bl->term >= 0) {
            instr_t *in = &f->stmts[bl->term].ins;
            char pe[EXL] = "";
            if (in->pred[0]) {
                operand_t pr;
                memset(&pr, 0, sizeof pr);
                pr.kind = OP_REG;
                snprintf(pr.name, sizeof pr.name, "%s", in->pred);
                char t[EXL];
                if (!rd(&k, &pr, VT_PRED, t, sizeof t)) goto fail_body;
                snprintf(pe, sizeof pe, "%s%s", in->pred_not ? "!" : "", t);
            }
            if (is_branch(in)) {
                int tb = B[bl->succ[0]].id;
                if (pe[0] && bl->fall >= 0) sb_printf(&body, "__bb = (%s) ? %du : %du; continue;\n", pe, tb, B[bl->fall].id);
                else if (pe[0]) sb_printf(&body, "if (%s) { __bb = %du; continue; }\nreturn;\n", pe, tb);
                else sb_printf(&body, "__bb = %du; continue;\n", tb);
            } else { /* ret / exit / trap */
                if (pe[0]) {
                    sb_printf(&body, "if (%s) return;\n", pe);
                    if (bl->fall >= 0) sb_printf(&body, "__bb = %du; continue;\n", B[bl->fall].id);
                    else sb_puts(&body, "return;\n");
                } else sb_puts(&body, "return;\n");
            }
        } else if (!linear) {
            if (bl->fall >= 0) sb_printf(&body, "__bb = %du; continue;\n", B[bl->fall].id);
            else sb_puts(&body, "return;\n");
        }
        if (!linear) sb_puts(&body, "}\n");
    }

    /* ---- kernel info ---- */
    sanitize(ki->msl_name, sizeof ki->msl_name, f->name, "mc_");
    snprintf(ki->name, sizeof ki->name, "%s", f->name);
    ki->params = f->params;
    ki->nparams = f->nparams;
    f->params = NULL;
    ki->param_bytes = f->param_bytes;
    ki->hidden_offset = (uint32_t)align_up(f->param_bytes, 8);
    ki->total_param_bytes = ki->hidden_offset + 16;
    ki->uses_smem = k.uses_smem || has_extern;
    ki->smem_dyn_offset = k.smem_dyn_off;
    ki->static_smem = k.smem_static;
    ki->local_bytes = k.local_bytes;
    memcpy(ki->reqntid, f->reqntid, sizeof ki->reqntid);
    ki->convergent = k.convergent;
    /* if the kernel never touches shared memory, avoid the threadgroup argument */
    int need_smem = ki->uses_smem;

    /* ---- signature + declarations ---- */
    sb_printf(out, "\n// PTX entry: %s\n", f->name);
    sb_printf(out, "kernel void %s(constant uchar* __params [[buffer(0)]],\n", ki->msl_name);
    if (need_smem) sb_puts(out, "    threadgroup uchar* __smem [[threadgroup(0)]],\n");
    sb_puts(out,
            "    uint3 __tid [[thread_position_in_threadgroup]],\n"
            "    uint3 __ntid [[threads_per_threadgroup]],\n"
            "    uint3 __ctaid [[threadgroup_position_in_grid]],\n"
            "    uint3 __nctaid [[threadgroups_per_grid]],\n"
            "    uint __lane [[thread_index_in_simdgroup]],\n"
            "    uint __warpid [[simdgroup_index_in_threadgroup]])\n{\n");
    sb_printf(out, "  ulong __gbase = *(constant ulong*)(__params + %uu);\n", ki->hidden_offset);
    sb_printf(out, "  uint __dynsmem = *(constant uint*)(__params + %uu);\n", ki->hidden_offset + 8);
    sb_puts(out, "  uint __cc = 0u;\n");
    if (k.local_bytes) {
        sb_printf(out, "  uint4 __ldepot_store[%u];\n  thread uchar* __ldepot = (thread uchar*)__ldepot_store;\n", (k.local_bytes + 15) / 16);
    } else {
        sb_puts(out, "  thread uchar* __ldepot = (thread uchar*)&__cc;\n");
    }
    for (int i = 0; i < f->nvars; i++) {
        var_t *v = &f->vars[i];
        if (v->space != SP_PARAM) continue;
        char cn[EXL];
        sanitize(cn, sizeof cn, v->name, "__cp_");
        sb_printf(out, "  uint4 %s_store[%llu]; thread uchar* %s = (thread uchar*)%s_store;\n", cn, (unsigned long long)((v->size + 15) / 16 + 1), cn, cn);
    }
    for (int i = 0; i < f->nregs; i++) {
        regdecl_t *r = &f->regs[i];
        int st = vt_reg_storage(r->vt);
        const char *init = st == VT_PRED ? "false" : "0";
        char cn[EXL];
        if (r->count == 0) {
            reg_cname(cn, sizeof cn, r->prefix);
            sb_printf(out, "  %s %s = %s;\n", VT_C[st], cn, init);
        } else {
            sb_printf(out, "  %s", VT_C[st]);
            for (int j = 0; j < r->count; j++) {
                char nm[MC_MAX_NAME + 16];
                snprintf(nm, sizeof nm, "%s%d", r->prefix, j);
                reg_cname(cn, sizeof cn, nm);
                sb_printf(out, "%s%s = %s", j ? ", " : " ", cn, init);
                if (j % 8 == 7 && j + 1 < r->count) sb_puts(out, "\n   ");
            }
            sb_puts(out, ";\n");
        }
    }
    if (linear) {
        sb_puts(out, body.p ? body.p : "");
    } else if (structured) {
        sb_puts(out, "  uint __jmp = 0u;\n");
        sb_puts(out, body.p ? body.p : "");
    } else {
        sb_puts(out, "  uint __bb = 0u;\n  for (;;) {\n");
        if (k.convergent) sb_puts(out, "  uint __cur = simd_min(__bb);\n  if (__bb != __cur) continue;\n");
        sb_puts(out, "  switch (__bb) {\n");
        sb_puts(out, body.p ? body.p : "");
        sb_puts(out, "  default: return;\n  }\n  }\n");
    }
    sb_puts(out, "}\n");
    free(body.p);
    free(order);
    free(B);
    free(labels);
    free(lblk);
    return 1;
fail_body:
    free(body.p);
    free(order);
fail:
    free(B);
    free(labels);
    free(lblk);
    return 0;
}

/* ───────────────────────────── public API ───────────────────────────── */

static void free_parser(P *p) {
    for (int i = 0; i < p->nfuncs; i++) {
        func_t *f = &p->funcs[i];
        for (int s = 0; s < f->nstmts; s++)
            if (!f->stmts[s].is_label)
                for (int o = 0; o < f->stmts[s].ins.nops; o++) free_operand(&f->stmts[s].ins.ops[o]);
        free(f->stmts);
        free(f->regs);
        free(f->vars);
        free(f->params);
    }
    free(p->funcs);
    free(p->mvars);
    free(p->t);
}

int mc_translate_ptx(const char *ptx, size_t len, mc_module_t *out) {
    memset(out, 0, sizeof *out);
    P p;
    memset(&p, 0, sizeof p);
    p.t = tokenize(ptx, len, &p.nt);
    if (!parse_module(&p)) {
        snprintf(out->error, sizeof out->error, "%s", p.err[0] ? p.err : "PTX parse error");
        free_parser(&p);
        free(p.ginit);
        free(p.warn.p);
        return -1;
    }
    sb_t msl = {0};
    sb_puts(&msl, PRELUDE);
    int nent = 0;
    for (int i = 0; i < p.nfuncs; i++) if (p.funcs[i].is_entry && !p.funcs[i].is_proto) nent++;
    out->kernels = (mc_kernel_t *)calloc((size_t)(nent ? nent : 1), sizeof(mc_kernel_t));
    for (int i = 0; i < p.nfuncs; i++) {
        func_t *f = &p.funcs[i];
        if (!f->is_entry || f->is_proto) continue;
        if (!emit_kernel(&p, f, &out->kernels[out->nkernels], &msl)) {
            snprintf(out->error, sizeof out->error, "%s", p.err[0] ? p.err : "translation failed");
            for (int k = 0; k <= out->nkernels; k++) free(out->kernels[k].params);
            free(out->kernels);
            out->kernels = NULL;
            out->nkernels = 0;
            free(msl.p);
            free_parser(&p);
            free(p.ginit);
            free(p.warn.p);
            return -1;
        }
        out->nkernels++;
    }
    /* module globals */
    int ng = 0;
    for (int i = 0; i < p.nmvars; i++) if (p.mvars[i].space == SP_GLOBAL || p.mvars[i].space == SP_CONST) ng++;
    out->globals = (mc_global_t *)calloc((size_t)(ng ? ng : 1), sizeof(mc_global_t));
    for (int i = 0; i < p.nmvars; i++) {
        var_t *v = &p.mvars[i];
        if (v->space != SP_GLOBAL && v->space != SP_CONST) continue;
        mc_global_t *g = &out->globals[out->nglobals++];
        snprintf(g->name, sizeof g->name, "%s", v->name);
        g->offset = v->offset;
        g->size = v->size;
        g->is_const = v->space == SP_CONST;
    }
    out->global_init = p.ginit;
    out->globals_bytes = p.gbytes;
    out->msl = msl.p;
    out->msl_len = msl.n;
    out->warnings = p.warn.p;
    free_parser(&p);
    return 0;
}

void mc_module_free(mc_module_t *m) {
    if (!m) return;
    for (int i = 0; i < m->nkernels; i++) free(m->kernels[i].params);
    free(m->kernels);
    free(m->globals);
    free(m->global_init);
    free(m->msl);
    free(m->warnings);
    memset(m, 0, sizeof *m);
}
