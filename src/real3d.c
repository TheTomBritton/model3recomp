/* model3recomp -- Real3D scene renderer.
 *
 * Walks the scene the game builds every field -- viewports, a tree of culling
 * nodes, models -- and rasterises it with a depth buffer.
 *
 * Every structure below was read off a live scene dump and then checked
 * against something that could only come out right if the decode was right.
 * That matters more here than anywhere else in the emulator, because a 3D
 * pipeline decoded wrongly does not fail: it draws. It draws confident,
 * plausible, wrong geometry, and there is no screen to check it against --
 * MAME does not render Model 3 3D at all, so the oracle that settled the
 * tilemaps is not available for any of this.
 *
 * The checks, per structure:
 *
 *   viewport      word 0x14 decodes to 496x384, the exact display size, and
 *                 words 0x0C..0x0F are sin/cos pairs for 19.38 and 15
 *                 degrees -- whose tangent ratio, 1.3125, is the aspect of a
 *                 496x384 screen.
 *   culling node  ten words; a run of them at stride 0x0A has [00] rising by
 *                 0x400 and the matrix index in [03] by 1, every time.
 *   pointer list  entries are addresses of valid nodes, and the unused ones
 *                 are all the same dummy, 0x800800.
 *   matrix        twelve floats, translation first, then a 3x3 by rows.
 *                 Matrix 1 is [0,0,0, 1,0,0, 0,1,0, 0,0,1], which is exactly
 *                 identity under this reading and nothing under any other.
 *   polygon       the whole of polygon RAM tiles into models with zero slack
 *                 -- the last model ends on the last word -- under the rule
 *                 that a header is seven words followed by four words per
 *                 vertex not reused from the previous polygon.
 *   normals       every one of the 1260 header normals in that block comes
 *                 out exactly unit length. This is the strongest check here,
 *                 and it is what pinned the 24-bit sign extension and the
 *                 2^22 scale.
 *   VROM          a 16-lane interleave read big-endian is the only
 *                 arrangement in which VROM contains any models at all:
 *                 31265 of them, against none for the 8-lane image the ROM
 *                 builder used to produce. See tools/rom_loader.py.
 *
 * What is NOT settled is the camera. Matrix 0 is used as the view transform
 * because it is the only one of the table's first four that is a plain
 * rotation and the only one that puts any of the scene in front of the eye.
 * That is an argument, not a check, and it has never been held up against a
 * frame whose correct appearance is known -- attract mode does not reach a
 * scene with anything recognisable in it before a run times out. The light
 * direction is invented outright, and there are no textures.
 */
#include "model3recomp/real3d.h"
#include "model3recomp/tilegen.h"
#include "model3recomp/savestate.h"
#include "model3recomp/bus.h"
#include "model3recomp/platform.h"
#include "model3recomp/model3recomp.h"

#include <stdio.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_TRIS   65536u     /* a field's, every viewport together */
#define MAX_DEPTH  24
#define AMBIENT    g_ambient

/* The vertex fixed-point scale: the 24-bit coordinate field over 2048.
 *
 * It was 256 for a long time, on a check that looked sound: a frame of
 * attract mode resolved into rocks and a cliff face at 1/256 and into
 * nothing recognisable at 1/16 or 1/1024. That check was run while the
 * matrix stack push was being lifted into unreachable code (see
 * tools/ppc_lifter.py), so every matrix in the scene carried the previous
 * one's scale. A wrong scale in the matrices and a wrong scale here
 * cancelled, and the frame looked right.
 *
 * With the push fixed the scale could be measured at all, and "the frame
 * looks like a scene" put it at 4096. That is a weak test: a scene looks
 * plausible over a wide range, and the range is wide because everything
 * in it scales together.
 *
 * A model either joins up or it does not, and that is a sharp test. Only
 * the vertices scale here -- the node translations that place a model's
 * parts are floats in world units and do not -- so at the wrong scale a
 * jointed model comes apart. The human figure and the jeep beside it on
 * the jungle floor are whole at 2048, spaced out at 4096 and scattered
 * into confetti at 8192. The T-Rex in attract survives any of them
 * because it is one rigid model, which is why it never showed the fault.
 *
 * MAME documents the coordinates as 13.11 fixed point outside Step 1.0,
 * which is the same 2048.
 *
 * Two earlier answers here were wrong, in opposite directions, and both came
 * with reasoning:
 *
 *   1/1024, from taking a flat 4:3 model for a backdrop that filled the
 *   screen and reading two matrices as independently agreeing on the size
 *   that implied. They were not agreeing on anything.
 *
 *   1/65536, from the observation that model extents crowd the top of the
 *   24-bit field. That observation is true and it is also worthless here:
 *   the raw integers fill the field whatever divisor you then apply, so it
 *   says nothing about the divisor at all. What it looked like was evidence.
 *
 * The lesson both times was the same. Nothing about this pipeline announces
 * a wrong scale -- it just draws something -- so the only thing that settles
 * it is a frame with a recognisable subject in it. M3_VERTEX_SCALE overrides.
 */
#define VERTEX_SCALE_DEFAULT 2048.0f

static float g_vertex_scale = VERTEX_SCALE_DEFAULT;
static int g_tex_size_mode;
static int g_tex_all;

/* Texturing. M3_NO_TEXTURES turns it off.
 *
 * This works now, and what unblocked it was not the renderer. The guest
 * could not read its own data ROM: the banked CROM window decoded the bank
 * select by shifting the written byte into an offset, which for the values
 * this game writes lands outside the image, so every read through the
 * window returned zero. With that fixed the guest streams 3.4 MB to the
 * Real3D in a run instead of 321 KB, and the difference is every texture in
 * the game.
 */
static int g_textures;
static float g_ambient = 0.20f;
static int   g_ambient_forced;
static int g_light_flip;
static int g_old_proj;
static int g_no_fog;        /* M3_NO_FOG: for comparing */
static int g_old_trans;     /* M3_OLD_TRANSLUCENCY: blend, not the two layers */
static int g_no_mips;       /* M3_NO_MIPS: base level only */
static int g_no_spec;       /* M3_NO_SPECULAR */      /* M3_OLD_PROJ: the symmetric whole-screen projection */
static uint32_t g_scene_hash;
static uint32_t g_fb_hash;
static float g_vp07;
static unsigned g_tri_count;

unsigned real3d_last_tri_count(void) { return g_tri_count; }

/* ---- memory ------------------------------------------------------------
 * Culling and polygon RAM are little-endian, like everything else the
 * graphics side reads. VROM is the exception: see the note above.
 */
static uint32_t rd32le(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint32_t rd32be(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

/* A culling address is a word index, not a byte offset: viewport 0 links to
 * 0x800030 and the next viewport node begins exactly at word 0x30. */
static const uint8_t *cull_at(uint32_t addr, unsigned words)
{
    size_t n = 0;
    const uint8_t *m;
    uint32_t base;

    addr &= 0x00FFFFFFu;
    if (addr >= 0x800000u && addr < 0x840000u) {
        m = bus_cull_hi(&n);
        base = (addr & 0x3FFFFu) * 4u;
    } else if (addr < 0x100000u) {
        m = bus_cull_lo(&n);
        base = addr * 4u;
    } else {
        return NULL;
    }
    if (!m || (size_t)base + (size_t)words * 4u > n)
        return NULL;
    return m + base;
}

static float f32_of(uint32_t w)
{
    float f;
    memcpy(&f, &w, sizeof f);
    return f;
}

/* Sign-extend the top 24 bits of a word. */
static int32_t sx24(uint32_t w)
{
    uint32_t v = w >> 8;
    return (v & 0x800000u) ? (int32_t)(v | 0xFF000000u) : (int32_t)v;
}

/* ---- matrices ---------------------------------------------------------- */
typedef struct { float m[3][4]; } mat_t;

static const mat_t IDENT = {{{1,0,0,0},{0,1,0,0},{0,0,1,0}}};

static int mat_load(uint32_t base, uint32_t off, mat_t *out)
{
    const uint8_t *p = cull_at(base + off * 12u, 12);
    float v[12];
    int i;
    if (!p) return 0;
    for (i = 0; i < 12; i++)
        v[i] = f32_of(rd32le(p + i * 4));
    out->m[0][0] = v[3];  out->m[0][1] = v[4];  out->m[0][2] = v[5];
    out->m[1][0] = v[6];  out->m[1][1] = v[7];  out->m[1][2] = v[8];
    out->m[2][0] = v[9];  out->m[2][1] = v[10]; out->m[2][2] = v[11];
    out->m[0][3] = v[0];  out->m[1][3] = v[1];  out->m[2][3] = v[2];
    return 1;
}

static mat_t mat_mul(const mat_t *a, const mat_t *b)
{
    mat_t o;
    int r, c;
    for (r = 0; r < 3; r++) {
        for (c = 0; c < 3; c++)
            o.m[r][c] = a->m[r][0]*b->m[0][c] + a->m[r][1]*b->m[1][c]
                      + a->m[r][2]*b->m[2][c];
        o.m[r][3] = a->m[r][0]*b->m[0][3] + a->m[r][1]*b->m[1][3]
                  + a->m[r][2]*b->m[2][3] + a->m[r][3];
    }
    return o;
}

static void mat_point(const mat_t *m, const float *p, float *o)
{
    int r;
    for (r = 0; r < 3; r++)
        o[r] = m->m[r][0]*p[0] + m->m[r][1]*p[1] + m->m[r][2]*p[2] + m->m[r][3];
}

static void mat_dir(const mat_t *m, const float *p, float *o)
{
    int r;
    for (r = 0; r < 3; r++)
        o[r] = m->m[r][0]*p[0] + m->m[r][1]*p[1] + m->m[r][2]*p[2];
}

/* ---- textures ----------------------------------------------------------
 *
 * Texture memory is two 2048x1024 pages of 16-bit texels, stacked here into
 * one 2048x2048 sheet, built from uploads. The guest streams entries into
 * the FIFO at 0x94000000, each one
 *
 *     word 0   length of the entry in bytes
 *     word 1   the texture header
 *     word 2+  the texels
 *
 * and an entry occupies (2 + length/2)/4 words, which is how the next one is
 * found. The FIFO is drained when the guest rings the doorbell at
 * 0x88000000.
 *
 * The header layout, the 8x8 tiling of the texel data, the order of the
 * texels inside a tile and where mip levels live are as MAME's model3_v.cpp
 * (BSD-3-Clause, R. Belmont and Ville Linde) has them; see
 * real3d_upload_texture() there.
 */
#define TEX_W 2048u
#define TEX_H 2048u

/* TILE16 and TILE8 are MAME's texture_decode16 and texture_decode8, from
 * src/mame/sega/model3_v.cpp -- license:BSD-3-Clause, copyright-holders:
 * R. Belmont, Ville Linde. The licence's full text is at the top of
 * scsp.c; see THIRD_PARTY.md.
 *
 * Which 16-bit word of an arriving 8x8 tile a texel comes from, by its
 * position in the tile (row-major), for 16-bit textures. Index it with the
 * position XOR 1: neighbouring texels arrive swapped. MAME's
 * texture_decode16. */
static const unsigned char TILE16[64] = {
     0,  1,  4,  5,  8,  9, 12, 13,
     2,  3,  6,  7, 10, 11, 14, 15,
    16, 17, 20, 21, 24, 25, 28, 29,
    18, 19, 22, 23, 26, 27, 30, 31,
    32, 33, 36, 37, 40, 41, 44, 45,
    34, 35, 38, 39, 42, 43, 46, 47,
    48, 49, 52, 53, 56, 57, 60, 61,
    50, 51, 54, 55, 58, 59, 62, 63
};

/* For 8-bit textures: a word carries two texels, the left one in its high
 * byte, so a tile row is four words. Indexed by row * 4 + column / 2.
 * MAME's texture_decode8. */
static const unsigned char TILE8[32] = {
     1,  3,  5,  7,
     0,  2,  4,  6,
     9, 11, 13, 15,
     8, 10, 12, 14,
    17, 19, 21, 23,
    16, 18, 20, 22,
    25, 27, 29, 31,
    24, 26, 28, 30
};

/* Mip level i of a page sits in its bottom-right corner: level 1 in the
 * quarter from (1024, 512), level 2 from (1536, 768), and so on, each
 * level's textures at their base position over 2^i. */
static unsigned mip_x(unsigned i) { return TEX_W - (TEX_W >> i); }
static unsigned mip_y(unsigned i) { return 1024u - (1024u >> i); }

static uint16_t *g_sheet;
static uint8_t  *g_sheet_set;          /* TEX_W * TEX_H texels */
static size_t    g_fifo_seen;      /* how much of the FIFO has been drained */
static unsigned  g_tex_uploads;

/* Set while a VROM texture is being loaded: that source is big-endian,
 * the FIFO is not. */
static int g_src_be;

static uint16_t src_word(const uint8_t *src, size_t i)
{
    return g_src_be ? (uint16_t)((src[i*2] << 8) | src[i*2 + 1])
                    : (uint16_t)(src[i*2] | (src[i*2 + 1] << 8));
}

/* Write one 8-bit texel into the byte(s) of a sheet texel the upload
 * selects: bit 0 of sel the low byte, bit 1 the high. */
static void put8(size_t o, unsigned v, unsigned sel)
{
    uint16_t t = g_sheet[o];
    if (sel & 1u) t = (uint16_t)((t & 0xFF00u) | v);
    if (sel & 2u) t = (uint16_t)((t & 0x00FFu) | (v << 8));
    g_sheet[o] = t;
    if (g_sheet_set) g_sheet_set[o] = 1;
}

/* One level, w x h at sheet position (x, y), from src (avail bytes).
 * Returns how many bytes of src it consumed. */
static size_t store_level(uint32_t x, uint32_t y, uint32_t w, uint32_t h,
                          int sixteen, unsigned sel, const uint8_t *src, size_t avail)
{
    size_t words = avail / 2u, base = 0, per_tile = sixteen ? 64u : 32u;
    uint32_t tx, ty, r, c;

    for (ty = 0; ty < h; ty += 8u) {
        for (tx = 0; tx < w; tx += 8u, base += per_tile) {
            if (base + per_tile > words) return base * 2u;
            for (r = 0; r < 8u; r++) {
                size_t row = (size_t)(y + ty + r) * TEX_W + (x + tx);
                if (sixteen) {
                    for (c = 0; c < 8u; c++) {
                        g_sheet[row + c] = src_word(src, base + TILE16[(r * 8u + c) ^ 1u]);
                        if (g_sheet_set) g_sheet_set[row + c] = 1;
                    }
                } else if (sel) {
                    for (c = 0; c < 8u; c += 2u) {
                        uint16_t d = src_word(src, base + TILE8[r * 4u + c / 2u]);
                        put8(row + c, d >> 8, sel);
                        put8(row + c + 1u, d & 0xFFu, sel);
                    }
                }
            }
        }
    }
    return base * 2u;
}

static void upload_texture(uint32_t header, const uint8_t *src, size_t avail)
{
    /* The origin and size of an upload, in the header's fields: x and y in
     * 32-texel units, the page, and width and height as 32 << n. Dumping
     * the sheet to a PNG shows a coherent atlas -- jeeps, Ian and Sarah,
     * dinosaur hide, INGEN signage.
     *
     * The FIFO only ever fills page 0. Page 1 comes from the VROM texture
     * port; see real3d_vrom_texture(). */
    uint32_t x = 32u * (header & 0x3Fu);
    uint32_t y = 32u * ((header >> 7) & 0x1Fu);
    uint32_t page = (header >> 20) & 1u;
    uint32_t w = 32u << ((header >> 14) & 7u);
    uint32_t h = 32u << ((header >> 17) & 7u);
    uint32_t type = header >> 24;
    int sixteen = (header >> 23) & 1u;
    /* An 8-bit upload writes one byte of each texel: bit 21 the low byte,
     * bit 22 the high, so two 8-bit textures can share one spot. */
    unsigned sel = ((header >> 21) & 1u) | (((header >> 22) & 1u) << 1);
    unsigned level;
    size_t used;

    if (!g_sheet || w > 1024u || h > 1024u) return;
    if (x + w > TEX_W || y + h > 1024u) return;
    /* The top byte: 0x00 a texture and its mip chain, 0x01 the texture
     * alone, 0x02 the chain alone, 0x80 possibly a gamma table. */
    if (type > 2u) return;
    level = type == 2u ? 1u : 0u;
    if (level == 0u && avail < (size_t)w * h * (sixteen ? 2u : 1u)) return;
    for (; w >> level >= 8u && h >> level >= 8u; level++) {
        used = store_level(mip_x(level) + (x >> level), page * 1024u + mip_y(level) + (y >> level),
                           w >> level, h >> level, sixteen, sel, src, avail);
        if (type == 1u || used >= avail) break;
        src += used; avail -= used;
    }
    g_tex_uploads++;
}

/* The VROM texture port at 0x90000000.
 *
 * Three registers. The game writes an address and a header and the third
 * write loads a texture that is already in VROM, rather than pushing its
 * pixels through the FIFO. A stage's scenery comes this way: in a round
 * the FIFO stops growing at field 1500 and every terrain and tree polygon
 * then looks for a texture that was never uploaded.
 *
 * Which register is which is worked out from what the game writes; see
 * M3_VROMTEX_TRACE. */
static uint32_t bswap32_(uint32_t v)
{
    return ((v >> 24) & 0xFFu) | ((v >> 8) & 0xFF00u)
         | ((v << 8) & 0xFF0000u) | ((v << 24) & 0xFF000000u);
}

/* Deferred (real3d_set_deferred): uploads wait in the field's queue for the
 * thread that draws it. */
static int g_deferred;
static void queue_tex(uint32_t kind, uint32_t a, uint32_t b, const uint8_t *p, size_t n);

void real3d_vrom_texture(uint32_t addr, uint32_t header, uint32_t ctrl)
{
    size_t n = 0;
    const uint8_t *v = bus_vrom(&n);
    /* The guest writes this port byte-reversed, the way it writes every
     * other Real3D register, and the address counts 16-bit texels. Read
     * as bytes, or little-endian, the sheet fills with noise; this way
     * it fills with planks, gravel, foliage and rock. */
    uint32_t a = bswap32_(addr), h = bswap32_(header);
    size_t off;
    if (getenv("M3_VROMTEX_TRACE"))
        fprintf(stderr, "[vromtex] load addr %08X header %08X ctrl %08X\n",
                a, h, bswap32_(ctrl));
    if (!v) return;
    /* The address counts 32-bit words, like a model pointer: consecutive
     * loads step by exactly one texture's size in words. The image is 32 MB
     * and the addresses reach past it, so it is mirrored -- the way bit 23
     * of a model address selects VROM and is then dropped. */
    off = ((size_t)a * 4u) % n;
    if (g_deferred) { queue_tex(1, h, (uint32_t)off, NULL, 0); return; }
    g_src_be = 1;                  /* VROM is big-endian; the FIFO is not */
    upload_texture(h, v + off, n - off);
    g_src_be = 0;
}

/* Drain whatever the guest has put in the FIFO since last time. */
static void apply_fifo(const uint8_t *f, size_t used)
{
    size_t i;
    for (i = 0; i + 8 <= used; ) {
        uint32_t len = rd32le(f + i);
        uint32_t words = (2u + len / 2u) / 4u;
        uint32_t header = rd32le(f + i + 4);
        if (getenv("M3_TEX_TRACE")) {
            static unsigned n;
            if (n++ < 12)
                fprintf(stderr, "[tex] entry at %u: len %u -> %u words, "
                        "header %08X%s", (unsigned)i, len, words, header,
                        "\n");
        }
        if (!words || (size_t)words * 4u > used - i) break;
        upload_texture(header, f + i + 8, (size_t)words * 4u - 8u);
        i += (size_t)words * 4u;
    }
}

static void flush_texture_fifo(void)
{
    size_t used = 0;
    const uint8_t *f = bus_texfifo(&used, NULL);
    if (!f || used <= g_fifo_seen) return;
    if (g_deferred) queue_tex(0, 0, 0, f, used);
    else apply_fifo(f, used);
    g_fifo_seen = used;
    if (getenv("M3_TEX_TRACE"))
        fprintf(stderr, "[tex] fifo %u bytes, %u uploads so far%s",
                (unsigned)used, g_tex_uploads, "\n");
}

/* The raw 16-bit texel, or -1 where nothing was ever uploaded. */
static int sheet_word(unsigned x, unsigned y)
{
    size_t o;
    if (!g_sheet) return -1;
    o = (size_t)(y & (TEX_H - 1u)) * TEX_W + (x & (TEX_W - 1u));
    if (g_sheet_set && !g_sheet_set[o]) return -1;
    return g_sheet[o];
}

/* A texel in one of the eight formats (header word 6 bits 7..9), as
 * 0xAARRGGBB, following MAME's get_texture(). Format 0 is 1-5-5-5 with red
 * the top field and bit 15 set meaning clear -- the opposite way round to the
 * tilemap palette, which a green jeep cannot show and Ian and Sarah's faces
 * can. 1-4 are four-bit luminance with four-bit alpha in one byte of the
 * texel, 5 and 6 eight-bit luminance from one byte, 7 is 4-4-4-4 with alpha
 * in the low nibble.
 *
 * key: the polygon has contour (cut-out) on. The eight-bit formats have no
 * alpha, and a cut-out in them is keyed on full white: the foliage card in
 * stage one's first scene is format 5 with contour, and 38758 of its 65536
 * texels are exactly 0xFF, all of them the space around the leaves. */
static uint32_t texel_argb(unsigned fmt, unsigned v, int key)
{
    unsigned r, g, b, a = 255;
    switch (fmt) {
    case 0:
        r = (v >> 10) & 0x1Fu; g = (v >> 5) & 0x1Fu; b = v & 0x1Fu;
        r = (r << 3) | (r >> 2); g = (g << 3) | (g >> 2); b = (b << 3) | (b >> 2);
        a = (v & 0x8000u) ? 0 : 255;
        return (a << 24) | (r << 16) | (g << 8) | b;
    case 1:
    case 2: r = (v & 0xFu) * 17u;         a = ((v >> 4) & 0xFu) * 17u;  break;
    case 3: r = ((v >> 8) & 0xFu) * 17u;  a = ((v >> 12) & 0xFu) * 17u; break;
    case 4: r = ((v >> 12) & 0xFu) * 17u; a = ((v >> 8) & 0xFu) * 17u;  break;
    case 5: r = v & 0xFFu;        if (key && r == 255u) a = 0; break;
    case 6: r = (v >> 8) & 0xFFu; if (key && r == 255u) a = 0; break;
    default:
        r = ((v >> 12) & 0xFu) * 17u; g = ((v >> 8) & 0xFu) * 17u;
        b = ((v >> 4) & 0xFu) * 17u;  a = (v & 0xFu) * 17u;
        return (a << 24) | (r << 16) | (g << 8) | b;
    }
    return (a << 24) | (r << 16) | (r << 8) | r;
}

/* One texel coordinate, wrapped: a mirrored axis runs forwards then
 * backwards, so odd repeats are reflected. */
static unsigned wrap_axis(long i, unsigned size, int mirror)
{
    /* Texture sizes are 32 << n, so a mask wraps -- two's complement makes
     * it right for negative coordinates too. */
    unsigned u = (unsigned)i, m = size - 1u;
    if (mirror && (u & size))
        return m - (u & m);
    return u & m;
}

/* Bilinear. Each texel's colour is first taken from whichever neighbour is
 * more opaque, so a cut-out's transparent texels do not bleed dark into its
 * edge. Returns -1 when the texture was never uploaded. */
struct tri_s;
static int64_t sample(const struct tri_s *tr, double u, double v, double lod);

/* ---- the triangle queue ------------------------------------------------ */
typedef struct tri_s {
    float v[3][3];          /* view space */
    float n[3];             /* the polygon's own normal, from the header */
    float vn[3][3];         /* per-vertex normals, for Gouraud */
    float uv[3][2];         /* texel coordinates */
    uint16_t tx, ty;        /* texture origin in the sheet */
    uint16_t tw, th;        /* its size, for wrapping */
    uint8_t page;
    uint8_t textured;
    uint8_t luminous;
    uint8_t two_sided;
    uint8_t alpha;          /* 255 opaque */
    uint8_t fmt;            /* texel format, 0..7 */
    uint8_t alpha_test;     /* contour: drop low-alpha texels */
    uint8_t tex_alpha;      /* texel alpha makes it see-through */
    uint8_t layer;          /* translucency pattern select, word 6 bit 17: 0 or 1 */
    uint8_t mirror;         /* bit 0 mirror v, bit 1 mirror u (header word 2) */
    uint8_t fog;            /* fog intensity in 16ths, 0..31 (header word 6) */
    uint8_t spec;           /* specular coefficient 0..63, 0 none (word 0) */
    uint8_t shine;          /* shininess 0..3 (word 6) */
    uint8_t r, g, b;
} tri_t;

static tri_t *g_tris;

static int64_t sample_level(const tri_t *tr, double u, double v, unsigned lv)
{
    unsigned tw = (tr->tw ? tr->tw : 32u) >> lv, th = (tr->th ? tr->th : 32u) >> lv;
    unsigned ox = mip_x(lv) + (tr->tx >> lv), oy = mip_y(lv) + ((tr->ty & 1023u) >> lv);
    double fu = u / (double)(1u << lv) - 0.5, fv = v / (double)(1u << lv) - 0.5;
    double flu = floor(fu), flv = floor(fv);
    long u0 = (long)flu, v0 = (long)flv;
    /* Weights in 8.8 fixed point: the sum of four products fits easily,
     * and it is a good deal cheaper than doing this in doubles per pixel. */
    unsigned wa = (unsigned)((fu - flu) * 256.0), wb = (unsigned)((fv - flv) * 256.0);
    unsigned w[4], all, k, c;
    uint32_t px[4], out;
    unsigned x0 = wrap_axis(u0, tw, tr->mirror & 2u), x1 = wrap_axis(u0 + 1, tw, tr->mirror & 2u);
    unsigned y0 = wrap_axis(v0, th, tr->mirror & 1u), y1 = wrap_axis(v0 + 1, th, tr->mirror & 1u);
    unsigned base = tr->page * 1024u;
    int t;

    t = sheet_word(ox + x0, ((oy + y0) & 1023u) + base); if (t < 0) return -1; px[0] = texel_argb(tr->fmt, (unsigned)t, tr->alpha_test);
    t = sheet_word(ox + x1, ((oy + y0) & 1023u) + base); if (t < 0) return -1; px[1] = texel_argb(tr->fmt, (unsigned)t, tr->alpha_test);
    t = sheet_word(ox + x0, ((oy + y1) & 1023u) + base); if (t < 0) return -1; px[2] = texel_argb(tr->fmt, (unsigned)t, tr->alpha_test);
    t = sheet_word(ox + x1, ((oy + y1) & 1023u) + base); if (t < 0) return -1; px[3] = texel_argb(tr->fmt, (unsigned)t, tr->alpha_test);

    /* Where the four are not all opaque, each takes its colour from the
     * most opaque of them, so a cut-out's clear texels do not bleed dark
     * into its edge. */
    all = px[0] & px[1] & px[2] & px[3];
    if ((all >> 24) != 255u) {
        unsigned best = 0;
        for (k = 1; k < 4; k++) if ((px[k] >> 24) > (px[best] >> 24)) best = k;
        for (k = 0; k < 4; k++)
            if ((px[k] >> 24) < (px[best] >> 24))
                px[k] = (px[k] & 0xFF000000u) | (px[best] & 0x00FFFFFFu);
    }
    w[0] = (256u - wa) * (256u - wb); w[1] = wa * (256u - wb);
    w[2] = (256u - wa) * wb;          w[3] = wa * wb;
    out = 0;
    for (c = 0; c < 32u; c += 8u) {
        unsigned acc = 0;
        for (k = 0; k < 4; k++) acc += w[k] * ((px[k] >> c) & 0xFFu);
        out |= ((acc + 32768u) >> 16) << c;
    }
    return (int64_t)out;
}

/* Trilinear: lod is log2 of how many base texels one pixel spans. Up to
 * the level whose smaller side is 8 texels, the last the chain stores. A
 * level that was never uploaded falls back to the base. */
static int64_t sample(const tri_t *tr, double u, double v, double lod)
{
    unsigned tw = tr->tw ? tr->tw : 32u, th = tr->th ? tr->th : 32u;
    unsigned top = 0, lv;
    int64_t a, b;
    double f;
    if (lod <= 0.0) return sample_level(tr, u, v, 0);
    while (((tw < th ? tw : th) >> (top + 1)) >= 8u) top++;
    if (lod >= (double)top) {
        a = sample_level(tr, u, v, top);
        return a < 0 ? sample_level(tr, u, v, 0) : a;
    }
    lv = (unsigned)lod;
    f = lod - lv;
    /* Blend two levels only across the middle half between them; near a
     * level, that level alone. Half the texel reads, and the seams between
     * levels stay as soft as anyone will see on a 384-line picture. */
    if (f < 0.25 || f > 0.75) {
        a = sample_level(tr, u, v, f < 0.25 ? lv : lv + 1u);
        return a < 0 ? sample_level(tr, u, v, 0) : a;
    }
    f = (f - 0.25) * 2.0;
    a = sample_level(tr, u, v, lv);
    b = sample_level(tr, u, v, lv + 1u);
    if (a < 0 || b < 0) return sample_level(tr, u, v, 0);
    {
        unsigned wb = (unsigned)(f * 256.0), c;
        uint32_t out = 0;
        for (c = 0; c < 32u; c += 8u)
            out |= ((((uint32_t)a >> c) & 0xFFu) * (256u - wb)
                    + (((uint32_t)b >> c) & 0xFFu) * wb + 128u) >> 8 << c;
        return (int64_t)out;
    }
}
static unsigned g_ntris;

/* The triangles raster() draws: one viewport's run of g_tris. */
static const tri_t *g_rtris;
static unsigned g_rn;

typedef struct { int on, lum, two_sided, alpha, alpha_test, tex_alpha, layer, mirror, fog, spec, shine; unsigned x, y, page, w, h, fmt; } texdesc_t;

static void emit(const float a[3], const float b[3], const float c[3],
                 const float n[3], uint32_t rgb,
                 const float na[3], const float nb[3], const float nc[3],
                 const float ua[2], const float ub[2], const float uc[2],
                 const texdesc_t *td)
{
    tri_t *t;
    if (g_ntris >= MAX_TRIS) return;
    t = &g_tris[g_ntris++];
    memcpy(t->v[0], a, sizeof t->v[0]);
    memcpy(t->v[1], b, sizeof t->v[1]);
    memcpy(t->v[2], c, sizeof t->v[2]);
    memcpy(t->n, n, sizeof t->n);
    memcpy(t->vn[0], na, sizeof t->vn[0]);
    memcpy(t->vn[1], nb, sizeof t->vn[1]);
    memcpy(t->vn[2], nc, sizeof t->vn[2]);
    memcpy(t->uv[0], ua, sizeof t->uv[0]);
    memcpy(t->uv[1], ub, sizeof t->uv[1]);
    memcpy(t->uv[2], uc, sizeof t->uv[2]);
    t->textured = (uint8_t)(td->on ? 1 : 0);
    t->luminous = (uint8_t)(td->lum ? 1 : 0);
    t->two_sided = (uint8_t)(td->two_sided ? 1 : 0);
    t->alpha = (uint8_t)td->alpha;
    t->fmt = (uint8_t)td->fmt;
    t->alpha_test = (uint8_t)td->alpha_test;
    t->tex_alpha = (uint8_t)td->tex_alpha;
    t->layer = (uint8_t)td->layer;
    t->mirror = (uint8_t)td->mirror;
    t->fog = (uint8_t)td->fog;
    t->spec = (uint8_t)td->spec;
    t->shine = (uint8_t)td->shine;
    t->tx = (uint16_t)td->x; t->ty = (uint16_t)td->y;
    t->tw = (uint16_t)td->w; t->th = (uint16_t)td->h;
    t->page = (uint8_t)td->page;
    t->r = (uint8_t)((rgb >> 16) & 0xFF);
    t->g = (uint8_t)((rgb >> 8) & 0xFF);
    t->b = (uint8_t)(rgb & 0xFF);
}

/* ---- models ------------------------------------------------------------ */
/* Texture offset inherited down the node tree (node word 2): added to every
 * polygon's texture origin below it, and its page bit flips the polygon's.
 * The same model is drawn with different skins this way. */
static unsigned g_texoff_x, g_texoff_y, g_texoff_page;
static uint32_t g_ctab;     /* colour table, a word index into polygon RAM */

static void draw_model(uint32_t addr, const mat_t *m)
{
    const uint8_t *mem;
    size_t n = 0;
    uint32_t (*rd)(const uint8_t *);
    size_t i;
    float prev[4][3], prevn[4][3], prevuv[4][2];
    unsigned nprev = 0;
    unsigned guard;

    if (addr & 0x800000u) {
        mem = bus_vrom(&n);
        rd = rd32be;                    /* VROM alone is read big-endian */
        i = (size_t)(addr & 0x7FFFFFu) * 4u;
    } else {
        mem = bus_poly(&n);
        rd = rd32le;
        i = (size_t)addr * 4u;
    }
    if (!mem) return;

    /* M3_MODEL_DUMP=<n>: the first n models drawn, with where they live,
     * how many polygons they have, the biggest raw coordinate in them and
     * the OR and AND of every polygon header word. Two models that want
     * different vertex scales differ in a header bit, and this is what
     * shows which. */
    {
        static long cap = -2;
        static unsigned told;
        static uint64_t m3_model_from;
        if (cap == -2) { const char *e = getenv("M3_MODEL_DUMP");
                         const char *f2 = getenv("M3_NODE_FROM");
                         m3_model_from = f2 ? strtoull(f2, NULL, 0) : 0;
                         cap = e ? strtol(e, NULL, 0) : 0; }
        if (cap > 0 && told < (unsigned)cap
            && model3recomp_frame_count() >= m3_model_from) {
            uint32_t hor[7], han[7];
            long biggest = 0;
            unsigned polys = 0, kk, g2;
            size_t j2 = i;
            for (kk = 0; kk < 7; kk++) { hor[kk] = 0; han[kk] = 0xFFFFFFFFu; }
            for (g2 = 0; g2 < 20000u; g2++) {
                uint32_t h2[7]; unsigned nv2, new2, q;
                if (j2 + 28u > n) break;
                for (kk = 0; kk < 7; kk++) h2[kk] = rd(mem + j2 + kk * 4);
                { float nn[3]; float l2;
                  for (kk = 0; kk < 3; kk++)
                      nn[kk] = (float)sx24(h2[kk + 1]) / 4194304.0f;
                  l2 = nn[0]*nn[0] + nn[1]*nn[1] + nn[2]*nn[2];
                  if (l2 < 0.98f || l2 > 1.02f) break; }
                for (kk = 0; kk < 7; kk++) { hor[kk] |= h2[kk]; han[kk] &= h2[kk]; }
                nv2 = (h2[0] & 0x40u) ? 4u : 3u;
                new2 = nv2;
                for (kk = 0; kk < 4; kk++) if ((h2[0] >> kk) & 1u) new2--;
                for (q = 0; q < new2; q++) {
                    size_t o2 = j2 + 28u + (size_t)q * 16u;
                    unsigned c2;
                    if (o2 + 12u > n) break;
                    for (c2 = 0; c2 < 3; c2++) {
                        long v2 = sx24(rd(mem + o2 + c2 * 4));
                        if (v2 < 0) v2 = -v2;
                        if (v2 > biggest) biggest = v2;
                    }
                }
                j2 += 28u + (size_t)new2 * 16u;
                polys++;
                if (h2[1] & 4u) break;
            }
            told++;
            fprintf(stderr, "[model] @%06X %4u polys  biggest raw %8ld"
                            "  (%.2f units at 1/%g)\n",
                    addr, polys, biggest,
                    (double)biggest / (double)g_vertex_scale,
                    (double)g_vertex_scale);
            fprintf(stderr, "          placed at %.1f %.1f %.1f\n",
                    (double)m->m[0][3], (double)m->m[1][3],
                    (double)m->m[2][3]);
            fprintf(stderr, "          hdr or  %08X %08X %08X %08X %08X %08X %08X\n",
                    hor[0], hor[1], hor[2], hor[3], hor[4], hor[5], hor[6]);
            fprintf(stderr, "          hdr and %08X %08X %08X %08X %08X %08X %08X\n",
                    han[0], han[1], han[2], han[3], han[4], han[5], han[6]);
        }
    }

    for (guard = 0; guard < 20000u; guard++) {
        uint32_t ph[7];
        float nrm[3], vs[4][3], vn[4][3], uv[4][2];
        texdesc_t td;
        unsigned nv, nnew, k, j, slot;
        uint32_t rgb;
        float len;

        if (i + 7u * 4u > n) return;
        for (k = 0; k < 7; k++) ph[k] = rd(mem + i + k * 4);

        /* Header words 1..3 are the polygon normal, 24-bit fixed over 2^22.
         * A real header's normal is a unit vector; this is the check that
         * catches a model pointer that is not a model, which otherwise
         * produces triangles rather than an error. */
        for (k = 0; k < 3; k++)
            nrm[k] = (float)sx24(ph[k + 1]) / 4194304.0f;
        len = nrm[0]*nrm[0] + nrm[1]*nrm[1] + nrm[2]*nrm[2];
        if (len < 0.98f || len > 1.02f) return;

        nv = (ph[0] & 0x40u) ? 4u : 3u;
        slot = 0;
        for (k = 0; k < 4; k++)             /* bits 0..3: reuse from previous */
            if (((ph[0] >> k) & 1u) && k < nprev && slot < 4) {
                memcpy(vs[slot], prev[k], sizeof vs[slot]);
                memcpy(vn[slot], prevn[k], sizeof vn[slot]);
                memcpy(uv[slot], prevuv[k], sizeof uv[slot]);
                slot++;
            }
        if (slot > nv) return;
        nnew = nv - slot;
        if (i + (7u + 4u * nnew) * 4u > n) return;

        for (k = 0; k < nnew && slot < 4; k++, slot++) {
            size_t o = i + (7u + 4u * k) * 4u;
            uint32_t wx = rd(mem + o), wy = rd(mem + o + 4),
                     wz = rd(mem + o + 8);
            vs[slot][0] = (float)sx24(wx) / g_vertex_scale;
            vs[slot][1] = (float)sx24(wy) / g_vertex_scale;
            vs[slot][2] = (float)sx24(wz) / g_vertex_scale;
            /* The low byte of each coordinate word is that vertex's own
             * normal, as a signed byte. They are not unit length -- the
             * lengths run from 0 to 127 -- but they do sit a median 25
             * degrees from the face normal, against the 90 that random
             * directions would give, which is what identifies them. */
            vn[slot][0] = (float)(int8_t)(wx & 0xFFu);
            vn[slot][1] = (float)(int8_t)(wy & 0xFFu);
            vn[slot][2] = (float)(int8_t)(wz & 0xFFu);
            {
                /* Word 3 of a vertex is u in the high half and v in the low,
                 * in eighths of a texel -- or whole texels when bit 6 of
                 * header word 1 is set (MAME: "UV format, 0 = 13.3,
                 * 1 = 16.0"). */
                uint32_t wt = rd(mem + o + 12);
                float us = (ph[1] & 0x40u) ? 1.0f : 0.125f;
                uv[slot][0] = (float)((wt >> 16) & 0xFFFFu) * us;
                uv[slot][1] = (float)(wt & 0xFFFFu) * us;
            }
        }

        rgb = (((ph[4] >> 24) & 0xFFu) << 16) | (((ph[4] >> 16) & 0xFFu) << 8)
            | ((ph[4] >> 8) & 0xFFu);
        /* Bit 1 of word 1 clear: the colour is an index into a table in
         * polygon RAM that a culling node above set up. */
        if (!(ph[1] & 2u)) {
            size_t pn = 0;
            const uint8_t *pr = bus_poly(&pn);
            size_t co = ((size_t)g_ctab + ((ph[4] >> 8) & 0xFFFu)) * 4u;
            if (pr && co + 4u <= pn) rgb = rd32le(pr + co) & 0xFFFFFFu;
        }

        /* The texture descriptor. Bit 26 of word 6 is the enable: it is set
         * on 1749 of the 1780 polygons in one model and clear on the rest.
         * The origin is in 32-texel units, which is also the tile size. */
        /* The texture descriptor, and three of these were wrong before.
         * They are now as MAME's draw_model() and its header documentation
         * read them.
         *
         * The enable is bit 10 of word 6, not bit 26. Width and height are
         * the low bits of word 3 -- the same word whose top 24 bits are the
         * polygon normal, which is why they were never found there. The page
         * is bit 6 of word 4, whose top 24 bits are the colour. Both size
         * fields read 6 or 7 to mean 32. */
        {
            uint32_t ws = (ph[3] >> 3) & 7u, hs = ph[3] & 7u;
            if (ws >= 6u) ws = 0u;
            if (hs >= 6u) hs = 0u;
            td.w = 32u << ws;
            td.h = 32u << hs;
        }
        td.fmt  = (ph[6] >> 7) & 7u;
        td.alpha_test = (ph[6] & 0x80000000u) != 0;
        td.tex_alpha  = (ph[6] & 7u) != 0;
        td.mirror     = ph[2] & 3u;
        td.on   = g_textures && (ph[6] & 0x400u) != 0;
        /* x is five bits of word 4 with its low bit borrowed from the top
         * of word 5, which looks odd and is right: reading it as six
         * contiguous bits of word 4, the way the upload header carries
         * it, drops a round from 89678 texels found to 18841. */
        td.x    = (32u * (((ph[4] & 0x1Fu) << 1)
                          | ((ph[5] >> 7) & 1u))) & 2047u;
        td.y    = (32u * (ph[5] & 0x1Fu)) & 2047u;
        td.page = ((ph[4] & 0x40u) >> 6) ^ g_texoff_page;
        td.x    = (td.x + g_texoff_x) & 2047u;
        td.y    = (td.y + g_texoff_y) & 1023u;
        /* Bit 16 of word 6 disables lighting for the polygon:
         * it is drawn at full brightness. */
        td.lum  = (ph[6] & 0x00010000u) != 0;
        /* Bits 11..15 of word 6, MAME's "light modifier", read here as
         * how much of the viewport's fog the polygon takes, in 16ths:
         * scenery carries 16, and it is scenery that fogs. */
        td.fog  = (int)((ph[6] >> 11) & 0x1Fu);
        /* Specular (MAME's documentation): word 0 bit 7 is the enable and
         * its top six bits the coefficient; word 6 bits 5..6 the shininess. */
        td.spec  = (ph[0] & 0x80u) && !g_no_spec ? (int)(ph[0] >> 26) : 0;
        td.shine = (int)((ph[6] >> 5) & 3u);
        /* Translucency, as MAME reads word 6: bit 23 set is opaque; clear,
         * bits 18..22 are the opacity in 32nds. The attract title
         * sequence puts a 40% mist sheet in front of the photographic
         * backdrop -- drawn opaque, it is 800 fields of flat grey. */
        td.alpha = (ph[6] & 0x00800000u) ? 255
                 : (int)(((ph[6] >> 18) & 0x1Fu) * 255u / 32u);
        if (td.alpha > 255) td.alpha = 255;
        /* Bit 17 picks which of the two translucency layers the polygon's
         * see-through pixels go into (MAME: "translucency pattern
         * select"); see composite_rows(). */
        td.layer = (ph[6] & 0x00020000u) != 0;
        td.two_sided = (ph[1] & 0x10u) != 0;
        if (td.w > 2048u) td.w = 2048u;
        if (td.h > 2048u) td.h = 2048u;

        /* Both discard bits set: the polygon is in the list but not drawn.
         * Its vertices still count for the next polygon's reuse. */
        if (slot >= 3 && (ph[0] & 0x300u) != 0x300u) {
            float wv[4][3], wn[3], wvn[4][3];
            for (j = 0; j < slot; j++) {
                mat_point(m, vs[j], wv[j]);
                mat_dir(m, vn[j], wvn[j]);
            }
            mat_dir(m, nrm, wn);
            for (j = 1; j + 1 < slot; j++)
                emit(wv[0], wv[j], wv[j+1], wn, rgb,
                     wvn[0], wvn[j], wvn[j+1],
                     uv[0], uv[j], uv[j+1], &td);
        }

        for (j = 0; j < slot; j++) {
            memcpy(prev[j], vs[j], sizeof prev[j]);
            memcpy(prevn[j], vn[j], sizeof prevn[j]);
            memcpy(prevuv[j], uv[j], sizeof prevuv[j]);
        }
        nprev = slot;

        i += (7u + 4u * nnew) * 4u;
        if (ph[1] & 4u) return;             /* last polygon of the model */
    }
}

/* ---- the node tree ----------------------------------------------------- */
static uint32_t g_matbase;

static void descend(uint32_t addr, const mat_t *m, int depth);

/* Which matrices a viewport actually uses. The table is sparse -- most
 * of it is zeros -- so "how many entries look valid" is not the question;
 * "which ones does the scene reach for" is. */
int m3_node_dump;
uint64_t m3_node_from;
static unsigned g_mat_loads, g_mat_max, g_mat_zero, g_mat_bigi;
static unsigned g_node_translate, g_node_matrix, g_node_neither;
static float    g_mat_bigt;


/* A link, as MAME's process_link() reads one: its top byte 0x00 is a
 * culling node, 0x01 or 0x03 a model, 0x04 a list of culling nodes. Zero,
 * 0x0FFFFFFF, 0x01000000 and the dummy 0x800800 are no link at all. */
static void follow(uint32_t p, const mat_t *m, int depth)
{
    uint32_t a = p & 0x00FFFFFFu;

    if (p == 0 || p == 0x0FFFFFFFu || p == 0x01000000u || a == 0x800800u)
        return;

    switch (p >> 24) {
    case 0x00:
        descend(a, m, depth + 1);
        break;
    case 0x01:
    case 0x03:
        draw_model(a, m);
        break;
    case 0x04: {
        /* The list runs to an entry with bit 25 set, which is the last, or
         * to one that is zero or has anything else in its top byte, which
         * is past the end. Drawn front to back: MAME walks it backwards,
         * which only changes the order translucent polygons blend in, and
         * nothing here says which the hardware does. */
        uint32_t ent[256];
        unsigned n = 0, k;
        if (depth >= MAX_DEPTH) return;
        while (n < 256u) {
            const uint8_t *e = cull_at(a + n, 1);
            uint32_t v;
            if (!e) break;
            v = rd32le(e);
            if (v & 0x02000000u) { ent[n++] = v; break; }
            if (v == 0 || (v >> 24) != 0) break;
            ent[n++] = v;
        }
        for (k = 0; k < n; k++) {
            uint32_t v = ent[k] & 0x00FFFFFFu;
            if (v && v != 0x800800u) descend(v, m, depth + 1);
        }
        break;
    }
    default:
        break;
    }
}

/* One culling node and then its siblings, as MAME's draw_block() has it:
 * word 0 bit 4 says the node carries a translation (words 4..6) rather than
 * a matrix index (word 3, low 12 bits); word 7 is the child link, or with
 * word 0 bit 3 a LOD table whose first entry is the model; word 8 is the
 * sibling link, absent when word 0's type is 6. The sibling takes the
 * parent's transform, not this node's. A sibling that is a culling node is
 * walked in a loop rather than recursed into -- a level of scenery is a long
 * sibling chain, and recursing on it ran into MAX_DEPTH. */
static void descend(uint32_t addr, const mat_t *m, int depth)
{
    unsigned chain;

    if (depth >= MAX_DEPTH) return;
    for (chain = 0; addr && chain < 4096u; chain++) {
        const uint8_t *node = cull_at(addr, 10);
        mat_t cur, mm;
        uint32_t mo, w0, child, sib;
        unsigned sx_ = g_texoff_x, sy_ = g_texoff_y, sp_ = g_texoff_page;
        float tx, ty, tz;

        if (!node) return;
        cur = *m;
        w0 = rd32le(node);
        sib = rd32le(node + 8 * 4);
        /* Word 0 bit 2: the node sets a colour table, its address spread
         * over the tops of words 3, 7 and 8 (MAME's node documentation). */
        if (w0 & 4u)
            g_ctab = ((rd32le(node + 3 * 4) >> 19)
                    | ((rd32le(node + 7 * 4) >> 28) << 13)
                    | ((rd32le(node + 8 * 4) >> 25) << 17)) & 0xFFFFFu;
        if ((w0 & 7u) == 6u) sib = 0;      /* no sibling link */

        /* Word 2: a texture offset for everything below, X in bits 7..13
         * and Y in 0..6 (in 32-texel units), bit 14 switching the page,
         * bit 15 turning it on. */
        {
            uint32_t w2 = rd32le(node + 2 * 4);
            if (w2 & 0x8000u) {
                g_texoff_x = 32u * ((w2 >> 7) & 0x3Fu);
                g_texoff_y = 32u * (w2 & 0x1Fu);
                g_texoff_page = (w2 >> 14) & 1u;
            }
        }

    mo = rd32le(node + 3 * 4) & 0xFFFu;
    tx = f32_of(rd32le(node + 4 * 4));
    ty = f32_of(rd32le(node + 5 * 4));
    tz = f32_of(rd32le(node + 6 * 4));

    /* A node carries EITHER a translation OR a matrix, and bit 4 of word
     * 0 says which. Applying both -- matrix first, then the translation
     * whenever it was non-zero -- is right for every node that has one
     * and wrong for every node that has the other, because a node with
     * the bit set still has something in its matrix field and it is not
     * a matrix index. */
    /* How the scene places things: a node carries either a translation or
     * a matrix, and this counts which. */
    if (w0 & 0x10u) g_node_translate++; else if (mo) g_node_matrix++;
    else g_node_neither++;
    if (w0 & 0x10u) {
        mat_t t = IDENT;
        t.m[0][3] = tx; t.m[1][3] = ty; t.m[2][3] = tz;
        cur = mat_mul(m, &t);
    } else if (mo && mat_load(g_matbase, mo, &mm)) {
        unsigned k;
        float big = 0.0f;
        for (k = 0; k < 3; k++) {
            float v = mm.m[k][3] < 0 ? -mm.m[k][3] : mm.m[k][3];
            if (v > big) big = v;
        }
        g_mat_loads++;
        { static unsigned told; extern int m3_node_dump;
          extern uint64_t m3_node_from;
          if (m3_node_dump && model3recomp_frame_count() >= m3_node_from
              && told++ < (unsigned)m3_node_dump)
            fprintf(stderr, "[node] @%06X w0=%08X w1=%08X w2=%08X"
                            " w3=%08X w9=%08X mo=%u"
                            "  |rows| %.3g %.3g %.3g  t %.4g %.4g %.4g\n",
                    addr, rd32le(node), rd32le(node + 1 * 4),
                    rd32le(node + 2 * 4), rd32le(node + 3 * 4),
                    rd32le(node + 9 * 4), mo,
                    (double)sqrtf(mm.m[0][0]*mm.m[0][0]+mm.m[0][1]*mm.m[0][1]+mm.m[0][2]*mm.m[0][2]),
                    (double)sqrtf(mm.m[1][0]*mm.m[1][0]+mm.m[1][1]*mm.m[1][1]+mm.m[1][2]*mm.m[1][2]),
                    (double)sqrtf(mm.m[2][0]*mm.m[2][0]+mm.m[2][1]*mm.m[2][1]+mm.m[2][2]*mm.m[2][2]),
                    (double)mm.m[0][3], (double)mm.m[1][3], (double)mm.m[2][3]); }
        if (mo > g_mat_max) g_mat_max = mo;
        if (big > g_mat_bigt) { g_mat_bigt = big; g_mat_bigi = mo; }
        if (mm.m[0][0] == 0.0f && mm.m[0][1] == 0.0f && mm.m[0][2] == 0.0f
         && mm.m[1][0] == 0.0f && mm.m[1][1] == 0.0f && mm.m[1][2] == 0.0f)
            g_mat_zero++;
        cur = mat_mul(m, &mm);
    }

        child = rd32le(node + 7 * 4);
        if (w0 & 0x08u) {
            const uint8_t *lod = cull_at(child & 0x00FFFFFFu, 1);
            if (lod) draw_model(rd32le(lod) & 0x00FFFFFFu, &cur);
        } else {
            follow(child, &cur, depth);
        }

        g_texoff_x = sx_; g_texoff_y = sy_; g_texoff_page = sp_;
        /* A sibling that is anything but a culling node is followed as a
         * link and ends the chain. 0x800800 is the dummy node, and an
         * unused node is all dummy words, so its sibling is itself: stop
         * rather than spin. */
        if (sib >> 24) { follow(sib, m, depth); break; }
        addr = (sib == addr || sib == 0x800800u) ? 0 : sib;
    }
}

/* ---- rasteriser -------------------------------------------------------- */
static _Thread_local unsigned g_drawn;
static float *g_zbuf;
static int g_zw, g_zh;
/* The two translucency layers: per pixel, the nearest see-through pixel of
 * each pattern drawn so far this field -- its colour, with its opacity in the
 * top byte and 0 for none -- and its depth. See composite_rows(). */
static uint32_t *g_tcol[2];
static float *g_tz[2];

#define ZNEAR 0.05f

/* Sutherland-Hodgman against z >= ZNEAR. Three input vertices give at most
 * four out, so the caller's buffer is [4][3]. */
static int clip_near(const float v[3][5], float out[4][5])
{
    int n = 0, i, k;
    for (i = 0; i < 3; i++) {
        const float *a = v[i];
        const float *b = v[(i + 1) % 3];
        int ain = a[2] >= ZNEAR, bin = b[2] >= ZNEAR;
        if (ain && n < 4) {
            for (k = 0; k < 5; k++) out[n][k] = a[k];
            n++;
        }
        if (ain != bin && n < 4) {
            float t = (ZNEAR - a[2]) / (b[2] - a[2]);
            for (k = 0; k < 5; k++) out[n][k] = a[k] + t * (b[k] - a[k]);
            n++;
        }
    }
    return n;
}

/* Thread-local: each band of the screen is rasterised on its own thread
 * (see raster_bands), and these are diagnostics. With M3_REAL3D_STATS or
 * M3_PICK set the whole screen is one band, so they stay exact. */
static _Thread_local unsigned g_cull_back, g_cull_near, g_cull_degen, g_cull_off;
static _Thread_local unsigned g_tex_tris, g_flat_tris, g_tex_px, g_flat_px;
static _Thread_local unsigned g_texel_hit, g_texel_clear, g_texel_missing;
/* M3_PICK=x,y: which polygon last wrote that pixel, and its texture. */
static long g_pick = -1;
static _Thread_local long g_pick_tri = -1;
/* Where the polygons that find nothing are looking. */
enum { MISS_SLOTS = 24 };
static _Thread_local struct { unsigned x, y, page, w, h, n; } g_miss[MISS_SLOTS];
static void note_miss(const tri_t *t)
{
    unsigned k;
    for (k = 0; k < MISS_SLOTS; k++) {
        if (g_miss[k].n && (g_miss[k].x != t->tx || g_miss[k].y != t->ty
                            || g_miss[k].page != t->page))
            continue;
        g_miss[k].x = t->tx; g_miss[k].y = t->ty; g_miss[k].page = t->page;
        g_miss[k].w = t->tw; g_miss[k].h = t->th; g_miss[k].n++;
        return;
    }
}

/* Where a viewport's view space lands on the screen: pixel = a * (X/Z) + b,
 * and the rectangle it may draw in. */
typedef struct { double ax, bx, ay, by; int x0, y0, x1, y1; } proj_t;
static proj_t g_proj;

/* A viewport's fog. MAME documents the words: 0x22 the fog colour, 0x23
 * the density (a float), 0x25 the "fog offset" in its low half and the
 * ambient fog in bits 16..23; and polygon word 6 bits 11..15 as the
 * "light modifier", how far that polygon burns through the fog.
 *
 * The model here: fog thickens linearly with view distance from the offset,
 *     f = clamp(offset + density * z, 0, 1) * modifier / 16
 * and a polygon blends toward the colour, darkened by the ambient fog, by f.
 * The game ramps the offset from 0.5 to 1 with a black fog colour to fade a
 * scene out, which only reads as a fade if the offset is a share of the
 * whole and not a distance. */
typedef struct { int on; float r, g, b, density, start; } fog_t;
static fog_t g_fog;

/* Narrow [*xs, *xe] to where a x + c >= 0, give or take a pixel. Zero when
 * nothing is left. */
static int span_x(double a, double c, int *xs, int *xe)
{
    if (a > 1e-12) {
        double b = -c / a - 1.0;
        if (b > *xs) { if (b > *xe) return 0; *xs = (int)b; }
    } else if (a < -1e-12) {
        double b = -c / a + 1.0;
        if (b < *xe) { if (b < *xs) return 0; *xe = (int)b; }
    } else if (c < -1e-9)
        return 0;
    return *xs <= *xe;
}

/* log2 to a few thousandths, which is plenty for picking a mip level: the
 * exponent, and a quadratic through the mantissa. */
static float fast_log2(float x)
{
    union { float f; uint32_t i; } v;
    float e, m;
    v.f = x;
    e = (float)((int)((v.i >> 23) & 0xFFu) - 127);
    v.i = (v.i & 0x007FFFFFu) | 0x3F800000u;
    m = v.f;
    return e + (-0.34484843f * m + 2.02466578f) * m - 0.67487759f;
}

/* ---- triangle setup ------------------------------------------------------
 *
 * Culling, the near clip, projection and the per-triangle constants, once
 * per viewport, into a list of screen triangles in drawing order: opaque
 * first, then translucent. raster() then only fills pixels, so cutting the
 * screen into many slices for the threads costs nothing per slice beyond a
 * y-range test, and the setup runs on the guest's thread while the last
 * field's picture is still being drawn. */
typedef struct {
    int y0, y1, x0, x1;             /* the screen box, clipped to the viewport */
    const tri_t *tr;
    unsigned src;                   /* index in the viewport's run, for M3_PICK */
    uint32_t argb;                  /* flat shade */
    double p[3][3], q[3][3], gq[3][2], invd;
} stri_t;

/* A viewport the walk found: its run of triangles and everything raster()
 * reads, so that it can be drawn later, and by several threads at once. */
typedef struct {
    unsigned first, count;          /* the viewport's run of g_tris */
    unsigned s0, s1;                /* its screen triangles, st[s0 .. s1) */
    const stri_t *st;
    proj_t proj;
    fog_t fog;
    float ambient, sx, sy, light[3], intensity;
    uint32_t fogw[3];               /* viewport words 0x20, 0x22, 0x25, for scroll_fog() */
} vpjob_t;

/* Everything a field's walk produces, twice over: deferred, the guest walks
 * the next field into one set while the other is still being drawn. */
typedef struct {
    tri_t *tris;
    stri_t *stris;
    unsigned cap_stris;
    vpjob_t jobs[64];               /* 16 viewports a priority, four priorities */
    unsigned njobs;
    uint8_t *tex;                   /* texture uploads queued for this field */
    size_t ntex, cap_tex;
} fieldset_t;
static fieldset_t g_set[2];
static int g_col;                   /* the set being walked into */
static stri_t *g_stris;             /* g_set[g_col]'s, while walking */
static unsigned g_nstris, g_cap_stris;

static void setup_tris(int fw, int fh, const float L[3], float intensity)
{
    unsigned t;

    /* Where the geometry actually is, in view space. A scene that comes
     * out empty is usually not a rasteriser problem: it is geometry that
     * never reaches the screen, and the bounds say so in one line. */
    if (getenv("M3_REAL3D_STATS") && g_rn) {
        float lo[3], hi[3];
        unsigned i, k, behind = 0;
        for (k = 0; k < 3; k++) { lo[k] = 1e30f; hi[k] = -1e30f; }
        for (i = 0; i < g_rn; i++) {
            unsigned j;
            for (j = 0; j < 3; j++) {
                for (k = 0; k < 3; k++) {
                    float v = g_rtris[i].v[j][k];
                    if (v < lo[k]) lo[k] = v;
                    if (v > hi[k]) hi[k] = v;
                }
                if (g_rtris[i].v[j][2] <= 0.0f) behind++;
            }
        }
        fprintf(stderr, "[real3d]     view bounds x %.0f..%.0f  y %.0f..%.0f"
                        "  z %.0f..%.0f  %u/%u verts at z<=0\n",
                lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], behind, g_rn * 3u);
        g_cull_back = g_cull_near = g_cull_degen = g_cull_off = 0;
    }

    /* Opaque first, then the translucent ones in submission order, so that
     * every see-through pixel is tested against the whole opaque scene on
     * its way into its translucency layer (composite_rows). A polygon whose
     * texels carry alpha (word 6's translucency mode) counts as
     * translucent: smoke, glass and shadows are made of those. Its fully
     * opaque texels are drawn as opaque pixels all the same. */
    for (t = 0; t < 2u * g_rn; t++) {
        const tri_t *tr = &g_rtris[t % g_rn];
        /* Screen-space vertices in double. A projected triangle here can
         * reach coordinates in the hundreds of thousands -- the scene runs
         * well past the edges of a 496-pixel screen -- and in float the
         * barycentric denominator then loses enough precision that pixels
         * inside the triangle test as outside. It does not look like a
         * precision bug: it looks like a cross-hatch, as though the polygon
         * were deliberately stippled for translucency. */
        double p[3][3], q[3][3], gq[3][2];
        float clipped[4][5], src[3][5];
        double d, invd;
        float lit, sh;
        if ((tr->alpha < 255 || (tr->textured && tr->tex_alpha)) != (t >= g_rn)) continue;
        int nclip, fan;
        int x0, x1, y0, y1, k;
        int r, g, b;
        uint32_t argb;

        /* Backface culling.
         *
         * Safe here because the winding is consistent: in a dumped model all
         * 588 polygons agree in sign between their header normal and the
         * normal computed from their own vertices, every one of them. So a
         * polygon faces away exactly when its normal points along the ray
         * from the eye to it, and there is no need to guess at a
         * double-sided flag to avoid throwing away visible surfaces.
         *
         * It removes about five triangles in six -- 2169 of 2617 in one
         * frame -- and the pixel count barely moves (105903 to 105527),
         * which is what culling only hidden geometry looks like. */
        {
            float cx = (tr->v[0][0] + tr->v[1][0] + tr->v[2][0]) / 3.0f;
            float cy = (tr->v[0][1] + tr->v[1][1] + tr->v[2][1]) / 3.0f;
            float cz = (tr->v[0][2] + tr->v[1][2] + tr->v[2][2]) / 3.0f;
            if (!tr->two_sided && tr->n[0]*cx + tr->n[1]*cy + tr->n[2]*cz > 0.0f)
                { g_cull_back++; continue; }
        }

        /* Cut against the near plane rather than dropping the triangle.
         * A triangle that straddles the eye plane cannot be projected, but
         * discarding it throws away exactly the large near-camera polygons
         * -- the ones that would fill the screen -- so a scene can come out
         * completely empty while every individual step looks correct. */
        for (k = 0; k < 3; k++) {
            src[k][0] = tr->v[k][0]; src[k][1] = tr->v[k][1];
            src[k][2] = tr->v[k][2];
            src[k][3] = tr->uv[k][0]; src[k][4] = tr->uv[k][1];
        }
        nclip = clip_near(src, clipped);
        if (nclip < 3) { g_cull_near++; continue; }

        for (fan = 1; fan + 1 < nclip; fan++) {
            const float *cv[3];
            cv[0] = clipped[0]; cv[1] = clipped[fan]; cv[2] = clipped[fan + 1];

        for (k = 0; k < 3; k++) {
            double iz = 1.0 / cv[k][2];
            p[k][0] = g_proj.ax * (double)cv[k][0] * iz + g_proj.bx;
            p[k][1] = g_proj.ay * (double)cv[k][1] * iz + g_proj.by;
            p[k][2] = cv[k][2];
            /* Perspective correction: interpolate u/z, v/z and 1/z linearly
             * in screen space and divide at the end. Interpolating u and v
             * directly is visibly wrong on polygons this large -- a ground
             * plane running to the horizon is exactly the case it breaks. */
            q[k][0] = (double)cv[k][3] * iz;
            q[k][1] = (double)cv[k][4] * iz;
            q[k][2] = iz;
        }

        d = (p[1][1]-p[2][1])*(p[0][0]-p[2][0])
          + (p[2][0]-p[1][0])*(p[0][1]-p[2][1]);
        if (d > -1e-9 && d < 1e-9) { g_cull_degen++; continue; }

        /* Flat shade as the fallback, for a polygon whose vertex normals
         * are degenerate. One-sided now that backfaces are culled: with the
         * real light the term runs from -0.72 to +0.93 across a frame, so
         * roughly two triangles in five are in shadow and the picture has
         * some range in it. Taking the absolute value, which is what this
         * did before, lit both sides equally and flattened that out. */
        lit = tr->n[0]*L[0] + tr->n[1]*L[1] + tr->n[2]*L[2];
        if (lit < 0) lit = 0;
        if (lit > 1.0f) lit = 1.0f;
        sh = tr->luminous ? intensity
                          : (AMBIENT + (1.0f - AMBIENT) * lit) * intensity;
        r = (int)(tr->r * sh); g = (int)(tr->g * sh); b = (int)(tr->b * sh);
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;
        argb = 0xFF000000u | ((uint32_t)r << 16) | ((uint32_t)g << 8)
             | (uint32_t)b;

        {
            double lo, hi;
            lo = p[0][0] < p[1][0] ? p[0][0] : p[1][0];
            if (p[2][0] < lo) lo = p[2][0];
            hi = p[0][0] > p[1][0] ? p[0][0] : p[1][0];
            if (p[2][0] > hi) hi = p[2][0];
            x0 = lo < 0 ? 0 : (lo > fw - 1 ? fw - 1 : (int)lo);
            x1 = hi < 0 ? 0 : (hi > fw - 1 ? fw - 1 : (int)hi + 1);
            if (x1 > fw - 1) x1 = fw - 1;
            lo = p[0][1] < p[1][1] ? p[0][1] : p[1][1];
            if (p[2][1] < lo) lo = p[2][1];
            hi = p[0][1] > p[1][1] ? p[0][1] : p[1][1];
            if (p[2][1] > hi) hi = p[2][1];
            y0 = lo < 0 ? 0 : (lo > fh - 1 ? fh - 1 : (int)lo);
            y1 = hi < 0 ? 0 : (hi > fh - 1 ? fh - 1 : (int)hi + 1);
            if (y1 > fh - 1) y1 = fh - 1;
        }
        if (x0 < g_proj.x0) x0 = g_proj.x0;
        if (y0 < g_proj.y0) y0 = g_proj.y0;
        if (x1 > g_proj.x1) x1 = g_proj.x1;
        if (y1 > g_proj.y1) y1 = g_proj.y1;
        if (x1 < x0 || y1 < y0) { g_cull_off++; continue; }
        if (tr->textured) g_tex_tris++; else g_flat_tris++;

        /* M3_TRI_MAX=n: drop any triangle whose screen bounding box is
         * bigger than n pixels, and say what it was. A scene that comes
         * out as two flat blocks is a scene with two polygons in front
         * of it; this is how you find out which two. */
        {
            static long cap = -2;
            long area = (long)(x1 - x0 + 1) * (long)(y1 - y0 + 1);
            if (cap == -2) { const char *e = getenv("M3_TRI_MAX");
                             cap = e ? strtol(e, NULL, 0) : -1; }
            if (cap > 0 && area > cap) {
                static unsigned told;
                if (told++ < 24)
                    fprintf(stderr, "[real3d] huge tri %ld px  rgb %02X%02X%02X"
                            "  tex %d lum %d  z %.1f %.1f %.1f"
                            "  v0 %.0f,%.0f,%.0f\n",
                            area, tr->r, tr->g, tr->b, tr->textured,
                            tr->luminous, p[0][2], p[1][2], p[2][2],
                            tr->v[0][0], tr->v[0][1], tr->v[0][2]);
                continue;
            }
        }

        invd = 1.0 / d;
        /* How u/z, v/z and 1/z change per pixel, for picking a mip level:
         * the barycentrics are linear in x and y, so these are constant
         * over the triangle. */
        {
            double l0x = (p[1][1]-p[2][1]) * invd, l0y = (p[2][0]-p[1][0]) * invd;
            double l1x = (p[2][1]-p[0][1]) * invd, l1y = (p[0][0]-p[2][0]) * invd;
            for (k = 0; k < 3; k++) {
                gq[k][0] = l0x * (q[0][k] - q[2][k]) + l1x * (q[1][k] - q[2][k]);
                gq[k][1] = l0y * (q[0][k] - q[2][k]) + l1y * (q[1][k] - q[2][k]);
            }
        }
        if (g_nstris >= g_cap_stris) {
            unsigned cap = g_cap_stris ? g_cap_stris * 2u : 8192u;
            stri_t *n = (stri_t *)realloc(g_stris, sizeof(stri_t) * cap);
            if (!n) continue;
            g_stris = n; g_cap_stris = cap;
        }
        {
            stri_t *st = &g_stris[g_nstris++];
            st->y0 = y0; st->y1 = y1; st->x0 = x0; st->x1 = x1;
            st->tr = tr; st->src = t % g_rn; st->argb = argb;
            memcpy(st->p, p, sizeof st->p);
            memcpy(st->q, q, sizeof st->q);
            memcpy(st->gq, gq, sizeof st->gq);
            st->invd = invd;
        }
        }
    }
}

static void raster(const vpjob_t *j, uint32_t *fb, int fw, int by0, int by1)
{
    /* The light comes from the viewport now, not from an invention.
     * Words 0x04..0x06 are a unit vector -- (-0.72288, -0.16226, -0.67156)
     * in this game, length 1.0000 to five places, and the same in every
     * scene dumped -- which is what identifies them. Word 0x07 sits with
     * Word 0x07 is a global brightness multiplier, and it took a second
     * look to see it. It was dismissed once for reading 0.93 in one attract
     * shot and 0.06 in another that was plainly not six per cent as bright
     * -- which was two moments compared without asking what happened
     * between them. Sampled every field it is unmistakable: it holds at
     * exactly 1.0000 for a thousand fields, ramps down in even steps to
     * 0.0000, holds there through the gap between demo shots, and ramps
     * back. It is the cross-fade.
     *
     * Ignoring it is what made the renderer look frozen: the game spent
     * twenty fields fading a shot to black and the picture stayed at full
     * brightness, byte-identical, while the scene underneath it changed
     * every field. */
    unsigned si;
    for (si = j->s0; si < j->s1; si++) {
        const stri_t *st = &j->st[si];
        const tri_t *tr = st->tr;
        const double (*p)[3] = st->p, (*q)[3] = st->q, (*gq)[2] = st->gq;
        const double invd = st->invd;
        const uint32_t argb = st->argb;
        int x0 = st->x0, x1 = st->x1, y0 = st->y0, y1 = st->y1, x, y;
        if (y0 < by0) y0 = by0;
        if (y1 > by1) y1 = by1;
        if (y1 < y0) continue;
        /* The barycentrics are linear in x along a row: l = a x + c. Each
         * row's span where all three are non-negative is solved once, a
         * pixel wider each side, so most of a thin or slanted triangle's
         * box is never visited; the per-pixel test still decides. */
        const double a0 = (p[1][1]-p[2][1]) * invd, b0 = (p[2][0]-p[1][0]) * invd;
        const double a1 = (p[2][1]-p[0][1]) * invd, b1 = (p[0][0]-p[2][0]) * invd;
        for (y = y0; y <= y1; y++) {
            const double c0 = b0 * (y - p[2][1]) - a0 * p[2][0];
            const double c1 = b1 * (y - p[2][1]) - a1 * p[2][0];
            int xs = x0, xe = x1;
            if (!span_x(a0, c0, &xs, &xe) || !span_x(a1, c1, &xs, &xe)
                || !span_x(-(a0 + a1), 1.0 - c0 - c1, &xs, &xe))
                continue;
            for (x = xs; x <= xe; x++) {
                double l0, l1, l2, z;
                int o;
                l0 = a0 * x + c0;
                l1 = a1 * x + c1;
                l2 = 1.0f - l0 - l1;
                if (l0 < 0 || l1 < 0 || l2 < 0) continue;
                z = l0*p[0][2] + l1*p[1][2] + l2*p[2][2];
                o = y * fw + x;
                /* Ties go to the later polygon, as in MAME's renderer: a
                 * HUD draws a boss's health over its red bar at the same
                 * depth, and first-wins left only rounding specks of it. */
                if (z <= g_zbuf[o]) {
                    uint32_t c = argb;
                    unsigned pa = tr->alpha;    /* this pixel's opacity */
                    if (tr->textured) {
                        double iz = l0*q[0][2] + l1*q[1][2] + l2*q[2][2];
                        if (iz > 1e-12) {
                            double uu = (l0*q[0][0] + l1*q[1][0] + l2*q[2][0]) / iz;
                            double vv = (l0*q[0][1] + l1*q[1][1] + l2*q[2][1]) / iz;
                            double lod = 0.0;
                            int64_t w;
                            if (!g_no_mips) {
                                /* d(U/W) = (dU - u dW) / W */
                                double ux = (gq[0][0] - uu * gq[2][0]) / iz, uy = (gq[0][1] - uu * gq[2][1]) / iz;
                                double vx = (gq[1][0] - vv * gq[2][0]) / iz, vy = (gq[1][1] - vv * gq[2][1]) / iz;
                                double rx = ux*ux + vx*vx, ry = uy*uy + vy*vy;
                                double rho = rx > ry ? rx : ry;
                                if (rho > 1.0) lod = 0.5f * fast_log2((float)rho);
                            }
                            w = sample(tr, uu, vv, lod);
                            if (w < 0) { g_texel_missing++; note_miss(tr); }
                            else {
                                uint32_t t = (uint32_t)w;
                                /* Format 0's transparent bit has always
                                 * been honoured here; the others only
                                 * when the polygon asks for it. */
                                if ((tr->fmt == 0 || tr->alpha_test) && (t >> 24) < 128u)
                                    { g_texel_clear++; continue; }
                                /* Texel alpha blends; next to nothing is
                                 * dropped rather than blended. */
                                if (tr->tex_alpha) {
                                    if ((t >> 24) < 8u) { g_texel_clear++; continue; }
                                    pa = pa * (t >> 24) / 255u;
                                }
                                /* Modulated by the polygon colour: that is
                                 * what tints the greyscale formats. */
                                c = 0xFF000000u
                                  | ((((t >> 16) & 0xFFu) * tr->r / 255u) << 16)
                                  | ((((t >> 8) & 0xFFu) * tr->g / 255u) << 8)
                                  | ((t & 0xFFu) * tr->b / 255u);
                                g_texel_hit++;
                            }
                        }
                    }
                    /* Gouraud: interpolate the vertex normals across the
                     * triangle. The hardware shades this way, and the
                     * faceting without it is the most visible thing wrong
                     * with a rounded surface like a boulder. */
                    float nx = (float)(l0*tr->vn[0][0] + l1*tr->vn[1][0]
                                     + l2*tr->vn[2][0]);
                    float ny = (float)(l0*tr->vn[0][1] + l1*tr->vn[1][1]
                                     + l2*tr->vn[2][1]);
                    float nz = (float)(l0*tr->vn[0][2] + l1*tr->vn[1][2]
                                     + l2*tr->vn[2][2]);
                    float nl = nx*nx + ny*ny + nz*nz;
                    if (nl > 1e-6f) {
                        float il = 1.0f / sqrtf(nl);
                        float d = (nx*j->light[0] + ny*j->light[1] + nz*j->light[2]) * il;
                        float s2;
                        int rr, gg, bb;
                        if (d < 0) d = 0;
                        if (d > 1.0f) d = 1.0f;
                        s2 = tr->luminous ? j->intensity
                           : (j->ambient + (1.0f - j->ambient) * d) * j->intensity;
                        rr = (int)(tr->r * s2); gg = (int)(tr->g * s2);
                        bb = (int)(tr->b * s2);
                        if (rr > 255) rr = 255;
                        if (gg > 255) gg = 255;
                        if (bb > 255) bb = 255;
                        if (tr->textured) {
                            rr = (int)(((c >> 16) & 0xFF) * s2);
                            gg = (int)(((c >> 8) & 0xFF) * s2);
                            bb = (int)((c & 0xFF) * s2);
                        }
                        if (tr->spec && !tr->luminous) {
                            /* Blinn: the normal against the half-way vector
                             * between the light and the eye, which looks
                             * down +z, raised to a power that doubles with
                             * each step of shininess, white, scaled by the
                             * coefficient over 63 and the sun's strength. */
                            float hx = j->light[0], hy = j->light[1], hz = j->light[2] - 1.0f;
                            float hl = hx*hx + hy*hy + hz*hz;
                            float nh = hl > 1e-12f ? (nx*hx + ny*hy + nz*hz) * il / sqrtf(hl) : 0.0f;
                            if (nh > 0.0f) {
                                float sp = powf(nh, (float)(16 << tr->shine)) * (float)tr->spec / 63.0f
                                         * (j->intensity < 1.0f ? j->intensity : 1.0f);
                                int add = (int)(sp * 255.0f);
                                rr += add; gg += add; bb += add;
                                if (rr > 255) rr = 255;
                                if (gg > 255) gg = 255;
                                if (bb > 255) bb = 255;
                            }
                        }
                        c = 0xFF000000u | ((uint32_t)rr << 16)
                          | ((uint32_t)gg << 8) | (uint32_t)bb;
                    }
                    if (j->fog.on && tr->fog) {
                        double w = l0*q[0][2] + l1*q[1][2] + l2*q[2][2];   /* 1/z */
                        double f = j->fog.start + (w > 1e-12 ? j->fog.density / w : 1.0);
                        f = (f < 0.0 ? 0.0 : f > 1.0 ? 1.0 : f) * tr->fog / 16.0;
                        if (f > 1.0) f = 1.0;
                        if (f > 0.0) {
                            float k = (float)f;
                            uint32_t m = 0xFF000000u;
                            int sh3;
                            const float fc[3] = { j->fog.b, j->fog.g, j->fog.r };
                            for (sh3 = 0; sh3 < 3; sh3++) {
                                float v = (float)((c >> (8 * sh3)) & 0xFFu);
                                m |= (uint32_t)(v + (fc[sh3] - v) * k + 0.5f) << (8 * sh3);
                            }
                            c = m;
                        }
                    }
                    if (pa < 255 && !g_old_trans) {
                        /* A see-through pixel goes into its polygon's
                         * layer, not onto the picture, and only if it is
                         * nearer than what that layer already holds here:
                         * the nearest wins and nothing accumulates. Under
                         * 1/32 is nothing at all. The depth buffer is the
                         * opaque scene's alone, so a translucent head no
                         * longer needs to write it to keep its inside
                         * from showing over its face. */
                        const unsigned L = tr->layer;
                        if (pa < 8u || !(z < g_tz[L][o])) continue;
                        g_tz[L][o] = (float)z;
                        g_tcol[L][o] = ((uint32_t)pa << 24) | (c & 0x00FFFFFFu);
                    } else if (pa < 255) {
                        /* M3_OLD_TRANSLUCENCY: each polygon blended over
                         * the last, writing depth from half opaque up. */
                        uint32_t a = pa, d0 = fb[o], m = 0;
                        int sh2;
                        for (sh2 = 0; sh2 < 24; sh2 += 8)
                            m |= ((((c >> sh2) & 0xFFu) * a
                                  + ((d0 >> sh2) & 0xFFu) * (255u - a)) / 255u) << sh2;
                        fb[o] = 0xFF000000u | m;
                        if (a >= 128u) g_zbuf[o] = (float)z;
                    } else {
                        g_zbuf[o] = (float)z; fb[o] = c;
                        /* What an opaque pixel covers of the layers is
                         * gone; what is in front of it stays. */
                        if (!(g_tz[0][o] < z)) g_tcol[0][o] = 0;
                        if (!(g_tz[1][o] < z)) g_tcol[1][o] = 0;
                    }
                    g_drawn++;
                    if (tr->textured) g_tex_px++; else g_flat_px++;
                    if (o == g_pick) g_pick_tri = (long)st->src;
                }
            }
        }
    }
}

/* ---- rasterising on every core -----------------------------------------
 *
 * The rasteriser is software and was the whole frame budget on its own in a
 * busy scene. The screen is cut into horizontal bands, one per thread, and
 * each thread draws every triangle clipped to its rows, in the same order --
 * so the picture is byte-identical to drawing it on one thread, the depth
 * buffer and the translucent pass included. Nothing here feeds back into
 * the guest, so netplay and save states are untouched.
 */
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
typedef SRWLOCK            m3_lock;
typedef CONDITION_VARIABLE m3_cond;
#  define LOCK(l)       AcquireSRWLockExclusive(l)
#  define UNLOCK(l)     ReleaseSRWLockExclusive(l)
#  define WAIT(c, l)    SleepConditionVariableSRW(c, l, INFINITE, 0)
#  define WAKE_ALL(c)   WakeAllConditionVariable(c)
#else
#  include <pthread.h>
#  include <unistd.h>
typedef pthread_mutex_t m3_lock;
typedef pthread_cond_t  m3_cond;
#  define LOCK(l)       pthread_mutex_lock(l)
#  define UNLOCK(l)     pthread_mutex_unlock(l)
#  define WAIT(c, l)    pthread_cond_wait(c, l)
#  define WAKE_ALL(c)   pthread_cond_broadcast(c)
#endif

#define MAX_BANDS 16
static struct {
    int      bands;             /* threads + 1: the caller draws band 0 */
    m3_lock  lock;
    m3_cond  go, done;
    unsigned gen, pending;
    void   (*fn)(int band);     /* what each thread does this round */
    const vpjob_t *job;
    uint32_t *fb; int fw, fh;
    int      y0, y1;            /* the rows being banded */
} g_pool;

/* Thread i draws every bands-th slice of SLICE rows, starting from its own:
 * one solid strip each left most threads idle while the one with the busy
 * part of the screen finished. */
#define SLICE 8

/* One viewport, every thread a share of its rows. */
static void band_draw(int i)
{
    int y;
    for (y = g_pool.y0 + i * SLICE; y <= g_pool.y1; y += g_pool.bands * SLICE) {
        int y1 = y + SLICE - 1;
        raster(g_pool.job, g_pool.fb, g_pool.fw, y, y1 < g_pool.y1 ? y1 : g_pool.y1);
    }
}

#ifdef _WIN32
static DWORD WINAPI band_worker(LPVOID arg)
#else
static void *band_worker(void *arg)
#endif
{
    int i = (int)(intptr_t)arg;
    unsigned seen = 0;
    for (;;) {
        LOCK(&g_pool.lock);
        while (g_pool.gen == seen) WAIT(&g_pool.go, &g_pool.lock);
        seen = g_pool.gen;
        UNLOCK(&g_pool.lock);
        g_pool.fn(i);
        LOCK(&g_pool.lock);
        if (--g_pool.pending == 0) WAKE_ALL(&g_pool.done);
        UNLOCK(&g_pool.lock);
    }
#ifdef _WIN32
    return 0;
#else
    return NULL;
#endif
}

static void pool_start(void)
{
    int n = 0, i;
    const char *e = getenv("M3_RENDER_THREADS");
#ifdef _WIN32
    n = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    InitializeSRWLock(&g_pool.lock);
    InitializeConditionVariable(&g_pool.go);
    InitializeConditionVariable(&g_pool.done);
#else
    n = (int)sysconf(_SC_NPROCESSORS_ONLN);
    pthread_mutex_init(&g_pool.lock, NULL);
    pthread_cond_init(&g_pool.go, NULL);
    pthread_cond_init(&g_pool.done, NULL);
#endif
    /* One core is the guest's; the rest draw, up to MAX_BANDS in all. */
    n = n > 2 ? n - 1 : 1;
    if (e) n = atoi(e);
    if (n < 1) n = 1;
    if (n > MAX_BANDS) n = MAX_BANDS;
    g_pool.bands = n;
    for (i = 1; i < n; i++) {
#ifdef _WIN32
        HANDLE h = CreateThread(NULL, 0, band_worker, (LPVOID)(intptr_t)i, 0, NULL);
        if (!h) { g_pool.bands = i; break; }
        CloseHandle(h);
#else
        pthread_t t;
        if (pthread_create(&t, NULL, band_worker, (void *)(intptr_t)i)) { g_pool.bands = i; break; }
        pthread_detach(t);
#endif
    }
}

/* fn(i) on every thread, the caller as thread 0, back when all are done. */
static void pool_run(void (*fn)(int band))
{
    static int started;
    if (!started) { started = 1; pool_start(); }
    g_pool.fn = fn;
    if (g_pool.bands <= 1) { fn(0); return; }
    LOCK(&g_pool.lock);
    g_pool.pending = (unsigned)g_pool.bands - 1u;
    g_pool.gen++;
    WAKE_ALL(&g_pool.go);
    UNLOCK(&g_pool.lock);

    fn(0);

    LOCK(&g_pool.lock);
    while (g_pool.pending) WAIT(&g_pool.done, &g_pool.lock);
    UNLOCK(&g_pool.lock);
}

static void raster_bands(const vpjob_t *j, uint32_t *fb, int fw, int fh)
{
    if (getenv("M3_REAL3D_STATS") || g_pick >= 0) {
        raster(j, fb, fw, 0, fh - 1);
        return;
    }
    g_pool.job = j; g_pool.fb = fb; g_pool.fw = fw; g_pool.fh = fh;
    g_pool.y0 = j->proj.y0; g_pool.y1 = j->proj.y1;
    pool_run(band_draw);
}

/* ---- scroll fog ----------------------------------------------------------
 *
 * Word 0x20's low byte (MAME: "Scroll Fog") is a haze over whatever is
 * already on screen behind a viewport: its rectangle is blended toward the
 * fog colour, dimmed by the ambient fog, by that byte over 255. The Lost
 * World's intro uses it -- "SOMETHING HAS SURVIVED" sets 0.29 of E0E0FF over
 * the storm backdrop -- and nothing else does. */
static void scroll_fog(uint32_t *fb, int fw, const vpjob_t *j, int ry0, int ry1)
{
    const uint32_t *w = j->fogw;
    float a = (float)(w[0] & 0xFFu) / 255.0f;     /* viewport words 0x20, 0x22, 0x25 */
    uint32_t col = w[1];
    float dim = (float)((w[2] >> 16) & 0xFFu) / 255.0f;
    float fc[3];
    int x, y, k;
    if (a <= 0.0f || g_no_fog) return;
    for (k = 0; k < 3; k++) fc[k] = dim * (float)((col >> (8 * k)) & 0xFFu);
    for (y = j->proj.y0 > ry0 ? j->proj.y0 : ry0; y <= j->proj.y1 && y <= ry1; y++)
        for (x = j->proj.x0; x <= j->proj.x1; x++) {
            uint32_t d = fb[y * fw + x], m = 0xFF000000u;
            for (k = 0; k < 3; k++) {
                float v = (float)((d >> (8 * k)) & 0xFFu);
                m |= (uint32_t)(v + (fc[k] - v) * a + 0.5f) << (8 * k);
            }
            fb[y * fw + x] = m;
        }
}

/* ---- drawing the field ------------------------------------------------
 *
 * Walking the scene reads guest memory, so it happens on the guest's thread
 * at the end of a field. Drawing it does not: each viewport the walk finds is
 * kept as a job (vpjob_t) and drawn either at once, or -- deferred -- later,
 * by real3d_compose() on the thread that composes the picture while the
 * guest runs the next field. Deferred, the walk goes into one of two sets and
 * the texture uploads are queued with it, so nothing the drawing threads
 * read is written under them; model3recomp_end_frame() does not hand the
 * guest a set back until its picture is finished. */

/* One viewport over rows ry0..ry1. A fresh depth buffer per viewport: they
 * are separate layers, not one scene. Sharing a buffer lets whichever
 * viewport is walked first decide the depth of the whole screen, and the
 * first one here holds a flat quad that covers it. */
static void job_rows(const vpjob_t *j, uint32_t *fb, int fw, int ry0, int ry1)
{
    int i, n = (ry1 - ry0 + 1) * fw;
    float *z = g_zbuf + (size_t)ry0 * fw;
    float *t0 = g_tz[0] + (size_t)ry0 * fw, *t1 = g_tz[1] + (size_t)ry0 * fw;
    /* The layers' depths start over too, but not their contents: what an
     * earlier viewport left in them is still to be composited, unless this
     * one draws over it. */
    for (i = 0; i < n; i++) z[i] = t0[i] = t1[i] = 1e30f;
    scroll_fog(fb, fw, j, ry0, ry1);
    raster(j, fb, fw, ry0, ry1);
}

/* Not deferred: the viewport just walked, now, on every core. */
static void draw_job(const vpjob_t *j, uint32_t *fb, int fw, int fh)
{
    int i, nn = fw * fh;
    for (i = 0; i < nn; i++) g_zbuf[i] = g_tz[0][i] = g_tz[1][i] = 1e30f;
    scroll_fog(fb, fw, j, 0, fh - 1);
    raster_bands(j, fb, fw, fh);
}

/* ---- the translucency layers --------------------------------------------
 *
 * The Real3D does not blend. A translucent polygon is drawn as a stipple, a
 * screen door whose density is its opacity, in one of two patterns that
 * word 6 bit 17 picks. Polygons with the same pattern cover the same pixels,
 * so of several behind one another only the nearest is seen: three 40% mist
 * sheets in a row are a 40% mist, where blending each over the last made
 * them 78%. Polygons with different patterns cover different pixels, and
 * where both are present there is nothing left of the scene behind.
 *
 * So each pattern is a layer that keeps the nearest see-through pixel at
 * each position, and the layers go onto the picture once every viewport of
 * the field is drawn: one layer alone blends by its opacity; both together
 * cover the pixel with a mix of the two, weighted toward the more opaque.
 * The stipple itself is averaged away, as it is on a monitor. Checked against
 * a reference renderer (Supermodel's New3D, the same model) on dumped scenes;
 * see the lostworld playtest log of 2026-10-25. Each composited pixel is
 * cleared for the next field. */
static void composite_rows(uint32_t *fb, int fw, int y0, int y1)
{
    size_t o = (size_t)y0 * (size_t)fw, end = (size_t)(y1 + 1) * (size_t)fw;
    for (; o < end; o++) {
        uint32_t c1 = g_tcol[0][o], c2 = g_tcol[1][o];
        uint32_t a1 = c1 >> 24, a2 = c2 >> 24, d = fb[o], m = 0;
        int sh;
        if (!(a1 | a2)) continue;
        g_tcol[0][o] = g_tcol[1][o] = 0;
        if (a1 && a2) {
            /* Layer 2's share, in 256ths: half, moved by how much more
             * opaque it is than layer 1. */
            uint32_t t = ((a2 + 255u - a1) * 256u + 255u) / 510u;
            for (sh = 0; sh < 24; sh += 8)
                m |= ((((c1 >> sh) & 0xFFu) * (256u - t)
                     + ((c2 >> sh) & 0xFFu) * t + 128u) >> 8) << sh;
        } else {
            uint32_t c = a1 ? c1 : c2, a = a1 ? a1 : a2;
            for (sh = 0; sh < 24; sh += 8)
                m |= ((((c >> sh) & 0xFFu) * a
                     + ((d >> sh) & 0xFFu) * (255u - a)) / 255u) << sh;
        }
        fb[o] = 0xFF000000u | m;
    }
}

/* Deferred texture uploads, in the order the guest made them: a copy of the
 * FIFO as the walk found it (kind 0), or a VROM load (kind 1: header, offset
 * into VROM). */
static void queue_tex(uint32_t kind, uint32_t a, uint32_t b, const uint8_t *p, size_t n)
{
    size_t need = 16u + ((n + 3u) & ~(size_t)3u);
    fieldset_t *q = &g_set[g_col];
    uint32_t hd[4];
    if (q->ntex + need > q->cap_tex) {
        size_t cap = q->cap_tex ? q->cap_tex * 2u : 1u << 20;
        uint8_t *m;
        while (cap < q->ntex + need) cap *= 2u;
        m = (uint8_t *)realloc(q->tex, cap);
        if (!m) return;
        q->tex = m; q->cap_tex = cap;
    }
    hd[0] = kind; hd[1] = a; hd[2] = b; hd[3] = (uint32_t)n;
    memcpy(q->tex + q->ntex, hd, 16);
    if (n) memcpy(q->tex + q->ntex + 16, p, n);
    q->ntex += need;
}

static void apply_tex(int k)
{
    size_t i = 0;
    while (i + 16u <= g_set[k].ntex) {
        uint32_t hd[4];
        memcpy(hd, g_set[k].tex + i, 16);
        if (hd[0] == 0) apply_fifo(g_set[k].tex + i + 16, hd[3]);
        else {
            size_t n = 0;
            const uint8_t *v = bus_vrom(&n);
            if (v && hd[2] < n) {
                g_src_be = 1;
                upload_texture(hd[1], v + hd[2], n - hd[2]);
                g_src_be = 0;
            }
        }
        i += 16u + ((hd[3] + 3u) & ~(size_t)3u);
    }
    g_set[k].ntex = 0;
}

void real3d_set_deferred(int on) { g_deferred = on; }

int real3d_flip(void)
{
    int k = g_col;
    g_col ^= 1;
    return k;
}

int real3d_set_index(void) { return g_col; }

/* The whole picture of set k, deferred, on every core: a thread takes its
 * slices of rows and draws them from the bottom up -- the tilemaps behind,
 * each viewport, the tilemaps in front -- so a field is one round of the
 * pool, not one per viewport plus the tilemaps on their own. */
static int g_ck;
static void compose_band(int i)
{
    int fw = g_pool.fw, fh = g_pool.fh, y;
    unsigned n;
    for (y = i * SLICE; y < fh; y += g_pool.bands * SLICE) {
        int y1 = y + SLICE - 1 < fh - 1 ? y + SLICE - 1 : fh - 1;
        memset(g_pool.fb + (size_t)y * fw, 0, sizeof(uint32_t) * (size_t)(y1 - y + 1) * fw);
        tilegen_render_rows(0, y, y1);
        for (n = 0; n < g_set[g_ck].njobs; n++)
            job_rows(&g_set[g_ck].jobs[n], g_pool.fb, fw, y, y1);
        composite_rows(g_pool.fb, fw, y, y1);
        tilegen_render_rows(1, y, y1);
    }
}

void real3d_compose(int k)
{
    int fw = 0, fh = 0;
    uint32_t *fb = platform_framebuffer(&fw, &fh);
    if (!fb || !g_zbuf || !g_tcol[0] || !g_tcol[1] || !g_tz[0] || !g_tz[1]
        || fw != g_zw || fh != g_zh) return;
    apply_tex(k);
    tilegen_set_source(k);
    g_ck = k;
    g_pool.fb = fb; g_pool.fw = fw; g_pool.fh = fh;
    pool_run(compose_band);
}

/* ---- entry point ------------------------------------------------------- */
void real3d_render(void)
{
    int fw = 0, fh = 0;
    uint32_t *fb = platform_framebuffer(&fw, &fh);
    uint32_t vpaddr;
    unsigned guard, total = 0, prio, first;
    static int inited;

    if (!fb) return;

    if (!inited) {
        const char *e = getenv("M3_VERTEX_SCALE");
        if (e && atof(e) > 0.0) g_vertex_scale = (float)atof(e);
        e = getenv("M3_TEX_SIZE");
        if (e) g_tex_size_mode = atoi(e);
        g_tex_all = getenv("M3_TEX_ALL") != NULL;
        g_textures = getenv("M3_NO_TEXTURES") == NULL;
        g_light_flip = getenv("M3_LIGHT_FLIP") != NULL;
        g_old_proj = getenv("M3_OLD_PROJ") != NULL;
        g_no_fog = getenv("M3_NO_FOG") != NULL;
        g_old_trans = getenv("M3_OLD_TRANSLUCENCY") != NULL;
        g_no_mips = getenv("M3_NO_MIPS") != NULL;
        g_no_spec = getenv("M3_NO_SPECULAR") != NULL;
        e = getenv("M3_AMBIENT");
        if (e) { g_ambient = (float)atof(e); g_ambient_forced = 1; }
        g_set[0].tris = (tri_t *)malloc(sizeof(tri_t) * MAX_TRIS);
        g_set[1].tris = (tri_t *)malloc(sizeof(tri_t) * MAX_TRIS);
        if (!g_sheet) g_sheet = (uint16_t *)calloc(TEX_W * TEX_H, sizeof(uint16_t));
        if (!g_sheet_set) g_sheet_set = (uint8_t *)calloc(TEX_W * TEX_H, 1);
        inited = 1;
    }
    if (!g_set[0].tris || !g_set[1].tris) return;
    g_tris = g_set[g_col].tris;
    g_stris = g_set[g_col].stris; g_cap_stris = g_set[g_col].cap_stris;

    flush_texture_fifo();

    /* No clear: the tilemap layers the guest put behind the 3D have
     * already been drawn into this buffer. */

    if (g_zw != fw || g_zh != fh) {
        int k;
        free(g_zbuf);
        g_zbuf = (float *)malloc(sizeof(float) * (size_t)fw * (size_t)fh);
        for (k = 0; k < 2; k++) {
            free(g_tcol[k]); free(g_tz[k]);
            g_tcol[k] = (uint32_t *)calloc((size_t)fw * (size_t)fh, sizeof(uint32_t));
            g_tz[k] = (float *)malloc(sizeof(float) * (size_t)fw * (size_t)fh);
        }
        g_zw = fw; g_zh = fh;
    }
    if (!g_zbuf || !g_tcol[0] || !g_tcol[1] || !g_tz[0] || !g_tz[1]) return;
    g_ntris = 0;
    g_nstris = 0;
    g_set[g_col].njobs = 0;

    /* M3_TEX_DUMP=<file>, M3_TEX_DUMP_AT=<field>: write the texture
     * sheet once. It used to fire after the hundredth upload, which
     * is during the boot, and made the sheet look as though only its
     * first 256 columns were ever written. */
    if (getenv("M3_TEX_DUMP")) {
        static int done, probed;
        static uint64_t at;
        if (!probed) { const char *e = getenv("M3_TEX_DUMP_AT");
                       probed = 1;
                       at = e ? strtoull(e, NULL, 0) : 0; }
        if (!done && g_sheet && model3recomp_frame_count() >= at) {
            FILE *fp = fopen(getenv("M3_TEX_DUMP"), "wb");
            done = 1;
            if (fp) { fwrite(g_sheet, 2, (size_t)TEX_W * TEX_H, fp);
                      fclose(fp); }
        }
    }
    g_drawn = 0;
    { static int probed; if (!probed) { probed = 1;
        { const char *e = getenv("M3_NODE_DUMP");
          m3_node_dump = e ? (int)strtol(e, NULL, 0) : 0;
          e = getenv("M3_NODE_FROM");
          m3_node_from = e ? strtoull(e, NULL, 0) : 0; } } }
    /* Viewports are drawn lowest priority first (word 0 bits 3..4), and a
     * viewport with bit 5 set is switched off. */
    for (prio = 0; prio < 4u; prio++) {
    vpaddr = 0x800000u;
    for (guard = 0; guard < 16u && vpaddr; guard++) {
        const uint8_t *vp = cull_at(vpaddr, 0x20);
        mat_t cam;
        float hf, vf, sx, sy, light[3], intensity;
        uint32_t next;

        if (!vp) break;
        if (((rd32le(vp) >> 3) & 3u) != prio || (rd32le(vp) & 0x20u)) {
            next = rd32le(vp + 0x01 * 4) & 0x00FFFFFFu;
            if (next == vpaddr) break;
            vpaddr = next;
            continue;
        }
        g_matbase = rd32le(vp + 0x16 * 4) & 0x00FFFFFFu;

        /* Words 0x0C and 0x0E are the sines of the horizontal and vertical
         * half angles. For this game they are 19.38 and 15 degrees, whose
         * tangents are in the ratio 1.3125 -- the aspect of a 496x384
         * screen, which is how the pair was identified. */
        hf = f32_of(rd32le(vp + 0x0C * 4));
        vf = f32_of(rd32le(vp + 0x0E * 4));
        if (hf < -1.0f) hf = -1.0f;
        if (hf > 1.0f) hf = 1.0f;
        if (vf < -1.0f) vf = -1.0f;
        if (vf > 1.0f) vf = 1.0f;
        hf = asinf(hf); vf = asinf(vf);
        if (hf < 1e-3f || vf < 1e-3f) { hf = 0.338f; vf = 0.262f; }
        sx = (fw * 0.5f) / tanf(hf);
        sy = (fh * 0.5f) / tanf(vf);
        g_proj.ax = sx; g_proj.bx = fw * 0.5;
        g_proj.ay = -sy; g_proj.by = fh * 0.5;
        g_proj.x0 = 0; g_proj.y0 = 0; g_proj.x1 = fw - 1; g_proj.y1 = fh - 1;

        /* The frustum, from words 8..0xB (MAME's "Cv", "Cw", "Io", "Jo").
         * Measured against the clip planes in 0x0C..0x13, whose normals
         * are (sin, cos) of each edge's angle: in a round 1/Cv is 0.70333
         * and the left and right planes are 19.38 degrees out, 2 tan of
         * which is 0.70332; 1/Cw is 0.5359 and the top and bottom are 15
         * degrees out, 2 tan 15 = 0.5359. So Cv and Cw are one over the
         * frustum's width and height as tangents, and Jo and Io (0.5 in
         * every scene) where its centre falls across each, from the left
         * and the top. The planes alone will not do: the HUD viewport's
         * bottom plane does not hold a unit normal.
         *
         * Words 0x1A and 0x14 are the viewport's rectangle on the screen,
         * X/Y in 12.4 and width/height in 14.2 (MAME's documentation). A
         * HUD is a viewport of its own; projected as though it were the
         * whole screen, its counters land somewhere they cannot be seen. */
        {
            float cvv = f32_of(rd32le(vp + 0x08 * 4)), cww = f32_of(rd32le(vp + 0x09 * 4));
            float io = f32_of(rd32le(vp + 0x0A * 4)), jo = f32_of(rd32le(vp + 0x0B * 4));
            uint32_t w14 = rd32le(vp + 0x14 * 4), w1a = rd32le(vp + 0x1A * 4);
            double vx = (w1a & 0xFFFFu) / 16.0, vy = (w1a >> 16) / 16.0;
            double vw = (w14 & 0xFFFFu) / 4.0, vh = (w14 >> 16) / 4.0;
            if (cvv > 1e-6f && cww > 1e-6f && vw >= 1.0 && vh >= 1.0 && !g_old_proj) {
                /* x/z = jo/cv at the left edge's distance from the centre,
                 * so pixel = vx + vw * (jo + cv * x/z), and likewise down
                 * from the top for y, which points up. */
                g_proj.ax = vw * cvv;
                g_proj.bx = vx + vw * jo;
                g_proj.ay = -vh * cww;
                g_proj.by = vy + vh * io;
                g_proj.x0 = (int)vx; g_proj.y0 = (int)vy;
                g_proj.x1 = (int)(vx + vw) - 1; g_proj.y1 = (int)(vy + vh) - 1;
                if (g_proj.x0 < 0) g_proj.x0 = 0;
                if (g_proj.y0 < 0) g_proj.y0 = 0;
                if (g_proj.x1 > fw - 1) g_proj.x1 = fw - 1;
                if (g_proj.y1 > fh - 1) g_proj.y1 = fh - 1;
            }
            if (getenv("M3_FOG_TRACE"))
                fprintf(stderr, "[fog] field %llu vp %u prio %u col %06X dens %g start %g att %g amb %g scroll %g scrollatt %g\n",
                        (unsigned long long)model3recomp_frame_count(), guard, (rd32le(vp) >> 3) & 3u,
                        rd32le(vp + 0x22 * 4) & 0xFFFFFFu, (double)f32_of(rd32le(vp + 0x23 * 4)),
                        (double)(int16_t)(rd32le(vp + 0x25 * 4) & 0xFFFF) / 255.0,
                        (double)((rd32le(vp + 0x24 * 4) >> 16) & 0xFF) / 255.0,
                        (double)((rd32le(vp + 0x25 * 4) >> 16) & 0xFF) / 255.0,
                        (double)(rd32le(vp + 0x20 * 4) & 0xFF) / 255.0,
                        (double)(rd32le(vp + 0x24 * 4) & 0xFF) / 255.0);
            if (getenv("M3_VP_TRACE"))
                fprintf(stderr, "[vp] %u w0 %08X prio %u%s cv %g cw %g io %g jo %g rect %g,%g %gx%g\n",
                        guard, rd32le(vp), (rd32le(vp) >> 3) & 3u,
                        (rd32le(vp) & 0x20u) ? " disabled" : "",
                        cvv, cww, io, jo, vx, vy, vw, vh);
        }

        /* Words 0x04..0x06 are the sun vector in that order. MAME takes
         * it as (w5, w6, w4), and both readings are unit vectors in
         * every scene dumped, so only a lit picture tells them apart: on
         * the jungle floor this order lights the tree trunks and the
         * signpost, and the other leaves them flat black. */
        light[0] = f32_of(rd32le(vp + 0x04 * 4));
        light[1] = f32_of(rd32le(vp + 0x05 * 4));
        light[2] = f32_of(rd32le(vp + 0x06 * 4));
        if (g_light_flip) { light[0] = -light[0];
            light[1] = -light[1]; light[2] = -light[2]; }
        g_vp07 = f32_of(rd32le(vp + 0x07 * 4));

        /* Ambient comes from the scene, not from a constant. Word 0x24
         * bits 8..15 over 255: this game writes 0x5A, so 0.353, where
         * the invented constant here was 0.20 and left the jungle floor
         * black. M3_AMBIENT still overrides, for comparing. */
        if (!g_ambient_forced) {
            unsigned a24 = (rd32le(vp + 0x24 * 4) >> 8) & 0xFFu;
            if (a24) g_ambient = (float)a24 / 255.0f;
        }
        {
            uint32_t col = rd32le(vp + 0x22 * 4), w25 = rd32le(vp + 0x25 * 4);
            float dim = (float)((w25 >> 16) & 0xFFu) / 255.0f;
            float d = fabsf(f32_of(rd32le(vp + 0x23 * 4)));
            g_fog.density = (d == d && d < 1e30f) ? d : 0.0f;        /* NaN, inf */
            g_fog.start = (float)(int16_t)(w25 & 0xFFFFu) / 255.0f;
            g_fog.r = dim * (float)((col >> 16) & 0xFFu);
            g_fog.g = dim * (float)((col >> 8) & 0xFFu);
            g_fog.b = dim * (float)(col & 0xFFu);
            g_fog.on = !g_no_fog && (g_fog.density > 0.0f || g_fog.start > 0.0f);
        }
        if (light[0] == 0.0f && light[1] == 0.0f && light[2] == 0.0f) {
            light[0] = 0.3f; light[1] = 0.5f; light[2] = -0.81f;
        }


        /* No camera matrix. Matrix 0 looked like one -- a plain rotation,
         * sitting where a view transform belongs -- and using it turned the
         * one scene available at the time from nothing into something, which
         * was taken for confirmation. On a frame that actually has a scene
         * in it, applying matrix 0 pushes the whole thing off the left of
         * the screen, and leaving it out gives rocks and a cliff face. The
         * node matrices already carry the geometry into view space.
         *
         * M3_CAM0 puts it back for comparison: 1 as stored, 2 transposed.
         * The scene a round builds is almost entirely off screen, which
         * is what a missing axis permutation at the root of the matrix
         * stack looks like. */
        cam = IDENT;
        {
            static long which = -2;
            if (which == -2) { const char *e = getenv("M3_CAM0");
                               which = e ? strtol(e, NULL, 0) : 0; }
            if (which == 1) mat_load(g_matbase, 0, &cam);
            else if (which == 2) {
                mat_t t; int r, c;
                if (mat_load(g_matbase, 0, &t)) {
                    for (r = 0; r < 3; r++)
                        for (c = 0; c < 3; c++) cam.m[r][c] = t.m[c][r];
                    for (r = 0; r < 3; r++) cam.m[r][3] = t.m[r][3];
                }
            }
        }

        intensity = f32_of(rd32le(vp + 0x07 * 4));
        if (!(intensity >= 0.0f)) intensity = 1.0f;   /* NaN guard */
        if (intensity > 2.0f) intensity = 2.0f;

        first = g_ntris;
        {
            /* M3_VP=n draws only viewport n. Three of them share the
             * screen in a round and each gets its own depth buffer, so
             * whichever is walked last wins wherever it draws -- and
             * the only way to tell which one is painting over the
             * scene is to draw them one at a time. */
            static long only = -2;
            unsigned before = g_drawn;
            if (only == -2) { const char *e = getenv("M3_VP");
                              only = e ? strtol(e, NULL, 0) : -1; }
            if (only < 0 || (long)guard == only) {
                unsigned *nj = &g_set[g_col].njobs;
                vpjob_t *j = &g_set[g_col].jobs[*nj < 63u ? (*nj)++ : 63u];
                descend(rd32le(vp + 0x02 * 4) & 0x00FFFFFFu, &cam, 0);
                { const char *e = getenv("M3_PICK");
                  g_pick = -1; g_pick_tri = -1;
                  if (e) { int px = atoi(e), py = 0; const char *c = strchr(e, ',');
                           if (c) py = atoi(c + 1);
                           g_pick = (long)py * fw + px; } }
                j->first = first; j->count = g_ntris - first;
                g_rtris = g_tris + first; g_rn = j->count;
                j->s0 = g_nstris;
                setup_tris(fw, fh, light, intensity);
                j->s1 = g_nstris;
                j->proj = g_proj; j->fog = g_fog; j->ambient = g_ambient;
                j->sx = sx; j->sy = sy; j->intensity = intensity;
                memcpy(j->light, light, sizeof j->light);
                j->fogw[0] = rd32le(vp + 0x20 * 4);
                j->fogw[1] = rd32le(vp + 0x22 * 4);
                j->fogw[2] = rd32le(vp + 0x25 * 4);
                j->st = g_stris;
                if (!g_deferred) draw_job(j, fb, fw, fh);
                if (g_pick_tri >= 0) {
                    const tri_t *p = &g_tris[first + g_pick_tri];
                    fprintf(stderr, "[pick] vp%u tri %ld tex %d at %u,%u page %u"
                            " %ux%u fmt %u alpha %u test %u texalpha %u lum %u fog %u z %.0f"
                            " rgb %02X%02X%02X uv %.1f,%.1f %.1f,%.1f %.1f,%.1f\n",
                            guard, g_pick_tri, p->textured, p->tx, p->ty,
                            p->page, p->tw, p->th, p->fmt, p->alpha,
                            p->alpha_test, p->tex_alpha, p->luminous, p->fog, (double)p->v[0][2],
                            p->r, p->g, p->b, p->uv[0][0], p->uv[0][1],
                            p->uv[1][0], p->uv[1][1], p->uv[2][0], p->uv[2][1]);
                }
            }
            if (getenv("M3_REAL3D_STATS"))
                fprintf(stderr, "[real3d]   vp%u @%06X rect %08X/%08X"
                                "  %u tris  %u px  back %u near %u"
                                "  degen %u off %u\n",
                        guard, vpaddr & 0x00FFFFFFu,
                        rd32le(vp + 0x14 * 4), rd32le(vp + 0x1A * 4),
                        g_ntris - first, g_drawn - before,
                        g_cull_back, g_cull_near, g_cull_degen, g_cull_off);
            if (getenv("M3_REAL3D_STATS"))
                fprintf(stderr, "[real3d]     textured %u tris / %u px,"
                                "  flat %u tris / %u px\n",
                        g_tex_tris, g_tex_px, g_flat_tris, g_flat_px);
            if (getenv("M3_REAL3D_STATS"))
                fprintf(stderr, "[real3d]     texels: %u from the sheet,"
                                "  %u transparent,  %u never uploaded\n",
                        g_texel_hit, g_texel_clear, g_texel_missing);
            if (getenv("M3_REAL3D_STATS")) {
                unsigned k;
                for (k = 0; k < MISS_SLOTS; k++)
                    if (g_miss[k].n)
                        fprintf(stderr, "[real3d]       miss at %4u,%4u"
                                        " page %u  %ux%u  x%u\n",
                                g_miss[k].x, g_miss[k].y, g_miss[k].page,
                                g_miss[k].w, g_miss[k].h, g_miss[k].n);
                memset(g_miss, 0, sizeof g_miss);
            }
            g_texel_hit = g_texel_clear = g_texel_missing = 0;
            g_tex_tris = g_flat_tris = g_tex_px = g_flat_px = 0;
            if (getenv("M3_REAL3D_STATS"))
                fprintf(stderr, "[real3d]     matrices: %u loads, max index"
                                " %u, %u all-zero, biggest |t| %.0f at %u\n",
                        g_mat_loads, g_mat_max, g_mat_zero, g_mat_bigt,
                        g_mat_bigi);
            if (getenv("M3_REAL3D_STATS"))
                fprintf(stderr, "[real3d]     nodes: %u by translation,"
                                "  %u by matrix,  %u with neither\n",
                        g_node_translate, g_node_matrix, g_node_neither);
            g_node_translate = g_node_matrix = g_node_neither = 0;
            g_mat_loads = g_mat_max = g_mat_zero = g_mat_bigi = 0;
            g_mat_bigt = 0.0f;
        }
        total = g_ntris;

        next = rd32le(vp + 0x01 * 4) & 0x00FFFFFFu;
        if (next == vpaddr) break;
        vpaddr = next;
    }
    }
    /* Not deferred, every viewport is drawn by now: the translucency
     * layers go on, before the tilemaps in front of the 3D. */
    if (!g_deferred) composite_rows(fb, fw, 0, fh - 1);
    g_tri_count = total;
    /* setup_tris() may have moved the list as it grew */
    g_set[g_col].stris = g_stris; g_set[g_col].cap_stris = g_cap_stris;
    { unsigned n; for (n = 0; n < g_set[g_col].njobs; n++) g_set[g_col].jobs[n].st = g_stris; }

    /* M3_REAL3D_STATS: how much geometry the game handed us, and how much of
     * it landed. Printed on change rather than per field, because it changes
     * a handful of times in an attract cycle and printing it every field
     * buries that. This is how you find a frame that has 3D in it at all. */
    /* A cheap fingerprint of what the guest handed us: the culling tree and
     * the matrix table. If the picture stops changing, this says whether the
     * guest stopped writing a new scene or kept writing the same one. */
    if (getenv("M3_REAL3D_STATS")) {
        size_t cn = 0;
        const uint8_t *cp = bus_cull_hi(&cn);
        uint32_t h = 2166136261u;
        if (cp) {
            size_t k, lim = cn < 0x20000u ? cn : 0x20000u;
            for (k = 0; k < lim; k += 4)
                h = (h ^ rd32le(cp + k)) * 16777619u;
        }
        g_scene_hash = h;
    }
    /* And a fingerprint of what came out, so a scene that changes while the
     * picture does not shows up as exactly that. */
    if (getenv("M3_REAL3D_STATS")) {
        uint32_t h = 2166136261u;
        int k, nn = fw * fh;
        for (k = 0; k < nn; k += 7)
            h = (h ^ fb[k]) * 16777619u;
        g_fb_hash = h;
    }
    if (getenv("M3_REAL3D_STATS")) {
        static unsigned last_total = 0xFFFFFFFFu;
        static unsigned last_drawn = 0xFFFFFFFFu;
        static uint32_t last_hash = 0xFFFFFFFFu;
        if (total != last_total || g_drawn != last_drawn ||
            g_scene_hash != last_hash) {
            last_total = total; last_drawn = g_drawn;
            last_hash = g_scene_hash;
            { size_t _fu = 0; bus_texfifo(&_fu, NULL);
            fprintf(stderr, "[real3d] field %llu trig %llu fifo %llu scene %08X fb %08X vp07 %.4f: %u triangles, %u drawn\n",
                    (unsigned long long)model3recomp_frame_count(),
                    (unsigned long long)bus_r3d_trigger_count(),
                    (unsigned long long)_fu, g_scene_hash, g_fb_hash,
                    g_vp07, total, g_drawn); }
            fflush(stderr);
        }
    }
}

/* The texture sheet is built from uploads the guest made long ago, so it is
 * state too: restoring RAM without it leaves every surface wrong. */
void real3d_state(m3_state_t *st)
{
    if (!g_sheet) g_sheet = (uint16_t *)calloc(TEX_W * TEX_H, sizeof(uint16_t));
    if (!g_sheet_set) g_sheet_set = (uint8_t *)calloc(TEX_W * TEX_H, 1);
    if (!g_sheet || !g_sheet_set) return;
    m3_state_io(st, g_sheet, (size_t)TEX_W * TEX_H * sizeof(uint16_t));
    m3_state_io(st, g_sheet_set, (size_t)TEX_W * TEX_H);
    M3_STATE_VAR(st, g_fifo_seen);
    M3_STATE_VAR(st, g_tex_uploads);
}
