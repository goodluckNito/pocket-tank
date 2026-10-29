/* tiny_brain.c — see tiny_brain.h. */
#include "tiny_brain.h"
#include <math.h>
#include <string.h>

static const char *const TB_GOALS[TB_N_GOAL] = {
    "seek_food", "flee_shadow", "visit_bubbles", "follow_friend",
    "explore", "rest", "dart_play", "inspect_reef",
};
static const char *const TB_STAGES[4] = { "fry", "juv", "adult", "elder" };
static const char *const TB_DRIVES[8] = { "hunger", "energy", "stress", "curiosity", "bold", "social", "trust", "bored" };
static const char *const TB_SIGHTS[4] = { "food", "friend", "bubble", "reef" };

static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static float    rdf(const uint8_t *p, int i) { float v; memcpy(&v, p + 4 * i, 4); return v; }

bool tb_is_tinybrain(const uint8_t *bin, size_t len) {
    return bin && len >= 28 && memcmp(bin, "TBRN", 4) == 0;
}

static const uint8_t *take_layer(tb_layer_t *l, int n_in, int n_out, const uint8_t *p, const uint8_t *end) {
    size_t need = (size_t)n_in * n_out + 8u * (size_t)n_out;
    if (!p || (size_t)(end - p) < need) return NULL;
    l->n_in = n_in; l->n_out = n_out;
    l->w = (const int8_t *)p; p += (size_t)n_in * n_out;
    l->scale = p; p += 4u * n_out;
    l->bias = p;  p += 4u * n_out;
    return p;
}

bool tb_open(tb_model_t *m, const uint8_t *bin, size_t len) {
    if (!tb_is_tinybrain(bin, len) || rd32(bin + 4) != 1) return false;
    uint32_t n_in = rd32(bin + 8), h1 = rd32(bin + 12), h2 = rd32(bin + 16), ng = rd32(bin + 20), nu = rd32(bin + 24);
    if (n_in != TB_N_FEAT || ng != TB_N_GOAL || nu != TB_N_URG || !h1 || !h2 || h1 > TB_MAX_H || h2 > TB_MAX_H) return false;
    const uint8_t *p = bin + 28, *end = bin + len;
    p = take_layer(&m->l1, (int)n_in, (int)h1, p, end);
    p = take_layer(&m->l2, (int)h1, (int)h2, p, end);
    p = take_layer(&m->goal, (int)h2, TB_N_GOAL, p, end);
    p = take_layer(&m->urg, (int)h2, TB_N_URG, p, end);
    if (!p) return false;                         /* bytes past the last layer are fine: the device passes the whole partition */
    m->h1 = (int)h1; m->h2 = (int)h2;
    return true;
}

/* ---- features: must match model/tinybrain/features.py exactly ---- */
static int index_of(const char *w, const char *const *list, int n) {
    for (int i = 0; i < n; i++) if (strcmp(w, list[i]) == 0) return i;
    return -1;
}
static int digit(const char *w) { return (w[0] >= '0' && w[0] <= '9' && w[1] == 0) ? w[0] - '0' : -1; }
static int clock_of(const char *w) {
    int c = 0;
    for (; *w; w++) { if (*w < '0' || *w > '9') return -1; c = c * 10 + (*w - '0'); }
    return c >= 1 && c <= 12 ? c : -1;
}
static void put_clock(float *x, int c) {
    float a = (float)c / 12.0f * 6.28318530718f;
    x[0] = sinf(a); x[1] = cosf(a);
}

bool tb_features(const char *state, float *x) {
    char buf[400], *w[48]; int nw = 0;
    size_t n = strlen(state); if (n >= sizeof buf) return false;
    memcpy(buf, state, n + 1);
    for (char *p = buf; *p && nw < 48; ) {
        while (*p == ' ') *p++ = 0;
        if (!*p) break;
        w[nw++] = p;
        while (*p && *p != ' ') p++;
    }
    for (int i = 0; i < TB_N_FEAT; i++) x[i] = 0;
    int seen = 0;
    for (int i = 0; i + 1 < nw; ) {
        const char *k = w[i], *v = w[i + 1];
        int s = index_of(k, TB_SIGHTS, 4), d;
        if (s >= 0 || strcmp(k, "wall") == 0) {
            bool absent = strcmp(v, "none") == 0 || strcmp(v, "clear") == 0;
            if (absent) { if (s < 0) x[42] = 1; i += 2; }
            else {
                if (i + 2 >= nw) return false;
                int dist = strcmp(v, "near") == 0 ? 0 : strcmp(v, "mid") == 0 ? 1 : strcmp(v, "far") == 0 ? 2 : -1;
                int c = clock_of(w[i + 2]);
                if (dist < 0 || c < 0) return false;
                if (s >= 0) { float *o = x + 18 + 6 * s; o[0] = 1; o[1 + dist] = 1; put_clock(o + 4, c); }
                else { if (dist > 1) return false; x[43 + dist] = 1; put_clock(x + 45, c); }
                i += 3;
            }
            seen++;
            continue;
        }
        if (strcmp(k, "zone") == 0) { d = digit(v); if (d < 1 || d > 6) return false; x[d - 1] = 1; }
        else if ((d = index_of(k, TB_DRIVES, 8)) >= 0) { int q = digit(v); if (q < 0) return false; x[6 + d] = (float)q / 9.0f; }
        else if (strcmp(k, "stage") == 0) { d = index_of(v, TB_STAGES, 4); if (d < 0) return false; x[14 + d] = 1; }
        else if (strcmp(k, "last") == 0) { d = index_of(v, TB_GOALS, TB_N_GOAL); if (d < 0) return false; x[47 + d] = 1; }
        else if (strcmp(k, "time") == 0) x[55] = strcmp(v, "night") == 0 ? 1.0f : 0.0f;
        else return false;                        /* not a v4 key (a v2/v3 line has fish / shadow) */
        seen++;
        i += 2;
    }
    return seen == 1 + 8 + 1 + 4 + 1 + 1 + 1;     /* zone, drives, stage, sightings, wall, last, time */
}

static void dense(const tb_layer_t *l, const float *in, float *out, bool relu) {
    for (int o = 0; o < l->n_out; o++) {
        const int8_t *row = l->w + (size_t)o * l->n_in;
        float acc = 0;
        for (int i = 0; i < l->n_in; i++) acc += (float)row[i] * in[i];
        float v = acc * rdf(l->scale, o) + rdf(l->bias, o);
        out[o] = relu && v < 0 ? 0 : v;
    }
}

void tb_forward(const tb_model_t *m, const float *x, float *goal_logits, float *urg_logits) {
    float a1[TB_MAX_H], a2[TB_MAX_H];
    dense(&m->l1, x, a1, true);
    dense(&m->l2, a1, a2, true);
    dense(&m->goal, a2, goal_logits, false);
    dense(&m->urg, a2, urg_logits, false);
}
