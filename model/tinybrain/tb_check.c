/* tb_check.c - run common/llm/tiny_brain.c over state lines (stdin) and print
 * each line's goal + urgency logits, for tb_check.py to compare against the
 * numpy forward pass. Also times the forward pass. Build from the repo root:
 *   cc -O2 -std=c11 -Icommon/llm -o model/tinybrain/tb_check \
 *     model/tinybrain/tb_check.c common/llm/tiny_brain.c -lm */
#define _POSIX_C_SOURCE 199309L
#include "tiny_brain.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n); if (b && fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    fclose(f); return b;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: tb_check tinybrain.bin < states\n"); return 1; }
    size_t n; uint8_t *b = slurp(argv[1], &n); tb_model_t m;
    if (!b || !tb_open(&m, b, n)) { fprintf(stderr, "bad tinybrain file\n"); return 1; }
    if (tb_open(&m, b, n - 1)) { fprintf(stderr, "a truncated file was accepted\n"); return 1; }
    tb_open(&m, b, n);
    char line[512]; float x[TB_N_FEAT], g[TB_N_GOAL], u[TB_N_URG];
    long count = 0; double secs = 0;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (!tb_features(line, x)) { printf("BAD\n"); continue; }
        struct timespec a, c; clock_gettime(CLOCK_MONOTONIC, &a);
        for (int r = 0; r < 100; r++) tb_forward(&m, x, g, u);
        clock_gettime(CLOCK_MONOTONIC, &c);
        secs += (c.tv_sec - a.tv_sec) + (c.tv_nsec - a.tv_nsec) / 1e9; count += 100;
        for (int i = 0; i < TB_N_GOAL; i++) printf("%.6f ", g[i]);
        for (int i = 0; i < TB_N_URG; i++) printf("%.6f%c", u[i], i + 1 < TB_N_URG ? ' ' : '\n');
    }
    fprintf(stderr, "tb_check: %.2f us per decision (features excluded), h %d-%d\n", count ? secs / count * 1e6 : 0, m.h1, m.h2);
    return 0;
}
