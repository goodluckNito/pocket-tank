/* label_q4.c - the shipped 4-bit model as a LABELER (tinybrain phase 1).
 *
 * Reads schema state lines (no trailing " ->") on stdin, one per line, and
 * writes one JSON object per line on stdout:
 *   {"state": "...", "p": [8 goal probs, GOAL_NAMES order], "u": [10 urgency probs]}
 * p is the softmax over the goal tokens at T = 1 - exactly the distribution
 * advisor_core samples from on the device. u is the urgency-digit
 * distribution after the GREEDY goal (the model's " urgency" token is fed
 * explicitly, so u is defined even when the model would have stopped).
 *
 * Same engine and tokenizer the sim and firmware run (common/llm). Build from
 * the repo root:
 *   cc -O3 -std=c11 -Icommon/llm -o model/tinybrain/label_q4 \
 *     model/tinybrain/label_q4.c common/llm/q4_model.c common/llm/word_tok.c -lm
 * Run (one process per core; label_states.py does the fan-out):
 *   ./model/tinybrain/label_q4 model/out/model_q4.bin model/out/tokenizer.bin < states.txt */
#include "q4_model.h"
#include "word_tok.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *const GOALS[8] = {
    "seek_food", "flee_shadow", "visit_bubbles", "follow_friend",
    "explore", "rest", "dart_play", "inspect_reef",
};

static uint8_t *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n); if (b && fread(b, 1, *n, f) != *n) { free(b); b = NULL; }
    fclose(f); return b;
}
static void *xalloc(size_t n) { return calloc(1, n); }

static int find_piece(const word_tok_t *t, const char *word) {
    for (int id = 0; id < t->vocab_size; id++) {
        const char *p = word_tok_piece(t, id);
        if (p[0] == ' ' && strcmp(p + 1, word) == 0) return id;
    }
    return -1;
}

static void softmax_over(const float *logits, const int *ids, int n, float *out) {
    float mx = -1e30f, sum = 0;
    for (int i = 0; i < n; i++) if (logits[ids[i]] > mx) mx = logits[ids[i]];
    for (int i = 0; i < n; i++) { out[i] = expf(logits[ids[i]] - mx); sum += out[i]; }
    for (int i = 0; i < n; i++) out[i] /= sum;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: label_q4 model_q4.bin tokenizer.bin < states\n"); return 1; }
    size_t ml, tl; uint8_t *mb = slurp(argv[1], &ml), *tb = slurp(argv[2], &tl);
    if (!mb || !tb) { fprintf(stderr, "could not read model/tokenizer\n"); return 1; }
    q4_model_t *m = q4_model_open(mb, ml, xalloc);
    if (!m) { fprintf(stderr, "bad model\n"); return 1; }
    word_tok_t tok;
    if (word_tok_init(&tok, tb, tl, q4_model_config(m)->vocab_size) != 0) { fprintf(stderr, "bad tokenizer\n"); return 1; }

    int goal_ids[8], digit_ids[10], urg_id = find_piece(&tok, "urgency");
    for (int g = 0; g < 8; g++) if ((goal_ids[g] = find_piece(&tok, GOALS[g])) < 0) { fprintf(stderr, "no token for %s\n", GOALS[g]); return 1; }
    for (int d = 0; d < 10; d++) { char w[2] = { (char)('0' + d), 0 }; if ((digit_ids[d] = find_piece(&tok, w)) < 0) { fprintf(stderr, "no digit %d\n", d); return 1; } }
    if (urg_id < 0) { fprintf(stderr, "no urgency token\n"); return 1; }

    char line[512], prompt[520];
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        snprintf(prompt, sizeof prompt, "%s ->", line);
        int toks[64], n = word_tok_encode(&tok, prompt, toks, 60);
        float *logits = q4_model_prefill(m, toks, n);
        if (!logits) { fprintf(stderr, "prefill failed: %s\n", line); return 1; }
        float p[8], u[10];
        softmax_over(logits, goal_ids, 8, p);
        int best = 0; for (int g = 1; g < 8; g++) if (p[g] > p[best]) best = g;
        q4_model_forward(m, goal_ids[best], n);
        logits = q4_model_forward(m, urg_id, n + 1);
        softmax_over(logits, digit_ids, 10, u);
        printf("{\"state\": \"%s\", \"p\": [", line);
        for (int g = 0; g < 8; g++) printf(g ? ", %.5f" : "%.5f", p[g]);
        printf("], \"u\": [");
        for (int d = 0; d < 10; d++) printf(d ? ", %.5f" : "%.5f", u[d]);
        printf("]}\n");
    }
    return 0;
}
