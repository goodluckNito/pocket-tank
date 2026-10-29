/* tiny_brain.h — the small distilled decision network (docs/tinybrain.md).
 * A ~9K-parameter MLP that reads the same schema-v4 state line as the 14M
 * transformer and returns the same thing: a goal distribution (the 8 goal
 * tokens, GOAL_NAMES order) and an urgency distribution (0..9), in
 * microseconds instead of seconds. model/tinybrain/train.py writes the file;
 * advisor_core picks this engine when the model blob starts with "TBRN".
 *
 * The weights are read in place (the blob can sit in mapped flash). */
#ifndef TINY_BRAIN_H
#define TINY_BRAIN_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TB_N_FEAT 56
#define TB_N_GOAL 8
#define TB_N_URG  10
#define TB_MAX_H  256

typedef struct {
    int n_in, n_out;
    const int8_t *w;          /* [n_out][n_in] */
    const uint8_t *scale;     /* f32 [n_out], read with memcpy (the blob may be unaligned) */
    const uint8_t *bias;      /* f32 [n_out] */
} tb_layer_t;

typedef struct {
    int h1, h2;
    tb_layer_t l1, l2, goal, urg;
} tb_model_t;

/* true if bin looks like a tinybrain file (magic only) */
bool tb_is_tinybrain(const uint8_t *bin, size_t len);
/* parse the header and point the layers into bin; false on a bad or short
 * file (len may run past the weights, e.g. a whole flash partition) */
bool tb_open(tb_model_t *m, const uint8_t *bin, size_t len);
/* the schema-v4 state line as TB_N_FEAT floats (model/tinybrain/features.py's
 * twin); false if the line is not a v4 line */
bool tb_features(const char *state, float *x);
/* raw logits for both heads */
void tb_forward(const tb_model_t *m, const float *x, float *goal_logits, float *urg_logits);
#endif
