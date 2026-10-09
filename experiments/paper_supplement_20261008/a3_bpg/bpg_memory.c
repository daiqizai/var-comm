/* In-memory entry points; the hash-bound upstream source remains unmodified. */
#define _GNU_SOURCE 1
#define main original_bpgenc_cli_main
#include "bpgenc.c"
#undef main
#include <limits.h>

typedef struct {
    uint8_t *data;
    size_t length;
    int failed;
} MemoryOutput;

static int memory_write(void *opaque, const uint8_t *buf, int count)
{
    MemoryOutput *m = opaque;
    uint8_t *p;
    if (count < 0 || m->length > SIZE_MAX - (size_t)count) {
        m->failed = 1;
        return 0;
    }
    p = realloc(m->data, m->length + count);
    if (!p) {
        m->failed = 1;
        return 0;
    }
    m->data = p;
    memcpy(p + m->length, buf, count);
    m->length += count;
    return count;
}

int varcomm_bpg_encode_png(const uint8_t *png, size_t length, int qp,
                          uint8_t **output, size_t *output_length)
{
    FILE *f;
    Image *img;
    BPGMetaData *md = NULL;
    BPGEncoderParameters *p;
    BPGEncoderContext *encoder;
    MemoryOutput result = {NULL, 0, 0};
    int ret;
    if (!png || !output || !output_length || !length || qp < 0 || qp > 51)
        return -1;
    *output = NULL;
    *output_length = 0;
    f = fmemopen((void *)png, length, "rb");
    if (!f) return -2;
    img = read_png(&md, f, BPG_CS_YCbCr, 8, 0, 0);
    fclose(f);
    if (!img) return -3;
    /* The frozen CLI uses no -keepmetadata. */
    bpg_md_free(md);
    p = bpg_encoder_param_alloc();
    if (!p) { image_free(img); return -4; }
    p->qp = qp;
    p->compress_level = 8;
    p->preferred_chroma_format = BPG_FORMAT_420;
    p->encoder_type = HEVC_ENCODER_X265;
    encoder = bpg_encoder_open(p);
    if (!encoder) { image_free(img); bpg_encoder_param_free(p); return -5; }
    bpg_encoder_set_extension_data(encoder, NULL);
    ret = bpg_encoder_encode(encoder, img, memory_write, &result);
    image_free(img);
    bpg_encoder_close(encoder);
    bpg_encoder_param_free(p);
    if (ret < 0 || result.failed || result.length == 0) {
        free(result.data);
        return -6;
    }
    *output = result.data;
    *output_length = result.length;
    return 0;
}

/* Positive return means actual malformed/unsupported received source data. */
int varcomm_bpg_decode_rgb(const uint8_t *bytes, size_t length,
                          uint8_t **output, int *width, int *height)
{
    BPGDecoderContext *decoder;
    BPGImageInfo info;
    uint8_t *pixels;
    int y;
    if (!bytes || !length || length > INT_MAX || !output || !width || !height)
        return 1;
    *output = NULL;
    *width = *height = 0;
    decoder = bpg_decoder_open();
    if (!decoder) return -1;
    if (bpg_decoder_decode(decoder, bytes, (int)length) < 0 ||
        bpg_decoder_get_info(decoder, &info) < 0) {
        bpg_decoder_close(decoder); return 1;
    }
    /* The frozen native-resolution baseline only admits256 RGB images. */
    if (info.width != 256 || info.height != 256 || info.has_animation || info.has_alpha) {
        bpg_decoder_close(decoder); return 2;
    }
    if (bpg_decoder_start(decoder, BPG_OUTPUT_FORMAT_RGB24) < 0) {
        bpg_decoder_close(decoder); return 1;
    }
    pixels = malloc((size_t)info.width * info.height * 3);
    if (!pixels) { bpg_decoder_close(decoder); return -2; }
    for (y = 0; y < (int)info.height; y++) {
        if (bpg_decoder_get_line(decoder, pixels + (size_t)y * info.width * 3) < 0) {
            free(pixels); bpg_decoder_close(decoder); return 1;
        }
    }
    *width = info.width;
    *height = info.height;
    *output = pixels;
    bpg_decoder_close(decoder);
    return 0;
}

void varcomm_bpg_free(void *pointer) { free(pointer); }
