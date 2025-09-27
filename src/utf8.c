#include <locale.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <wctype.h>

// Function to strip accents from a wide character
wchar_t strip_accent_wchar(wchar_t wc) {
    switch (wc) {
    // French accented characters
    case L'À':
    case L'Á':
    case L'Â':
    case L'Ã':
    case L'Ä':
    case L'Å':
    case L'à':
    case L'á':
    case L'â':
    case L'ã':
    case L'ä':
    case L'å':
        return L'a';

    case L'È':
    case L'É':
    case L'Ê':
    case L'Ë':
    case L'è':
    case L'é':
    case L'ê':
    case L'ë':
        return L'e';

    case L'Ì':
    case L'Í':
    case L'Î':
    case L'Ï':
    case L'ì':
    case L'í':
    case L'î':
    case L'ï':
        return L'i';

    case L'Ò':
    case L'Ó':
    case L'Ô':
    case L'Õ':
    case L'Ö':
    case L'ò':
    case L'ó':
    case L'ô':
    case L'õ':
    case L'ö':
        return L'o';

    case L'Ù':
    case L'Ú':
    case L'Û':
    case L'Ü':
    case L'ù':
    case L'ú':
    case L'û':
    case L'ü':
        return L'u';

    case L'Ý':
    case L'Ÿ':
    case L'ý':
    case L'ÿ':
        return L'y';

    case L'Ç':
    case L'ç':
        return L'c';

    case L'Ñ':
    case L'ñ':
        return L'n';

    case L'ß': // German eszett, it shoud be 'ss' but we keep it simple
        return L's';

    default:
        return wc;
    }
}

// Function to strip accents from a UTF-8 string
char *strip_accents_utf8(const char *input) {
    if (!input) return NULL;

    // Set locale to handle UTF-8
    setlocale(LC_ALL, "en_US.UTF-8");

    // Convert input to wide string
    size_t wlen = mbstowcs(NULL, input, 0);
    if (wlen == (size_t)-1) {
        return NULL; // Invalid multibyte sequence
    }

    wchar_t *wide_input = malloc((wlen + 1) * sizeof(wchar_t));
    if (!wide_input) return NULL;

    mbstowcs(wide_input, input, wlen + 1);

    // Process wide characters
    wchar_t *wide_output = malloc((wlen + 1) * sizeof(wchar_t));
    if (!wide_output) {
        free(wide_input);
        return NULL;
    }

    for (size_t i = 0; i < wlen; i++) {
        wide_output[i] = strip_accent_wchar(wide_input[i]);
    }
    wide_output[wlen] = L'\0';

    // Convert back to multibyte string
    size_t output_size = wcstombs(NULL, wide_output, 0);
    if (output_size == (size_t)-1) {
        free(wide_input);
        free(wide_output);
        return NULL;
    }

    char *output = malloc(output_size + 1);
    if (!output) {
        free(wide_input);
        free(wide_output);
        return NULL;
    }

    wcstombs(output, wide_output, output_size + 1);

    free(wide_input);
    free(wide_output);
    return output;
}

// Function to convert to ASCII by stripping accents and removing non-ASCII
char *to_ascii(const char *input) {
    if (!input) return NULL;

    setlocale(LC_ALL, "en_US.UTF-8");

    // Convert to wide string
    size_t wlen = mbstowcs(NULL, input, 0);
    if (wlen == (size_t)-1) return NULL;

    wchar_t *wide_input = malloc((wlen + 1) * sizeof(wchar_t));
    if (!wide_input) return NULL;

    mbstowcs(wide_input, input, wlen + 1);

    // Process and filter to ASCII
    char *output =
        malloc(wlen + 1); // ASCII output will be <= wide input length
    if (!output) {
        free(wide_input);
        return NULL;
    }

    size_t out_pos = 0;
    for (size_t i = 0; i < wlen; i++) {
        wchar_t stripped = strip_accent_wchar(wide_input[i]);
        // Only keep ASCII characters
        if (stripped < 128) {
            output[out_pos++] = (char)stripped;
        }
    }
    output[out_pos] = '\0';

    free(wide_input);
    return output;
}
