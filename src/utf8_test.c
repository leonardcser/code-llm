#include "utf8.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Test strip_accents_utf8
    {
        char *res = strip_accents_utf8("café");
        assert(strcmp(res, "cafe") == 0);
        free(res);
    }
    {
        char *res = strip_accents_utf8("Müller");
        assert(strcmp(res, "Muller") == 0);
        free(res);
    }
    {
        char *res = strip_accents_utf8("Straße");
        assert(strcmp(res, "Strase") == 0);
        free(res);
    }
    {
        char *res =
            strip_accents_utf8("Hello \xe4\xb8\x96\xe7\x95\x8c"); // 世界
        const char *expected = "Hello \xe4\xb8\x96\xe7\x95\x8c";
        assert(strcmp(res, expected) == 0);
        free(res);
    }
    {
        char *res = strip_accents_utf8(NULL);
        assert(res == NULL);
    }

    // Test to_ascii
    {
        char *res = to_ascii("café");
        assert(strcmp(res, "cafe") == 0);
        free(res);
    }
    {
        char *res = to_ascii("Müller");
        assert(strcmp(res, "Muller") == 0);
        free(res);
    }
    {
        char *res = to_ascii("Hello \xe4\xb8\x96\xe7\x95\x8c");
        assert(strcmp(res, "Hello ") == 0);
        free(res);
    }
    {
        char *res = to_ascii(NULL);
        assert(res == NULL);
    }
    return 0;
}
