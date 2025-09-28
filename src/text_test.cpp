#include "text.hpp"
#include <assert.h>

int main() {
    // Test strip_accents
    {
        std::string res = strip_accents("café");
        assert(res == "cafe");
    }
    {
        std::string res = strip_accents("Müller");
        assert(res == "Muller");
    }
    {
        std::string res = strip_accents("Straße");
        assert(res == "Strase");
    }
    {
        std::string res = strip_accents("Hello 世界");
        assert(res == "Hello 世界");
    }
    {
        std::string res = strip_accents("");
        assert(res == "");
    }

    // Test to_ascii
    {
        std::string res = to_ascii("café");
        assert(res == "cafe");
    }
    {
        std::string res = to_ascii("Müller");
        assert(res == "Muller");
    }
    {
        std::string res = to_ascii("Hello 世界");
        assert(res == "Hello ");
    }
    {
        std::string res = to_ascii("");
        assert(res == "");
    }
    return 0;
}
