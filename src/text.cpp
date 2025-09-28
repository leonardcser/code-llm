#include <string>
#include <unordered_map>

std::string strip_accents(const std::string &input) {
    // Map of accented characters to their ASCII equivalents
    // Covers common French and German accented characters
    static const std::unordered_map<std::string, char> accentMap = {
        // French accented characters
        {"à", 'a'}, {"á", 'a'}, {"â", 'a'}, {"ã", 'a'}, {"ä", 'a'}, {"À", 'A'},
        {"Á", 'A'}, {"Â", 'A'}, {"Ã", 'A'}, {"Ä", 'A'}, {"è", 'e'}, {"é", 'e'},
        {"ê", 'e'}, {"ë", 'e'}, {"È", 'E'}, {"É", 'E'}, {"Ê", 'E'}, {"Ë", 'E'},
        {"ì", 'i'}, {"í", 'i'}, {"î", 'i'}, {"ï", 'i'}, {"Ì", 'I'}, {"Í", 'I'},
        {"Î", 'I'}, {"Ï", 'I'}, {"ò", 'o'}, {"ó", 'o'}, {"ô", 'o'}, {"õ", 'o'},
        {"ö", 'o'}, {"Ò", 'O'}, {"Ó", 'O'}, {"Ô", 'O'}, {"Õ", 'O'}, {"Ö", 'O'},
        {"ù", 'u'}, {"ú", 'u'}, {"û", 'u'}, {"ü", 'u'}, {"Ù", 'U'}, {"Ú", 'U'},
        {"Û", 'U'}, {"Ü", 'U'}, {"ç", 'c'}, {"Ç", 'C'}, {"ÿ", 'y'}, {"Ÿ", 'Y'},
        {"ß", 's'} // German eszett (sharp s)
    };

    std::string result;
    result.reserve(input.length());

    for (size_t i = 0; i < input.length();) {
        // Check if current position starts a UTF-8 multibyte character
        if ((input[i] & 0x80) == 0) {
            // ASCII character, just copy it
            result += input[i];
            i++;
        } else {
            // UTF-8 multibyte character
            // Determine length of UTF-8 character
            int charLength = 1;
            if ((input[i] & 0xE0) == 0xC0)
                charLength = 2;
            else if ((input[i] & 0xF0) == 0xE0)
                charLength = 3;
            else if ((input[i] & 0xF8) == 0xF0)
                charLength = 4;

            // Extract the UTF-8 character
            if (i + charLength <= input.length()) {
                std::string utf8Char = input.substr(i, charLength);

                // Look up the character in our accent map
                auto it = accentMap.find(utf8Char);
                if (it != accentMap.end()) {
                    result += it->second;
                } else {
                    // Unknown character, keep as is
                    result += utf8Char;
                }

                i += charLength;
            } else {
                // Incomplete UTF-8 sequence, just copy the byte
                result += input[i];
                i++;
            }
        }
    }

    return result;
}

std::string to_ascii(const std::string &input) {
    // First strip accents
    std::string stripped = strip_accents(input);

    // Then remove any remaining non-ASCII characters
    std::string result;
    result.reserve(stripped.length());

    for (char c : stripped) {
        if (static_cast<unsigned char>(c) < 128) {
            result += c;
        }
    }

    return result;
}
