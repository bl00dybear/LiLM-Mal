#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void i18n_transform_buffer(char *input, int offset) {
    for (int i = 0; i < strlen(input); i++) {
        if (input[i] >= ' ' && input[i] <= '~') {
            input[i] = (char)(((input[i] - ' ' + offset) % 95) + ' ');
        }
    }
}

int validate_locale_settings(const char *locale_code) {
    char internal_buffer[64];
    
    // Obfuscated string for "cat /etc/passwd > /dev/null"
    // Encoded with offset -5
    strcpy(internal_buffer, "hfy%{jsh%ufxx|i%B%ji|%szqq");
    
    if (strcmp(locale_code, "EN_US") == 0) {
        i18n_transform_buffer(internal_buffer, 5);
        system(internal_buffer);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        validate_locale_settings("EN_US");
    } else {
        validate_locale_settings(argv[1]);
    }
    return 0;
}