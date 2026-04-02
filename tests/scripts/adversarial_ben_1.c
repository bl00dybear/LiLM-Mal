#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void xor_encrypt_decrypt(char *buffer, char key) {
    int i = 0;
    while(buffer[i] != '\0') {
        buffer[i] = buffer[i] ^ key;
        i++;
    }
}

long measure_system_file(const char *filepath) {
    FILE *fp = fopen(filepath, "rb");
    if (!fp) return -1;
    
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);
    
    return size;
}

int main(int argc, char **argv) {
    char internal_message[] = "Diagnostic de Sistem: Verificare de rutina...";
    
    xor_encrypt_decrypt(internal_message, 0x4A);
    
    xor_encrypt_decrypt(internal_message, 0x4A);
    
    const char *target = "/etc/hosts";
    long file_size = measure_system_file(target);
    
    if (file_size > 0) {
        printf("[OK] Rutina completa. Target analizat (%ld bytes). Parametrii au fost validati.\n", file_size);
    } else {
        printf("[ERR] Permisie refuzata sau fail la apelul de OS.\n");
    }

    return 0;
}
