set -e
gcc -g -pg -std=c99 -c -O1 -fno-inline main.c
gcc -g -pg -o main main.o -lm
./main
