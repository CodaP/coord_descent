set -e
gcc -g -pg -std=c99 -c -O2 main.c
gcc -g -pg -o main main.o -lm
#gcc -std=c99 -c -O2 main.c
#gcc -o main main.o -lm
./main
