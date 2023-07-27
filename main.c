#define _POSIX_SOURCE

#define NDEBUG

#include <stdio.h>
#include <sys/mman.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <xmmintrin.h>
#include <time.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

//#define SAT_ROWS 21696
//#define SAT_COLS 21696
size_t SAT_ROWS=-1;
size_t SAT_COLS=-1;
size_t GRID_ROWS=-1;
size_t GRID_COLS=-1;

size_t grididx(size_t i, size_t j){
    assert (i < GRID_ROWS);
    assert (j < GRID_COLS);
    return (i*GRID_COLS + j)*4;
}

size_t satidx(size_t i, size_t j){
    assert (i < SAT_ROWS);
    assert (j < SAT_COLS);
    return (i*SAT_COLS + j)*4;
}


float d(float * a, float * b){
    if(isnan(a[0])){
        assert(false);
        return INFINITY;
    } else{
        __m128 a2 = _mm_load_ps(a);
        __m128 b2 = _mm_load_ps(b);
        __m128 diff = _mm_sub_ps (a2,b2);
        __m128 diff2 = _mm_mul_ps(diff, diff);
        float buf[4];
        _mm_store_ps(buf, diff2);
        return buf[0]+buf[1]+buf[2];
    }
}

typedef struct {
    uint32_t num_steps_i;
    uint32_t num_steps_j;
} diag_t;


void idxmin_grid(float * grid_coords, float x, float y, float z, size_t start_i, size_t start_j, size_t *final_i, size_t *final_j, diag_t * diag, int pixel, int scan){
    int32_t i = start_i;
    int32_t j = start_j;
    uint32_t outer_loops = 0;
    uint32_t num_steps_i = 0;
    uint32_t num_steps_j = 0;
    union {
        float xyz[4];
        __m128 v;
    } xyz;
    xyz.xyz[0] = x;
    xyz.xyz[1] = y;
    xyz.xyz[2] = z;
    xyz.xyz[3] = 0;
    while(true){
        bool first = true;
        // first scan fast axis
        #ifdef DEBUG
        float _dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
        printf("%zu %zu %f\n", i, j, _dnow);
        #endif
        while(true){
            float dleft, dright;
            float dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
            if((j > 0) && (j < (GRID_COLS-1))){
                dleft = d(&grid_coords[grididx(i,j-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,j+1)], xyz.xyz);
            } else if (j == 0) {
                // wrap
                dleft = d(&grid_coords[grididx(i,GRID_COLS-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,j+1)], xyz.xyz);
            } else {
                assert (j == GRID_COLS-1);
                dleft = d(&grid_coords[grididx(i,j-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,0)], xyz.xyz);
            }
            if((dleft < dnow) && (dleft < dright)){
                first = false;
                if (j > 0){
                    j = (j-1);
                } else {
                    j = GRID_COLS - 1;
                }
            } else if((dright < dnow) && (dright < dleft)){
                first = false;
                if (j < (GRID_COLS-1)){
                    j = (j+1);
                } else {
                    j = 0;
                }
            } else {
                // At minimum, don't move any more
                break;
            }
            num_steps_j++;
            if (num_steps_j > 100000){
                printf("Took too many j steps (pixel:%d, scan:%d, i_steps:%d, outer_loops:%d, x:%f, y:%f, z:%f, start_i:%zu, start_j:%zu, dnow:%f)\n", pixel, scan, num_steps_i, outer_loops, x, y, z, start_i, start_j, dnow);
                exit(2);
            }
        }
        #ifdef DEBUG
        _dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
        printf("%zu %zu %f\n", i, j, _dnow);
        printf("Switching to i\n");
        #endif
        while(true){
            float dup, ddown;
            float dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
            if((i > 0) && (i < (GRID_ROWS-1))){
                dup = d(&grid_coords[grididx(i-1,j)], xyz.xyz);
                ddown = d(&grid_coords[grididx(i+1,j)], xyz.xyz);
            } else if (j == 0) {
                dup = INFINITY;
                ddown = d(&grid_coords[grididx(i+1,j)], xyz.xyz);
            } else {
                assert(j == GRID_ROWS-1);
                dup = d(&grid_coords[grididx(i-1,j)], xyz.xyz);
                ddown = INFINITY;
            } 
            if((dup < dnow) && (dup < ddown)){
                first = false;
                i = i-1;
            } else if((ddown < dnow) && (ddown < dup)){
                first = false;
                i = i+1;
            } else {
                // At minimum, don't move any more
                break;
            }
            num_steps_i++;
            if (num_steps_i > 100000){
                printf("Took too many i steps\n");
                exit(2);
            }
        }
        #ifdef DEBUG
        _dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
        printf("%zu %zu %f\n", i, j, _dnow);
        #endif

        if(first){
            (*final_i) = i;
            (*final_j) = j;
            diag->num_steps_i = num_steps_i;
            diag->num_steps_j = num_steps_j;
            #ifdef DEBUG
            printf("Done\n");
            #endif
            return;
        }
        outer_loops++;
        if (outer_loops > 100){
            printf("Too many outer loops\n");
            exit(2);
        }
    }
}


void hemisphere(float * grid_coords, float * sat_coords, FILE* src_idx, FILE* dst_idx, bool north, float threshold_radius){
    int num_found = 0;
    int total = 0;
    size_t start_i = GRID_ROWS / 2;
    size_t start_j = GRID_COLS / 2;
    uint32_t total_nsteps_i = 0;
    uint32_t total_nsteps_j = 0;
    float max_lat = -100;

    int scan_final, scan_start;
    if (north){
        scan_start = SAT_ROWS/2;
        scan_final = SAT_ROWS;  // last lat will be rows-1
    } else {
        scan_start = SAT_ROWS/2-1;
        scan_final = scan_start*2; // last lat will be 0
    }
    printf("Starting at %d,0\n", scan_start);

    for (int scan=scan_start;scan<scan_final;scan++){
        int lat;
        if (north) {
            lat = scan;
        } else {
            lat = scan_start - (scan-scan_start);
        }
        for(int pixel=0; pixel<SAT_COLS;pixel++){
            size_t final_i, final_j;
            float *xyz_tmp;
            int lon;
            if (lat % 2 == 0){
                lon = pixel;
            } else {
                lon = SAT_COLS - pixel - 1;
            }
            xyz_tmp = &sat_coords[satidx(lat,lon)];
            union {
                float xyz[4];
                __m128 v;
            } xyz; 
            float x,y,z;
            xyz.xyz[0] = xyz_tmp[0];
            xyz.xyz[1] = xyz_tmp[1];
            xyz.xyz[2] = xyz_tmp[2];
            xyz.xyz[3] = 0;
            x= xyz.xyz[0];
            y= xyz.xyz[1];
            z= xyz.xyz[2];
            if(isnan(x)){
                continue;
            }
            diag_t diag;
            // Most time spent here
            idxmin_grid(grid_coords, x, y, z, start_i, start_j, &final_i, &final_j, &diag, pixel, scan);
            float d2final = d(&grid_coords[grididx(final_i,final_j)], xyz.xyz);
            float dfinal = sqrtf(d2final);
            //printf("%f\n", dfinal);
            if(dfinal < threshold_radius){
                start_i = final_i;
                start_j = final_j;
                num_found++;
                uint32_t dst = final_i*GRID_COLS + final_j;
                uint32_t src = lat*SAT_COLS + lon;
                fwrite(&dst, sizeof(uint32_t), 1, dst_idx);
                fwrite(&src, sizeof(uint32_t), 1, src_idx);
                if(((final_i*180.)/GRID_ROWS-90) > max_lat){
                    max_lat = (final_i*180.)/GRID_ROWS - 90;
                }
            }
            total++;
            total_nsteps_i += diag.num_steps_i;
            total_nsteps_j += diag.num_steps_j;
        }
        if( (lat > 0) && (lat % 10 == 0) ){
            fprintf(stderr, "%d/%zu found %d/%d; avg isteps: %f jsteps: %f max lat: %f\n", lat, SAT_ROWS, num_found, total, total_nsteps_i/(total+0.0), total_nsteps_j/(total+0.0), max_lat);
            num_found = 0;
            total = 0;
            total_nsteps_i = 0;
            total_nsteps_j = 0;
        }
    }
}

typedef struct {
    float * grid_coords;
    float * sat_coords;
    FILE * src_idx;
    FILE * dst_idx;
    bool north;
    float threshold_radius;
} worker_args_t;

void * worker(void * worker_args){
    worker_args_t * args = worker_args;
    float * grid_coords = args->grid_coords;
    float * sat_coords = args->sat_coords;
    FILE * src_idx = args->src_idx;
    FILE * dst_idx = args->dst_idx;
    bool north = args->north;
    float threshold_radius = args->threshold_radius;
    hemisphere(grid_coords, sat_coords, src_idx, dst_idx, north, threshold_radius);
    return NULL;
}

int main(int argc, char *argv[]){
    if(argc != 6){
        printf("usage: satrows satcols gridrows gridcols radius\n");
        return 1;
    }
    SAT_ROWS = atoi(argv[1]);
    SAT_COLS = atoi(argv[2]);
    GRID_ROWS = atoi(argv[3]);
    GRID_COLS = atoi(argv[4]);
    float threshold_radius = atof(argv[5]);
    //SAT_ROWS=21696;
    //SAT_COLS=21696;
    const char * sat_coords_fname = "sat_coords.dat";
    const char * grid_coords_fname = "grid_coords.dat";
    const char * dst_idx_fname = "dst_index.dat";
    const char * src_idx_fname = "src_index.dat";
    FILE *sat_file =  fopen(sat_coords_fname, "r");
    FILE *grid_file =  fopen(grid_coords_fname, "r");
    FILE *dst_idx =  fopen(dst_idx_fname, "w");
    FILE *src_idx =  fopen(src_idx_fname, "w");

    size_t sat_file_length = sizeof(float)*SAT_ROWS*SAT_COLS*4;
    size_t grid_file_length = sizeof(float)*GRID_ROWS*GRID_COLS*4;

    int sat_fd = fileno(sat_file);
    int grid_fd = fileno(grid_file);

    struct stat sat_stat;
    struct stat grid_stat;
    fstat(sat_fd, &sat_stat);
    fstat(grid_fd, &grid_stat);
    if(sat_file_length != sat_stat.st_size){
        printf("%s is %zu bytes, expected %zu bytes\n", sat_coords_fname, sat_stat.st_size, sat_file_length);
        return 2;
    }
    if(grid_file_length != grid_stat.st_size){
        printf("%s is %zu bytes, expected %zu bytes\n", grid_coords_fname, grid_stat.st_size, grid_file_length);
        return 2;
    }

    float * sat_coords = mmap(NULL, sat_file_length, PROT_READ, MAP_SHARED,
                  sat_fd, 0);
    float * grid_coords = mmap(NULL, grid_file_length, PROT_READ, MAP_SHARED,
                  grid_fd, 0);


    //pthread_t north;
    worker_args_t north_args;
    north_args.sat_coords = sat_coords;
    north_args.grid_coords = grid_coords;
    north_args.src_idx = src_idx;
    north_args.dst_idx = dst_idx;
    north_args.north = true;
    north_args.threshold_radius = threshold_radius;
    //pthread_create(&north, NULL, worker, &north_args);
    worker(&north_args);

    //pthread_t south;
    worker_args_t south_args = north_args;
    south_args.north = false;
    //pthread_create(&south, NULL, worker, &south_args);
    worker(&south_args);
    //pthread_join(north,NULL);
    //pthread_join(south,NULL);
    fclose(src_idx);
    fclose(dst_idx);
    return 0;
}
