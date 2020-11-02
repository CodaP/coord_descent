#define GRID_ROWS 3600
#define GRID_COLS 7200
#define SAT_ROWS 21696
#define SAT_COLS 21696

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
    if isnan(a[0]){
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


void idxmin_grid(float * grid_coords, float x, float y, float z, size_t start_i, size_t start_j, size_t *final_i, size_t *final_j, diag_t * diag){
    size_t i = start_i;
    size_t j = start_j;
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
        while(true){
            float dleft, dright;
            float dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
            //printf("%d %d %f\n", i, j, dnow);
            if((j > 0) && (j < (GRID_COLS-1))){
                dleft = d(&grid_coords[grididx(i,j-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,j+1)], xyz.xyz);
            } else if (j == 0) {
                // wrap
                dleft = d(&grid_coords[grididx(i,GRID_COLS-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,j+1)], xyz.xyz);
            } else if (j == GRID_COLS-1){
                dleft = d(&grid_coords[grididx(i,j-1)], xyz.xyz);
                dright = d(&grid_coords[grididx(i,0)], xyz.xyz);
            }
            if((dleft < dnow) && (dleft < dright)){
                first = false;
                j = (j-1) % GRID_COLS;
            } else if((dright < dnow) && (dright < dleft)){
                first = false;
                j = (j+1) % GRID_COLS;
            } else {
                // At minimum, don't move any more
                break;
            }
            num_steps_j++;
        }
        while(true){
            float dup, ddown;
            float dnow = d(&grid_coords[grididx(i,j)], xyz.xyz);
            //printf("%d %d %f\n", i, j, dnow);
            if((i > 0) && (i < (GRID_ROWS-1))){
                dup = d(&grid_coords[grididx(i-1,j)], xyz.xyz);
                ddown = d(&grid_coords[grididx(i+1,j)], xyz.xyz);
            } else if (j == 0) {
                dup = INFINITY;
                ddown = d(&grid_coords[grididx(i+1,j)], xyz.xyz);
            } else if (j == GRID_ROWS-1){
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
        }

        if(first){
            (*final_i) = i;
            (*final_j) = j;
            diag->num_steps_i = num_steps_i;
            diag->num_steps_j = num_steps_j;
            return;
        }
    }
}


void hemisphere(float * grid_coords, float * sat_coords, FILE* src_idx, FILE* dst_idx, bool north){
    const float threshold_radius = 5e3;
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
            idxmin_grid(grid_coords, x, y, z, start_i, start_j, &final_i, &final_j, &diag);
            float d2final = d(&grid_coords[grididx(final_i,final_j)], xyz.xyz);
            float dfinal = sqrtf(d2final);
            if(dfinal < threshold_radius){
                start_i = final_i;
                start_j = final_j;
                num_found++;
                uint32_t dst = final_i*GRID_COLS + final_j;
                uint32_t src = lat*SAT_COLS + lon;
                fwrite(&dst, sizeof(uint32_t), 1, dst_idx);
                fwrite(&src, sizeof(uint32_t), 1, src_idx);
                if((final_i*.05-90) > max_lat){
                    max_lat = final_i*.05 - 90;
                }
            }
            total++;
            total_nsteps_i += diag.num_steps_i;
            total_nsteps_j += diag.num_steps_j;
        }
        if( lat % 10 == 0 ){
            fprintf(stderr, "%d/%d found %d/%d; avg isteps: %f jsteps: %f max lat: %f\n", lat, SAT_ROWS, num_found, total, total_nsteps_i/(total+0.0), total_nsteps_j/(total+0.0), max_lat);
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
} worker_args_t;

void * worker(void * worker_args){
    worker_args_t * args = worker_args;
    float * grid_coords = args->grid_coords;
    float * sat_coords = args->sat_coords;
    FILE * src_idx = args->src_idx;
    FILE * dst_idx = args->dst_idx;
    bool north = args->north;
    hemisphere(grid_coords, sat_coords, src_idx, dst_idx, north);
    return NULL;
}

int main(){
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

    float * sat_coords = mmap(NULL, sat_file_length, PROT_READ, MAP_SHARED,
                  sat_fd, 0);
    float * grid_coords = mmap(NULL, grid_file_length, PROT_READ, MAP_SHARED,
                  grid_fd, 0);


    pthread_t north;
    worker_args_t north_args;
    north_args.sat_coords = sat_coords;
    north_args.grid_coords = grid_coords;
    north_args.src_idx = src_idx;
    north_args.dst_idx = dst_idx;
    north_args.north = true;
    //pthread_create(&north, NULL, worker, &north_args);
    worker(&north_args);

    pthread_t south;
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
