#define GRID_ROWS 3600
#define GRID_COLS 7200
#define SAT_ROWS 5424
#define SAT_COLS 5424

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


size_t grididx(size_t i, size_t j){
    assert (i < GRID_ROWS);
    assert (j < GRID_COLS);
    return (i*GRID_COLS + j)*3;
}

size_t satidx(size_t i, size_t j){
    assert (i < SAT_ROWS);
    assert (j < SAT_COLS);
    return (i*SAT_COLS + j)*4;
    //div_t di = div(i,16);
    //div_t dj = div(j,16);
    //size_t chunki = di.quot;
    //size_t chunkj = dj.quot;
    //fprintf(stderr, "%d %d %d %d\n", chunki, chunkj, di.rem, dj.rem);
    //339, 339, 16, 16, 4
    //return (chunki*339*16*16 + + chunkj*16*16+di.rem*16+dj.rem)*4;
}


float d(float * a, float * b){
    if isnan(a[0]){
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


float dslow(float * a, float * b){
    if isnan(a[0]){
        return INFINITY;
    } else{
        float dx = a[0]-b[0];
        float dy = a[1]-b[1];
        float dz = a[2]-b[2];
        return dx*dx + dy*dy + dz*dz;
    }
}

typedef struct {
    uint32_t num_steps_i;
    uint32_t num_steps_j;
} diag_t;

void idxmin(float * sat_coords, float x, float y, float z, size_t start_i, size_t start_j, size_t *final_i, size_t *final_j, diag_t * diag){
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
            float dnow = d(&sat_coords[satidx(i,j)], xyz.xyz);
            //printf("%d %d %f\n", i, j, dnow);
            if((j > 0) && (j < (SAT_COLS-1))){
                dleft = d(&sat_coords[satidx(i,j-1)], xyz.xyz);
                dright = d(&sat_coords[satidx(i,j+1)], xyz.xyz);
            } else if (j == 0) {
                dleft = INFINITY;
                dright = d(&sat_coords[satidx(i,j+1)], xyz.xyz);
            } else if (j == SAT_COLS-1){
                dleft = d(&sat_coords[satidx(i,j-1)], xyz.xyz);
                dright = INFINITY;
            }
            if((dleft < dnow) && (dleft < dright)){
                first = false;
                j = j-1;
            } else if((dright < dnow) && (dright < dleft)){
                first = false;
                j = j+1;
            } else {
                // At minimum, don't move any more
                break;
            }
            num_steps_j++;
        }
        while(true){
            float dup, ddown;
            float dnow = d(&sat_coords[satidx(i,j)], xyz.xyz);
            //printf("%d %d %f\n", i, j, dnow);
            if((i > 0) && (i < (SAT_ROWS-1))){
                dup = d(&sat_coords[satidx(i-1,j)], xyz.xyz);
                ddown = d(&sat_coords[satidx(i+1,j)], xyz.xyz);
            } else if (j == 0) {
                dup = INFINITY;
                ddown = d(&sat_coords[satidx(i+1,j)], xyz.xyz);
            } else if (j == SAT_ROWS-1){
                dup = d(&sat_coords[satidx(i-1,j)], xyz.xyz);
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


void idxmin_fast(float * sat_coords, float x, float y, float z, size_t start_i, size_t start_j, size_t *final_i, size_t *final_j, diag_t * diag){
    idxmin(sat_coords, x, y, z, start_i, start_j, final_i, final_j, diag);
}


void idxmin_slow(float * sat_coords, float x, float y, float z, size_t start_i, size_t start_j, size_t *final_i, size_t *final_j, diag_t * diag){
    idxmin(sat_coords, x, y, z, 2500, 2500, final_i, final_j, diag);
}
int main(){
    const char * sat_coords_fname = "sat_coords.dat";
    const char * grid_coords_fname = "grid_coords.dat";
    FILE *sat_file =  fopen(sat_coords_fname, "r");
    FILE *grid_file =  fopen(grid_coords_fname, "r");

    size_t sat_file_length = sizeof(float)*SAT_ROWS*SAT_COLS*4;
    size_t grid_file_length = sizeof(float)*GRID_ROWS*GRID_COLS*3;

    int sat_fd = fileno(sat_file);
    int grid_fd = fileno(grid_file);

    float * sat_coords = mmap(NULL, sat_file_length, PROT_READ, MAP_SHARED,
                  sat_fd, 0);
    float * grid_coords = mmap(NULL, grid_file_length, PROT_READ, MAP_SHARED,
                  grid_fd, 0);

    //printf("%f\n", sat_coords[satidx(2500,2500)]);
    //return;
    
    //time_t time(time_t *t);
    int num_found = 0;
    int total = 0;
    size_t start_i = 2500;
    size_t start_j = 2500;
    uint32_t total_nsteps_i = 0;
    uint32_t total_nsteps_j = 0;
    uint32_t fast_nsteps_i = 0;
    uint32_t fast_nsteps_j = 0;
    for (int lat=1800;lat<GRID_ROWS;lat++){
    //for (int lat=500;lat<3000;lat++){
        //for(int lon=700; lon<2000;lon++){
        
        for(int lon=0; lon<GRID_COLS;lon++){
            size_t final_i, final_j;
            float *xyz_tmp;
            if (lat % 2 == 0){
                xyz_tmp = &grid_coords[grididx(lat,lon)];
            } else {
                xyz_tmp = &grid_coords[grididx(lat,GRID_COLS - lon)];
            }
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
            //fprintf(stderr, "%f %f %f\n",x,y,z);
            diag_t diag;
            if((start_i == 2500) && (start_j==2500)){
                idxmin_slow(sat_coords, x, y, z, start_i, start_j, &final_i, &final_j, &diag);
            } else {
                idxmin_fast(sat_coords, x, y, z, start_i, start_j, &final_i, &final_j, &diag);
            }
            float d2final = d(&sat_coords[satidx(final_i,final_j)], xyz.xyz);
            float dfinal = sqrtf(d2final);
            if(dfinal < 100e3){
                start_i = final_i;
                start_j = final_j;
                fast_nsteps_i += diag.num_steps_i;
                fast_nsteps_j += diag.num_steps_j;
                num_found++;
            } else {
                //start_i = 2500;
                //start_j = 2500;
            }
            total++;
            total_nsteps_i += diag.num_steps_i;
            total_nsteps_j += diag.num_steps_j;
            //printf("%d %d %f\n", final_i, final_j, dfinal);
            //break;
        }
        if( lat % 10 == 0 ){
            fprintf(stderr, "%d/%d found %d/%d; avg isteps: %f jsteps: %f\n", lat, GRID_ROWS, num_found, total, total_nsteps_i/(total+0.0), total_nsteps_j/(total+0.0));
            num_found = 0;
            total = 0;
            total_nsteps_i = 0;
            total_nsteps_j = 0;
            fast_nsteps_i = 0;
            fast_nsteps_j = 0;
        }
        //break;
    }
    //fprintf(stderr, "%d %d\n", final_i, final_j);
    //float *xyz = &sat_coords[satidx(final_i,final_j)];
    //x = xyz[0];
    //y = xyz[1];
    //z = xyz[2];
    //fprintf(stderr, "%f %f %f\n", x,y,z);
    return 0;
}
