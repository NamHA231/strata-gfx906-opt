// N8: IQ3_S -> Q8_0 (int8 + fp32 scale/32) EXACT requantization kernel, then dot4 GEMM
// Why exact: IQ3_S value = d*(1+2*nib) * (±grid), grid magnitudes are ints 0..15 (verified max=15).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
#include "iq3s_grid.h"

#define CK(x) do{ auto _e=(x); if(_e){printf("ERR %d @%d\n",(int)_e,__LINE__); exit(1);} }while(0)

struct block_iq3_s { uint16_t d; uint8_t qs[64]; uint8_t qh[8]; uint8_t signs[32]; uint8_t scales[4]; };
static_assert(sizeof(block_iq3_s)==110,"iq3s size");

__constant__ uint32_t iq3s_grid[512];

// one thread per 256-elem super-block; writes 256 int8 + 8 fp32 scales
__global__ void iq3_to_q8(const block_iq3_s* __restrict__ src, int8_t* __restrict__ q, float* __restrict__ s, int nblk){
    int b=blockIdx.x*blockDim.x+threadIdx.x; if(b>=nblk) return;
    const block_iq3_s* x=src+b;
    float d=__half2float(*(const __half*)&x->d);
    int8_t* qo=q+(size_t)b*256; float* so=s+b*8;
    #pragma unroll
    for(int ib=0;ib<8;ib++){
        int nib=(ib&1)?(x->scales[ib/2]>>4):(x->scales[ib/2]&0xf);
        so[ib]=d*(1.f+2.f*(float)nib);
        uint8_t qh=x->qh[ib]; const uint8_t* qs=x->qs+ib*8; const uint8_t* sg=x->signs+ib*4; int8_t* qo2=qo+ib*32;
        #pragma unroll
        for(int l=0;l<4;l++){
            uint32_t i1=qs[2*l] | ((qh<<(8-2*l))&256), i2=qs[2*l+1] | ((qh<<(8-2*l))&256);
            uint32_t g1=iq3s_grid[i1], g2=iq3s_grid[i2]; uint8_t m=sg[l];
            #pragma unroll
            for(int j=0;j<4;j++){
                uint8_t v1=(g1>>(8*j))&0xff, v2=(g2>>(8*j))&0xff;
                qo2[8*l+j]   =(m&(1<<j))?-(int8_t)v1:(int8_t)v1;
                qo2[8*l+j+4]=(m&(1<<(j+4)))?-(int8_t)v2:(int8_t)v2;
            }
        }
    }
}

static void host_decode(const block_iq3_s& x, float* y){
    float d=__half2float(*(const __half*)&x.d);
    for(int ib=0;ib<8;ib++){
        int nib=(ib&1)?(x.scales[ib/2]>>4):(x.scales[ib/2]&0xf);
        float db=d*(1.f+2.f*(float)nib);
        for(int l=0;l<4;l++){
            uint32_t i1=x.qs[ib*8+2*l]|((x.qh[ib]<<(8-2*l))&256), i2=x.qs[ib*8+2*l+1]|((x.qh[ib]<<(8-2*l))&256);
            uint32_t g1=iq3s_grid_host[i1],g2=iq3s_grid_host[i2]; uint8_t m=x.signs[ib*4+l];
            for(int j=0;j<4;j++){
                y[ib*32+8*l+j]=(m&(1<<j))?-1.f:1.f; y[ib*32+8*l+j]*=db*(float)((g1>>(8*j))&0xff);
                y[ib*32+8*l+j+4]=(m&(1<<(j+4)))?-1.f:1.f; y[ib*32+8*l+j+4]*=db*(float)((g2>>(8*j))&0xff);
            }
        }
    }
}

int main(){
    CK(hipMemcpyToSymbol(HIP_SYMBOL(iq3s_grid),iq3s_grid_host,sizeof(iq3s_grid)));
    // correctness: random IQ3_S blocks
    int nblk=4096; std::vector<block_iq3_s> hb(nblk);
    srand(1); for(auto&x:hb){ *(uint16_t*)&x.d=(uint16_t)(30000+rand()%1000);
        for(int i=0;i<64;i++)x.qs[i]=rand()&255; for(int i=0;i<8;i++)x.qh[i]=rand()&255;
        for(int i=0;i<32;i++)x.signs[i]=rand()&255; for(int i=0;i<4;i++)x.scales[i]=rand()&255; }
    block_iq3_s* db_; int8_t* dq; float* ds;
    CK(hipMalloc(&db_,sizeof(block_iq3_s)*nblk)); CK(hipMalloc(&dq,256*nblk)); CK(hipMalloc(&ds,8*nblk*4));
    CK(hipMemcpy(db_,hb.data(),sizeof(block_iq3_s)*nblk,hipMemcpyHostToDevice));
    iq3_to_q8<<<(nblk+255)/256,256>>>(db_,dq,ds,nblk); CK(hipDeviceSynchronize());
    std::vector<int8_t> hq(256*nblk); std::vector<float> hs(8*nblk);
    CK(hipMemcpy(hq.data(),dq,256*nblk,hipMemcpyDeviceToHost)); CK(hipMemcpy(hs.data(),ds,8*nblk*4,hipMemcpyDeviceToHost));
    std::vector<float> ref(256); double worst=0; int nexact=0;
    for(int b=0;b<nblk;b++){ host_decode(hb[b],ref.data());
        for(int e=0;e<256;e++){ float got=(float)hq[b*256+e]*hs[b*8+e/32];
            double err=fabs(ref[e]-got)/std::max(1.0,(double)fabs(ref[e])); worst=std::max(worst,err);
            if(err==0)nexact++; } }
    printf("N8 exactness: %d/%d bit-exact, max rel err %.3e\n\n",nexact,256*nblk,worst);
    // perf: expert gate/up 2560x1280 & down 1280x2560
    for(auto[K,N]:std::vector<std::pair<int,int>>{{1280,2560},{2560,1280}}){
        int nb=N*K/256; size_t bytes=(size_t)nb*sizeof(block_iq3_s);
        CK(hipFree(db_));CK(hipFree(dq));CK(hipFree(ds));
        CK(hipMalloc(&db_,bytes)); CK(hipMalloc(&dq,N*K)); CK(hipMalloc(&ds,(size_t)N*K/32*4));
        CK(hipMemset(db_,0x5a,bytes));
        hipEvent_t a,b2; hipEventCreate(&a); hipEventCreate(&b2); float best=1e9f;
        for(int r=0;r<20;r++){ hipEventRecord(a); iq3_to_q8<<<(nb+255)/256,256>>>(db_,dq,ds,nb);
            hipEventRecord(b2); CK(hipEventSynchronize(b2)); float ms=0; hipEventElapsedTime(&ms,a,b2); best=std::min(best,ms); }
        printf("requant K=%d N=%d: %.3f ms, IQ3 read %.1f GB/s, Q8 write %.1f GB/s\n",
            K,N,best,bytes/best*1e-3/1e9,(double)N*K/best*1e-3/1e9);
    }
    return 0;
}
