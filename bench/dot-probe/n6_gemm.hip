#define HIPBLAS_V2
// N6: expert-GEMM dot4 (int8 x int8, q8_0 block-32 scales) vs hipBLAS HGEMM (fp16 path)
// Design wave64-native: block=256 (4 waves), tile 64Mx64N, wave-tile 16x16 (lane: 1 row x 4 col),
// k-tile 64 elements in LDS (swizzled quads, b128 LDS), i32 acc flushed per 32-elem scale block.
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CK(x) do{ auto _e=(x); if(_e){printf("ERR %d @%d\n",(int)_e,__LINE__); exit(1);} }while(0)

static void quant8(const float* src, int rows, int K, std::vector<int8_t>& q, std::vector<float>& s){
    q.resize((size_t)rows*K); s.resize((size_t)rows*(K/32));
    for(int r=0;r<rows;r++) for(int b=0;b<K/32;b++){
        float mx=0; for(int i=0;i<32;i++) mx=std::max(mx,std::fabs(src[(size_t)r*K+b*32+i]));
        float sc=mx>0?mx/127.f:1.f; size_t bi=(size_t)r*(K/32)+b; s[bi]=sc; float inv=1.f/sc;
        for(int i=0;i<32;i++){ float v=src[(size_t)r*K+b*32+i]*inv;
            q[(size_t)r*K+b*32+i]=(int8_t)lroundf(std::max(-127.f,std::min(127.f,v))); }
    }
}

#define SW(row,q) ( (row)*16 + ( ((q) ^ ((row)&3)) << 2 ) )

__launch_bounds__(256,1) __global__ void gemm_dot4(const int8_t* __restrict__ Aq,const float* __restrict__ dA,
        const int8_t* __restrict__ Bq,const float* __restrict__ dB,
        float* __restrict__ C,int M,int N,int K){
    __shared__ int as[64*16], bs[64*16];
    __shared__ float das[64*2], dbs[64*2];
    const int n0=blockIdx.x*64, m0=blockIdx.y*64, t=threadIdx.x;
    const int row=t>>2, cq=t&3;
    const int wid=t>>6, lane=t&63;
    const int rb=(wid>>1)*32+(lane&7)*4, cb=(wid&1)*32+(lane>>3)*4;
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++)for(int j=0;j<4;j++)acc[i][j]=0.f;
    int4 va,vb,va2,vb2; int gm=m0+row;
    va=(gm<M)?*(const int4*)(Aq+(size_t)gm*K+cq*16):make_int4(0,0,0,0);
    vb=*(const int4*)(Bq+(size_t)(n0+row)*K+cq*16);
    for(int k0=0;k0<K;k0+=64){
        *(int4*)&as[SW(row,cq)]=va; *(int4*)&bs[SW(row,cq)]=vb;
        if(t<128){ int r=t>>1,sb=t&1;
            das[r*2+sb]=(m0+r<M)?dA[(size_t)(m0+r)*(K/32)+(k0>>5)+sb]:0.f;
            dbs[r*2+sb]=dB[(size_t)(n0+r)*(K/32)+(k0>>5)+sb]; }
        __syncthreads();
        if(k0+64<K){
            va2=(gm<M)?*(const int4*)(Aq+(size_t)gm*K+k0+64+cq*16):make_int4(0,0,0,0);
            vb2=*(const int4*)(Bq+(size_t)(n0+row)*K+k0+64+cq*16);
        }
        #pragma unroll
        for(int sb=0;sb<2;sb++){
            int ia[4][4];
            #pragma unroll
            for(int i=0;i<4;i++)for(int j=0;j<4;j++)ia[i][j]=0;
            #pragma unroll
            for(int q=0;q<2;q++){
                int4 Aa[4],Bb[4];
                #pragma unroll
                for(int i=0;i<4;i++){ int r=rb+i; Aa[i]=*(int4*)&as[r*16+(((sb*2+q)^(r&3))<<2)]; }
                #pragma unroll
                for(int j=0;j<4;j++){ int c=cb+j; Bb[j]=*(int4*)&bs[c*16+(((sb*2+q)^(c&3))<<2)]; }
                #pragma unroll
                for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++){
                    ia[i][j]=__builtin_amdgcn_sdot4(Aa[i].x,Bb[j].x,ia[i][j],false);
                    ia[i][j]=__builtin_amdgcn_sdot4(Aa[i].y,Bb[j].y,ia[i][j],false);
                    ia[i][j]=__builtin_amdgcn_sdot4(Aa[i].z,Bb[j].z,ia[i][j],false);
                    ia[i][j]=__builtin_amdgcn_sdot4(Aa[i].w,Bb[j].w,ia[i][j],false);
                }
            }
            #pragma unroll
            for(int i=0;i<4;i++)
            #pragma unroll
            for(int j=0;j<4;j++)
                acc[i][j]+=(float)ia[i][j]*das[(rb+i)*2+sb]*dbs[(cb+j)*2+sb];
        }
        __syncthreads();
        va=va2; vb=vb2;
    }
    #pragma unroll
    for(int i=0;i<4;i++){ int gm2=m0+rb+i;
        if(gm2<M){ size_t o=(size_t)gm2*N+n0+cb;
            C[o]=acc[i][0]; C[o+1]=acc[i][1]; C[o+2]=acc[i][2]; C[o+3]=acc[i][3]; } }
}

static float bench(void(*launch)(), const char* name){
    hipEvent_t a,b; hipEventCreate(&a); hipEventCreate(&b);
    launch(); CK(hipDeviceSynchronize());
    float best=1e9f;
    for(int r=0;r<3;r++){ hipEventRecord(a); launch(); hipEventRecord(b);
        CK(hipEventSynchronize(b)); float ms=0; hipEventElapsedTime(&ms,a,b); best=std::min(best,ms); }
    printf("%-14s %9.3f ms\n",name,best); return best;
}

static int M,N,K;
static int8_t *Aq,*Bq; static float *dA,*dB,*C,*A16,*B16,*Cref16;
static hipblasHandle_t hb;

static void run_dot(){
    dim3 g(N/64,(M+63)/64);
    gemm_dot4<<<g,256>>>(Aq,dA,Bq,dB,C,M,N,K);
}
static void run_blas(){
    float al=1.f,be=0.f;
    // C_cm(N,M) = B_cm(K,N)^T * A_cm(K,M): A_rm[M][K] -> cm [K][M], B_rm[N][K] -> cm [K][N]
    CK(hipblasGemmEx(hb,HIPBLAS_OP_T,HIPBLAS_OP_N,N,M,K,&al,B16,HIP_R_16F,K,A16,HIP_R_16F,K,&be,C,HIP_R_32F,N,HIPBLAS_COMPUTE_32F,HIPBLAS_GEMM_DEFAULT));
}

int main(){
    // ---------- correctness ----------
    { int m=64,n=64,k=128; std::vector<float> Af(m*k),Bf(n*k);
      srand(42); for(auto&x:Af)x=(rand()%2000-1000)/500.f; for(auto&x:Bf)x=(rand()%2000-1000)/500.f;
      std::vector<int8_t> q,qb; std::vector<float> s,sb; quant8(Af.data(),m,k,q,s); quant8(Bf.data(),n,k,qb,sb);
      int8_t *qa,*qb2; float *sa,*sb2,*cc;
      CK(hipMalloc(&qa,q.size())); CK(hipMalloc(&qb2,qb.size()));
      CK(hipMalloc(&sa,s.size()*4)); CK(hipMalloc(&sb2,sb.size()*4)); CK(hipMalloc(&cc,(size_t)m*n*4));
      CK(hipMemcpy(qa,q.data(),q.size(),hipMemcpyHostToDevice)); CK(hipMemcpy(qb2,qb.data(),qb.size(),hipMemcpyHostToDevice));
      CK(hipMemcpy(sa,s.data(),s.size()*4,hipMemcpyHostToDevice)); CK(hipMemcpy(sb2,sb.data(),sb.size()*4,hipMemcpyHostToDevice));
      gemm_dot4<<<dim3(n/64,m/64),256>>>(qa,sa,qb2,sb2,cc,m,n,k); CK(hipDeviceSynchronize());
      std::vector<float> hc(m*n); CK(hipMemcpy(hc.data(),cc,(size_t)m*n*4,hipMemcpyDeviceToHost));
      double worst=0;
      for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double ref=0;
          for(int b=0;b<k/32;b++){ int blk=0; for(int e=0;e<32;e++) blk+=(int)q[i*k+b*32+e]*(int)qb[j*k+b*32+e];
              ref+=blk*(double)s[i*(k/32)+b]*sb[j*(k/32)+b]; }
          double err=fabs(ref-hc[i*n+j])/std::max(1.0,fabs(ref)); worst=std::max(worst,err); }
      printf("correctness max rel err = %.3e\n\n",worst);
      hipFree(qa);hipFree(qb2);hipFree(sa);hipFree(sb2);hipFree(cc); }

    // ---------- benchmark: expert shapes ----------
    CK(hipblasCreate(&hb));
    struct S{const char*tag;int m,n,k;} shapes[]={
        {"gate/up M40 ",40,2560,1280},{"gate/up M128",128,2560,1280},{"gate/up M512",512,2560,1280},
        {"down M40    ",40,1280,2560},{"down M512   ",512,1280,2560}};
    for(auto&sh:shapes){
        M=sh.m;N=sh.n;K=sh.k;
        std::vector<float> Af((size_t)M*K),Bf((size_t)N*K);
        srand(7); for(auto&x:Af)x=(rand()%2000-1000)/500.f; for(auto&x:Bf)x=(rand()%2000-1000)/500.f;
        std::vector<int8_t> q,qb; std::vector<float> s,sb;
        quant8(Af.data(),M,K,q,s); quant8(Bf.data(),N,K,qb,sb);
        std::vector<__half> ha16((size_t)M*K),hb16((size_t)N*K);
        for(size_t i=0;i<Af.size();i++) ha16[i]=__float2half(Af[i]);
        for(size_t i=0;i<Af.size();i++){} // noop
        for(size_t r=0;r<(size_t)N;r++)for(int b=0;b<K/32;b++)for(int e=0;e<32;e++)
            hb16[r*K+b*32+e]=__float2half((float)qb[r*K+b*32+e]*sb[r*(K/32)+b]);
        CK(hipMalloc(&Aq,q.size())); CK(hipMalloc(&Bq,qb.size()));
        CK(hipMalloc(&dA,s.size()*4)); CK(hipMalloc(&dB,sb.size()*4));
        CK(hipMalloc(&C,(size_t)M*N*4));
        CK(hipMalloc(&A16,ha16.size()*2)); CK(hipMalloc(&B16,hb16.size()*2));
        CK(hipMemcpy(Aq,q.data(),q.size(),hipMemcpyHostToDevice)); CK(hipMemcpy(Bq,qb.data(),qb.size(),hipMemcpyHostToDevice));
        CK(hipMemcpy(dA,s.data(),s.size()*4,hipMemcpyHostToDevice)); CK(hipMemcpy(dB,sb.data(),sb.size()*4,hipMemcpyHostToDevice));
        CK(hipMemcpy(A16,ha16.data(),ha16.size()*2,hipMemcpyHostToDevice)); CK(hipMemcpy(B16,hb16.data(),hb16.size()*2,hipMemcpyHostToDevice));
        double flop=2.0*M*N*K;
        printf("[%s K=%d N=%d]\n",sh.tag,K,N);
        float t1=bench(run_dot,"dot4");
        float t2=bench(run_blas,"hipBLAS f16");
        printf("   dot4: %.2f TFLOP/s | f16: %.2f TFLOP/s | dot4/f16 = %.2fx\n\n",
            flop/t1*1e3/1e12, flop/t2*1e3/1e12, t2/t1);
        hipFree(Aq);hipFree(Bq);hipFree(dA);hipFree(dB);hipFree(C);hipFree(A16);hipFree(B16);
    }
    CK(hipblasDestroy(hb)); return 0;
}
