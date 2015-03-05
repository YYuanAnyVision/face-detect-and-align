#include <iostream>
#include <math.h>
#include <string.h>

#include "sseFun.h"
#include "wrappers.hpp"
#define PI 3.14159265358979323846264338


using namespace std;

// HOG helper: compute HOG or FHOG channels
void hogChannels( float *feature,               // out: output feature
                  const float *gradientHist,    // in : gradient Hist
                  const float *normalizedValue, // in :
                  int hb,                       // in : number of block in height direction
                  int wb,                       // in : number of block in width direction
                  int nOrients,                 // in : number of orientation
                  float clip,                   // in : clip value
                  int type )                    // in : 0 - > store each orientation and normalization (nOrients*4 channels) hog
                                                //      1 - > sum across all normalizatins, fhog. 2 -> sum accross all orientations
{
  #define GETT(blk) t=R1[y]*N1[y-(blk)]; if(t>clip) t=clip; c++;
  const float r=.2357f; int o, x, y, c; float t;
  const int nb=wb*hb, nbo=nOrients*nb, wb1=wb+1;

  for( o=0; o<nOrients; o++ ) for( x=0; x<hb; x++ ) 
  {
    const float *R1=gradientHist+o*nb+x*wb, *N1=normalizedValue+x*wb1+wb1+1;
    float *H1 = (type<=1) ? (feature+o*nb+x*wb) : (feature+x*wb);

    if( type==0) for( y=0; y<wb; y++ ) {
      // store each orientation and normalization (nOrients*4 channels)
      c=-1; GETT(0); H1[c*nbo+y]=t; GETT(wb1); H1[c*nbo+y]=t;
      GETT(1); H1[c*nbo+y]=t; GETT(wb1+1); H1[c*nbo+y]=t;
    } else if( type==1 ) for( y=0; y<wb; y++ ) {
      // sum across all normalizations (nOrients channels)
      c=-1; GETT(0); H1[y]+=t*.5f; GETT(wb1); H1[y]+=t*.5f;
      GETT(1); H1[y]+=t*.5f; GETT(wb1+1); H1[y]+=t*.5f;
    } else if( type==2 ) for( y=0; y<wb; y++ ) {
      // sum across all orientations (4 channels)
      c=-1; GETT(0); H1[c*nb+y]+=t*r; GETT(wb1); H1[c*nb+y]+=t*r;
      GETT(1); H1[c*nb+y]+=t*r; GETT(wb1+1); H1[c*nb+y]+=t*r;
    }
  }
  #undef GETT
}

// HOG helper: compute 2x2 block normalization values (padded by 1 pixel)
float* hogNormMatrix( float *gradientHist,  //in : gradientHist
                      int nOrients,         //in : number of orientation
                      int hb,               //in : number of block in y direction
                      int wb,               //in : number of block in x direction
                      int binSize )         //in : binSize
{
  float *N, *N1, *n; int o, x, y, dx, dy, hb1=hb+1, wb1=wb+1;
  float eps = 1e-4f/4/binSize/binSize/binSize/binSize; // precise backward equality
  N = (float*) wrCalloc(hb1*wb1,sizeof(float)); N1=N+wb1+1;

  for( o=0; o<nOrients; o++ )for( y=0; y<hb; y++ )for( x=0; x<wb; x++ )
    N1[y*wb1+x] += gradientHist[o*wb*hb+y*wb+x]*gradientHist[o*wb*hb+y*wb+x];

   for( y=0; y<hb-1; y++ ) for( x=0; x<wb-1; x++ ) {
    n=N1+y*wb1+x; *n=1/float(sqrt(n[0]+n[1]+n[wb1]+n[wb1+1]+eps)); }
    
   //special case on the border
  x=0;     dx= 1; dy= 1; y=0;                  N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  x=0;     dx= 1; dy= 0; for(y=0; y<hb1; y++)  N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  x=0;     dx= 1; dy=-1; y=hb1-1;              N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  x=wb1-1; dx=-1; dy= 1; y=0;                  N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  x=wb1-1; dx=-1; dy= 0; for( y=0; y<hb1; y++) N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  x=wb1-1; dx=-1; dy=-1; y=hb1-1;              N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  y=0;     dx= 0; dy= 1; for(x=0; x<wb1; x++)  N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  y=hb1-1; dx= 0; dy=-1; for(x=0; x<wb1; x++)  N[x+y*wb1]=N[(x+dx)+(y+dy)*wb1];
  return N;
}


// compute HOG features
void ssehog(const float *Mag,   //in : magnitude
            const float *Ori,   //in : orientation
            float *feature,     //out: computed feature
            int height,         //in : height of the image
            int width,          //in : width of the image
            int binSize,        //in : binSize of the cell, eg 8
            int nOrients,       //in : number of the orientation, eg 9
            bool full,          //in : true - [0, 2pi], false -> [0, pi]
            float clip )        //in : clip value, mag = (mag> clip?clip:mag)

{
  float *NormalizedValue, *gradientHist; const int hb=height/binSize, wb=width/binSize;
  // compute unnormalized gradient histograms
  gradientHist = (float*) wrCalloc(wb*hb*nOrients,sizeof(float));
  gradHist( Mag, Ori, gradientHist, height, width, binSize, nOrients, 1, full );
  // compute block normalization values
  NormalizedValue = hogNormMatrix( gradientHist, nOrients, hb, wb, binSize );
  // perform four normalizations per spatial block
  hogChannels( feature, gradientHist, NormalizedValue, hb, wb, nOrients, clip, 0 );
  wrFree(NormalizedValue); wrFree(gradientHist);
}


// compute FHOG features
void ssefhog( const float *Mag,     // in : magnitude
              const float *Ori,     // in : orientation
              float *feature,       // out: computed feature
              int height,           // in : height of the image
              int width,            // in : width of the image
              int binSize,          // in : binSize of the cell, eg 8
              int nOrients,         // in : number of orientation, eg 9
              float clip )          // in : clip value, if mag > clip, then mag = clip, eg 0.2
{
  const int hb=height/binSize, wb=width/binSize, nb=hb*wb, nbo=nb*nOrients;
  float *N, *R1, *R2; int o, x;
  // compute unnormalized constrast sensitive histograms
  // add binSize as the buffer for sse
  R1 = (float*) wrCalloc(wb*hb*nOrients*2+binSize,sizeof(float));
  gradHist( Mag, Ori, R1, height, width, binSize, nOrients*2, -1, true );
  // compute unnormalized contrast insensitive histograms
  R2 = (float*) wrCalloc(wb*hb*nOrients + binSize,sizeof(float));
  for( o=0; o<nOrients; o++ ) for( x=0; x<nb; x++ )
    R2[o*nb+x] = R1[o*nb+x]+R1[(o+nOrients)*nb+x];
  // compute block normalization values
  N = hogNormMatrix( R2, nOrients, hb, wb, binSize );
  // normalized histograms and texture channels
  hogChannels( feature+nbo*0, R1, N, hb, wb, nOrients*2, clip, 1 );
  hogChannels( feature+nbo*2, R2, N, hb, wb, nOrients*1, clip, 1 );
  hogChannels( feature+nbo*3, R1, N, hb, wb, nOrients*2, clip, 2 );

  wrFree(N); wrFree(R1); wrFree(R2);
}


// compute nOrients gradient histograms per bin x bin block of pixels
/*
 *  gradHist first quantize the orientation into nOrients bins, due to the interpolation,
 *  angle theta will contribute to two successive bins(Orientation0 and Orientation1) and they
 *  split the magnitude value( magnitude0 and magnitude1)
 *
 *  then put the gradient value into the spatial block accorrding to the diffierent method( hog or fhog)
 *  output size: height/binSize x width/binSize x nOrients
 *  momory continuous in row, like
 *
 *      G1
 *      G2
 *      G3
 *      ...
 *      Gn
*/
void gradHist( const float *Magnitude,  // in : magnitude, size width x height
               const float *Orientation,// in : oritentation, same size as magnitude
               float *gHist,            // out: gradHist,  size ( width/binSize x nOrients ) x height/binSize, big matrix
               int height,              // in : height
               int width,               // in : width
               int binSize,             // in : size of spatial bin, degree of aggregation,  eg : 4,
               int nOrients,            // in : number of orientation, eg : 6
               int softBin,             // in : softBin=1 -> hog, softBin=-1 -> fhog, softBin=0 -> channel feature
               bool full )              // in : true -> 0-2pi, false -> 0-pi
{
    const int height_block=height/binSize, width_block=width/binSize, h0=height_block*binSize, w0=width_block*binSize, nb=width_block*height_block;
    const float s=(float)binSize, sInv=1/s, sInv2=1/s/s;
    float *gHist0, *gHist1, *Magnitude0, *Magnitude1;
    int row_index, col_index; int *Orientation0, *Orientation1; float yb, init;

    Orientation0=(int*)alMalloc(width*sizeof(int),16); Magnitude0=(float*) alMalloc(width*sizeof(float),16);
    Orientation1=(int*)alMalloc(width*sizeof(int),16); Magnitude1=(float*) alMalloc(width*sizeof(float),16);

    // main loop
    for( row_index=0; row_index<h0; row_index++ )
    {
        // compute target orientation bins for entire row - very fast
        gradQuantize(Orientation+row_index*width,Magnitude+row_index*width,Orientation0,Orientation1,Magnitude0,Magnitude1,nb,w0,sInv2,nOrients,full,softBin>=0);
        if( softBin<0 && softBin%2==0 ) {
            // no interpolation w.r.t. either orienation or spatial bin
            gHist1=gHist+(row_index/binSize)*width_block;
#define GH gHist1[Orientation0[col_index]]+=Magnitude0[col_index]; col_index++;
            if( binSize==1 )      for(col_index=0; col_index<w0;) { GH; gHist1++; }
            else if( binSize==2 ) for(col_index=0; col_index<w0;) { GH; GH; gHist1++; }
            else if( binSize==3 ) for(col_index=0; col_index<w0;) { GH; GH; GH; gHist1++; }
            else if( binSize==4 ) for(col_index=0; col_index<w0;) { GH; GH; GH; GH; gHist1++; }
            else for( col_index=0; col_index<w0;) { for( int y1=0; y1<binSize; y1++ ) { GH; } gHist1++; }
#undef GH

        } else if( softBin%2==0 || binSize==1 ) { //channnel feature case
            // interpolate w.r.t. orientation only, not spatial bin
            gHist1=gHist+(row_index/binSize)*width_block;
#define GH gHist1[Orientation0[col_index]]+=Magnitude0[col_index]; gHist1[Orientation1[col_index]]+=Magnitude1[col_index]; col_index++;
            if( binSize==1 )      for(col_index=0; col_index<w0;) { GH; gHist1++; }
            else if( binSize==2 ) for(col_index=0; col_index<w0;) { GH; GH; gHist1++; }
            else if( binSize==3 ) for(col_index=0; col_index<w0;) { GH; GH; GH; gHist1++; }
            else if( binSize==4 ) for(col_index=0; col_index<w0;) { GH; GH; GH; GH; gHist1++; }
            else for( col_index=0; col_index<w0;) { for( int y1=0; y1<binSize; y1++ ) { GH; } gHist1++; }
#undef GH
        }else {
	      // interpolate using trilinear interpolation
	      float ms[4], xyd, xb, xd, yd; __m128 _m, _m0, _m1;
          bool hasTop, hasBot; int xb0, yb0;
          if( row_index==0 ) { init=(0+.5f)*sInv-0.5f; yb=init; }
          hasTop = yb>=0; yb0 = hasTop?(int)yb:-1; hasBot = yb0 < height_block-1;
          yd=yb-yb0; yb+=sInv; xb=init; col_index=0;

           // macros for code conciseness
          #define GHinit xd=xb-xb0; xb+=sInv; gHist0=gHist+yb0*width_block+xb0; xyd=xd*yd; \
           ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
          #define GH(H,ma,mb) gHist1=H; STRu(*gHist1,ADD(LDu(*gHist1),MUL(ma,mb)));

         // leading cols, no left bin
         for( ; col_index<binSize/2; col_index++ )
         {
           xb0=-1; GHinit;
           if(hasTop) { gHist0[Orientation0[col_index]+1]+=ms[2]*Magnitude0[col_index]; gHist0[Orientation1[col_index]+1]+=ms[2]*Magnitude1[col_index]; }
           if(hasBot) { gHist0[Orientation0[col_index]+width_block+1]+=ms[3]*Magnitude0[col_index]; gHist0[Orientation1[col_index]+width_block+1]+=ms[3]*Magnitude1[col_index]; }
         }
        
         // main cols, has left and right bins, use SSE for minor speedup
         if( softBin<0 ) for( ; ; col_index++ ) {      //fhog
           xb0 = (int) xb; if(xb0>=width_block-1) break; GHinit; _m0=SET(Magnitude0[col_index]);
           if(hasTop) { _m=SET(0,0,ms[2],ms[0]); GH(gHist0+Orientation0[col_index],_m,_m0); }
           if(hasBot) { _m=SET(0,0,ms[3],ms[1]); GH(gHist0+Orientation0[col_index]+width_block,_m,_m0);}

         } else for( ; ; col_index++ ) { // hog
           xb0 = (int) xb; if(xb0>=width_block-1) break; GHinit;
           _m0=SET(Magnitude0[col_index]); _m1=SET(Magnitude1[col_index]);
           if(hasTop) { _m=SET(0,0,ms[2],ms[0]);
             GH(gHist0+Orientation0[col_index],_m,_m0); GH(gHist0+Orientation1[col_index],_m,_m1); }
           if(hasBot) { _m=SET(0,0,ms[3],ms[1]);
             GH(gHist0+Orientation0[col_index]+width_block,_m,_m0); GH(gHist0+Orientation1[col_index]+width_block,_m,_m1); }
         }
        // final cols, no right bin
         for( ; col_index<w0; col_index++ ) {
           xb0 = (int) xb; GHinit;
           if(hasTop) { gHist0[Orientation0[col_index]]+=ms[0]*Magnitude0[col_index]; gHist0[Orientation1[col_index]]+=ms[0]*Magnitude1[col_index]; }
           if(hasBot) { gHist0[Orientation0[col_index]+width_block]+=ms[1]*Magnitude0[col_index]; gHist0[Orientation1[col_index]+width_block]+=ms[1]*Magnitude1[col_index]; }
         }
         #undef GHinit
         #undef GH
	    }
    }
    alFree(Orientation0); alFree(Orientation1); alFree(Magnitude0); alFree(Magnitude1);
    // normalize boundary bins which only get 7/8 of weight of interior bins
    if( softBin%2!=0 ) for( int o=0; o<nOrients; o++ ) {
        row_index=0; for( col_index=0; col_index<height_block; col_index++ ) gHist[o*nb+row_index+col_index*width_block]*=8.f/7.f;
        col_index=0; for( row_index=0; row_index<width_block; row_index++ ) gHist[o*nb+row_index+col_index*width_block]*=8.f/7.f;
        row_index=width_block-1; for( col_index=0; col_index<height_block; col_index++ ) gHist[o*nb+row_index+col_index*width_block]*=8.f/7.f;
        col_index=height_block-1; for( row_index=0; row_index<width_block; row_index++ ) gHist[o*nb+row_index+col_index*width_block]*=8.f/7.f;
    }
}


// helper for gradHist, quantize O and M into O0, O1 and M0, M1 (uses sse)
void gradQuantize( const float *O, const float *M, int *O0, int *O1, float *M0, float *M1,
                   int nb, int n, float norm, int nOrients, bool full, bool interpolate )
{
    // assumes all *OUTPUT* matrices are 4-byte aligned
    int i, o0, o1; float o, od, m;
    __m128i _o0, _o1, *_O0, *_O1; __m128 _o, _od, _m, *_M0, *_M1;
    // define useful constants
    const float oMult=(float)nOrients/(full?2*(float)PI:(float)PI); const int oMax=nOrients*nb;
    const __m128 _norm=SET(norm), _oMult=SET(oMult), _nbf=SET((float)nb);
    const __m128i _oMax=SET(oMax), _nb=SET(nb);
    // perform the majority of the work with sse
    _O0=(__m128i*) O0; _O1=(__m128i*) O1; _M0=(__m128*) M0; _M1=(__m128*) M1;
    if( interpolate ) for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(_o); _od=SUB(_o,CVT(_o0));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        _o1=ADD(_o0,_nb); _o1=AND(CMPGT(_oMax,_o1),_o1); *_O1++=_o1;
        _m=MUL(LDu(M[i]),_norm); *_M1=MUL(_od,_m); *_M0++=SUB(_m,*_M1); _M1++;
    } else for( i=0; i<=n-4; i+=4 ) {
        _o=MUL(LDu(O[i]),_oMult); _o0=CVT(ADD(_o,SET(.5f)));
        _o0=CVT(MUL(CVT(_o0),_nbf)); _o0=AND(CMPGT(_oMax,_o0),_o0); *_O0++=_o0;
        *_M0++=MUL(LDu(M[i]),_norm); *_M1++=SET(0.f); *_O1++=SET(0);
    }
    // compute trailing locations without sse
    if( interpolate ) for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) o; od=o-o0;
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        o1=o0+nb; if(o1==oMax) o1=0; O1[i]=o1;
        m=M[i]*norm; M1[i]=od*m; M0[i]=m-M1[i];
    } else for(; i<n; i++ ) {
        o=O[i]*oMult; o0=(int) (o+.5f);
        o0*=nb; if(o0>=oMax) o0=0; O0[i]=o0;
        M0[i]=M[i]*norm; M1[i]=0; O1[i]=0;
    }
}



// convolve one row of I by a [1 p 1] filter (uses SSE)
void convTri1Y( const float *I, float *O, int w, float p, int s ) {
#define C4(m,o) ADD(ADD(LDu(I[m*j-1+o]),MUL(p,LDu(I[m*j+o]))),LDu(I[m*j+1+o]))
    int j=0, k=((~((size_t) O) + 1) & 15)/4, h2=(w-1)/2;
    if( s==2 )
    {
        for( ; j<k; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        for( ; j<h2-4; j+=4 ) STR(O[j],_mm_shuffle_ps(C4(2,1),C4(2,5),136));
        for( ; j<h2; j++ ) O[j]=I[2*j]+p*I[2*j+1]+I[2*j+2];
        if( w%2==0 ) O[j]=I[2*j]+(1+p)*I[2*j+1];
    }
    else
    {
        O[j]=(1+p)*I[j]+I[j+1]; j++; if(k==0) k=(w<=4) ? w-1 : 4;
        for( ; j<k; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        for( ; j<w-4; j+=4 ) STR(O[j],C4(1,0));
        for( ; j<w-1; j++ ) O[j]=I[j-1]+p*I[j]+I[j+1];
        O[j]=I[j-1]+(1+p)*I[j];
    }
#undef C4
}

// convolve I by a [1 p 1] filter (uses SSE)
void convTri1( const float *I, float *O, int h, int w, int d, float p, int s) {
    const float nrm = 1.0f/((p+2)*(p+2)); int i, j, w0=w-(w%4);
    const float *Il, *Im, *Ir;
    float *T=(float*) alMalloc(w*sizeof(float),16);
    for( int d0=0; d0<d; d0++ )
        for( i=s/2; i<h; i+=s )
        {
            Il=Im=Ir=I+i*w+d0*h*w; if(i>0) Il-=w; if(i<h-1) Ir+=w;
            for( j=0; j<w0; j+=4 )
                STR(T[j],MUL(nrm,ADD(ADD(LDu(Il[j]),MUL(p,LDu(Im[j]))),LDu(Ir[j]))));
            for( j=w0; j<w; j++ ) T[j]=nrm*(Il[j]+p*Im[j]+Ir[j]);
            convTri1Y(T,O,w,p,s); O+=w/s;
        }
    alFree(T);
}


// compute x and y gradients for just one row (uses sse)
void grad1( const float *I,   //in :data
            float *Gx,  //out: gradient of x
            float *Gy,  //out: gradient of y
            int h,      //in : height
            int w,      //in : width
            int x )     //in : index of row
{
    int y, y1;
    const float *Ip, *In; float r; __m128 *_Ip, *_In, *_G, _r;
    //compute row of Gy
    Ip=I-w; In=I+w; r=.5f;
    if(x==0) { r=1; Ip+=w; } else if(x==h-1) { r=1; In-=w; }      //on the border
    if( w<4 || w%4>0 || (size_t(I)&15) || (size_t(Gy)&15) ) 
    {     //data align?
        int col_index = 0;
        while( col_index < w)
        {
            *Gy =(*In - *Ip)*r;
            Gy++;In++;Ip++;
            col_index++;
            if( col_index >=w)
                break;
        }
    } else {
        _G=(__m128*) Gy; _Ip=(__m128*) Ip; _In=(__m128*) In; _r = SET(r);
        for(int c=0; c<w; c+=4) *_G++=MUL(SUB(*_In++,*_Ip++),_r);
    }
    // compute row of Gx
#define GRADX(r) *Gx++=(*In++-*Ip++)*r;
    Ip=I; In=Ip+1;
    // equivalent --> GRADX(1); Ip--; for(y=1; y<w-1; y++) GRADX(.5f); In--; GRADX(1);
    y1=((~((size_t) Gx) + 1) & 15)/4; if(y1==0) y1=4; if(y1>w-1) y1=w-1;      // y1 -> the number of element with out using sse
    GRADX(1); Ip--; for(y=1; y<y1; y++) GRADX(.5f);
    _r = SET(.5f); _G=(__m128*) Gx;
    for(; y+4<w-1; y+=4, Ip+=4, In+=4, Gx+=4)
        *_G++=MUL(SUB(LDu(*In),LDu(*Ip)),_r);
    for(; y<w-1; y++) GRADX(.5f); In--; GRADX(1);
#undef GRADX
}

float* acosTable() {
    const int n=10000, b=10; int i;
    static float a[n*2+b*2]; static bool init=false;
    float *a1=a+n+b; if( init ) return a1;
    for( i=-n-b; i<-n; i++ )   a1[i]=(float)PI;
    for( i=-n; i<n; i++ )      a1[i]=float(acos(i/float(n)));
    for( i=n; i<n+b; i++ )     a1[i]=0;
    for( i=-n-b; i<n/10; i++ ) if( a1[i] > (float)PI-1e-6f ) a1[i]=(float)PI-1e-6f;
    init=true; return a1;
}

// compute gradient magnitude and orientation at each location (uses sse)
void gradMag( const float *I, float *M, float *O, int h, int w, int d, bool full )
{
    int x, y, y1, c, w4, s; float *Gx, *Gy, *M2; __m128 *_Gx, *_Gy, *_M2, _m;
    float *acost = acosTable(), acMult=10000.0f;

    // allocate memory for storing one row of output (padded so w4%4==0)
    w4=(w%4==0) ? w : w-(w%4)+4; s=d*w4*sizeof(float);
    M2=(float*) alMalloc(s,16); _M2=(__m128*) M2;
    Gx=(float*) alMalloc(s,16); _Gx=(__m128*) Gx;
    Gy=(float*) alMalloc(s,16); _Gy=(__m128*) Gy;

    // compute gradient magnitude and orientation for each row
    for( x=0; x<h; x++ )
    {
        // compute gradients (Gx, Gy) with maximum squared magnitude (M2)
        for(c=0; c<d; c++)          // compute for each channel, take the max value
        {
            grad1( I+x*w+c*w*h, Gx+c*w4, Gy+c*w4, h, w, x );
            for( y=0; y<w4/4; y++ )
            {
                y1=w4/4*c+y;
                _M2[y1]=ADD(MUL(_Gx[y1],_Gx[y1]),MUL(_Gy[y1],_Gy[y1]));
                if( c==0 ) continue; _m = CMPGT( _M2[y1], _M2[y] );
                _M2[y] = OR( AND(_m,_M2[y1]), ANDNOT(_m,_M2[y]) );
                _Gx[y] = OR( AND(_m,_Gx[y1]), ANDNOT(_m,_Gx[y]) );
                _Gy[y] = OR( AND(_m,_Gy[y1]), ANDNOT(_m,_Gy[y]) );
            }
        }
        // compute gradient mangitude (M) and normalize Gx // avoid the exception when arctan(Gy/Gx)
        for( y=0; y<w4/4; y++ ) {
            _m = SSEMIN( RCPSQRT(_M2[y]), SET(1e10f) );
            _M2[y] = RCP(_m);
            if(O) _Gx[y] = MUL( MUL(_Gx[y],_m), SET(acMult) );
            if(O) _Gx[y] = XOR( _Gx[y], AND(_Gy[y], SET(-0.f)) );
        };
        memcpy( M+x*w, M2, w*sizeof(float) );
        // compute and store gradient orientation (O) via table lookup
        if( O!=0 ) for( y=0; y<w; y++ ) O[x*w+y] = acost[(int)Gx[y]];
        if( O!=0 && full ) {
            y1=((~size_t(O+x*w)+1)&15)/4; y=0;
            for( ; y<y1; y++ ) O[y+x*w]+=(Gy[y]<0)*(float)PI;
            for( ; y<w-4; y+=4 ) STRu( O[y+x*w],
                    ADD( LDu(O[y+x*w]), AND(CMPLT(LDu(Gy[y]),SET(0.f)),SET((float)PI)) ) );
            for( ; y<w; y++ ) O[y+x*w]+=(Gy[y]<0)*(float)PI;
        }
    }

    alFree(Gx); alFree(Gy); alFree(M2);
}


// normalize gradient magnitude at each location (uses sse)
void gradMagNorm( float *M,                     // output: M = M/(S + norm)
                  const float *S,               // input : Source Matrix
                  int h, int w, float norm )    // input : parameters
{
    __m128 *_M, *_S, _norm; int i=0, n=h*w, n4=n/4;
    _S = (__m128*) S; _M = (__m128*) M; _norm = SET(norm);
    bool sse = !(size_t(M)&15) && !(size_t(S)&15);
    if(sse)
        for(; i<n4; i++)
        { *_M=MUL(*_M,RCP(ADD(*_S++,_norm))); _M++; }
    if(sse)
        i*=4;
    for(; i<n; i++) M[i] /= (S[i] + norm);
}

// convolve one row of I by a 2rx1 triangle filter
void convTriY( float *I, float *O, int w, int r, int s ) 
{
    r++; float t, u; int j, r0=r-1, r1=r+1, r2=2*w-r, h0=r+1, h1=w-r+1, h2=w;
    u=t=I[0]; for( j=1; j<r; j++ ) u+=t+=I[j]; u=2*u-t; t=0;
    if( s==1 ) {
        O[0]=u; j=1;
        for(; j<h0; j++) O[j] = u += t += I[r-j]  + I[r0+j] - 2*I[j-1];
        for(; j<h1; j++) O[j] = u += t += I[j-r1] + I[r0+j] - 2*I[j-1];
        for(; j<h2; j++) O[j] = u += t += I[j-r1] + I[r2-j] - 2*I[j-1];
    } else {
        int k=(s-1)/2; h2=(w/s)*s; if(h0>h2) h0=h2; if(h1>h2) h1=h2;
        if(++k==s) { k=0; *O++=u; } j=1;
        for(;j<h0;j++) { u+=t+=I[r-j] +I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h1;j++) { u+=t+=I[j-r1]+I[r0+j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
        for(;j<h2;j++) { u+=t+=I[j-r1]+I[r2-j]-2*I[j-1]; if(++k==s){ k=0; *O++=u; }}
    }
}

void convTri_sse( const float *I, float *O, int width, int height, int r,int d , int s) 
{
    r++; float nrm = 1.0f/(r*r*r*r); int i, j, k=(s-1)/2, h0, h1, w0;
    if(width%4==0)
        h0=h1=width;
    else
    { h0=width-(width%4); h1=h0+4; }
    w0=(height/s)*s;

    float *T=(float*) alMalloc(2*h1*sizeof(float),16), *U=T+h1;
    while( d-->0)
    {
        // initialize T and U
        for(j=0; j<h0; j+=4) STR(U[j], STR(T[j], LDu(I[j])));
        for(i=1; i<r; i++) for(j=0; j<h0; j+=4) INC(U[j],INC(T[j],LDu(I[j+i*width])));
        for(j=0; j<h0; j+=4) STR(U[j],MUL(nrm,(SUB(MUL(2,LD(U[j])),LD(T[j])))));
        for(j=0; j<h0; j+=4) STR(T[j],0);
        for(j=h0; j<width; j++ ) U[j]=T[j]=I[j];
        for(i=1; i<r; i++) for(j=h0; j<width; j++ ) U[j]+=T[j]+=I[j+i*width];
        for(j=h0; j<width; j++ ) { U[j] = nrm * (2*U[j]-T[j]); T[j]=0; }
        // prepare and convolve each column in turn
        k++; if(k==s) { k=0; convTriY(U,O,width,r-1,s); O+=width/s; }
        for( i=1; i<w0; i++ )
        {
            const float *Il=I+(i-1-r)*width; if(i<=r) Il=I+(r-i)*width; const float *Im=I+(i-1)*width;
            const float *Ir=I+(i-1+r)*width; if(i>height-r) Ir=I+(2*height-r-i)*width;
            for( j=0; j<h0; j+=4 ) {
                INC(T[j],ADD(LDu(Il[j]),LDu(Ir[j]),MUL(-2,LDu(Im[j]))));
                INC(U[j],MUL(nrm,LD(T[j])));
            }
            for( j=h0; j<width; j++ ) U[j]+=nrm*(T[j]+=Il[j]+Ir[j]-2*Im[j]);
            k++; if(k==s) { k=0; convTriY(U,O,width,r-1,s); O+=width/s; }
        }
        I+=width*height;
    }
    alFree(T);
}







