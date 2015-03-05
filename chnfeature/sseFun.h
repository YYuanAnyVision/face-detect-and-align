#ifndef SSEFUN_H
#define SSEFUN_h
#include <math.h>
#include "sse.hpp"

#define PI 3.14159265358979323846264338


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gradHist
 *  Description:  compute nOrients gradient histograms per bin x bin block of pixels
 * =====================================================================================
 */
void gradHist( const float *Mag,	// in : magnitude, size width x height
		       const float *Ori,	// in : oritentation, same size as magnitude
			   float *gHist,		// out: gradHist,  size ( width/binSize x nOrients ) x height/binSize, big matrix
			   int height,			// in : height
			   int width,			// in : width
			   int binSize,			// in : size of spatial bin, degree of aggregation,  eg : 4, 
			   int nOrients,		// in : number of orientation, eg : 6
               int softBin=0,		// in : softBin=1 -> hog, softBin=-1 -> fhog, softBin=0 -> channel feature
			   bool full=false);	// in : true -> 0-2pi, false -> 0-pi			



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTri1Y
 *  Description:  convolve one row of I by a [1 p 1] filter (uses SSE)
 * =====================================================================================
 */
void convTri1Y( const float *InputData,		// in : input data
				float *OutputData,			// out: output data
				int width,					// in : length of this row( width of the image )
				float p,					// in : 
				int s=1);					// in : resample factor, only 1 or 2 is supported


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTri1
 *  Description:   convolve I by a [1 p 1] filter (uses SSE)
 * =====================================================================================
 */
void convTri1( const float *InputData,		// in : input data
			   float *OutputData,			// out: output data
			   int height,					// in : the height of the image
			   int width,					// in : the width of the image
			   int dim,						// in : dim is 1 for single channel image, 3 for color image
			   float p,						// in : 
			   int s=1);					// in : resample factor, only 1 or 2 is supported


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  conTriY
 *  Description:  convolve one row of I by a 2rx1 triangle filter
 * =====================================================================================
 */
void convTriY( const float *InputData,  // in : input data
			   float *OouputData,       // out: output data
			   int width,               // in : the width of the image
			   int rad,                 // in : radius
			   int s=1);                // in : resample factor, only 1 or 2 is supported



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  convTri_sse
 *  Description:   convolve image of I by a 2rx1 triangle filter
 * =====================================================================================
 */

void convTri_sse( const float *InputData,       // in : input image's header
                  float *OutputData,            // out: output image's header, same size as input 
                  int width,                    // in : width of the image
                  int height,                   // in : height of the image
                  int r,                        // in : radius of the smooth kernel
                  int d = 1,                    // in : dimension of the input image, 1 for single channel, 3 for color image
                  int s=1 );                    // in : resample factor, only 1 or 2 is supported



/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  grad1
 *  Description:  compute x and y gradients for just one row (uses sse)
 * =====================================================================================
 */
void grad1( const float *Inputdata,         //in : input data
            float *Gx,                      //out: gradient of x
            float *Gy,                      //out: gradient of y
            int height,                     //in : height of the image
            int width,                      //in : width of the image
            int x );                        //in : index of row

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  acosTable
 *  Description:  return the acos lookup tabel, static 
 * =====================================================================================
 */
float* acosTable();


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gradMag
 *  Description:  compute gradient magnitude and orientation at each location (uses sse)
 * =====================================================================================
 */
void gradMag( const float *InputData,   // in : input image data,  width x height
              float *Mag,               //out : output magnitude, same size as InputData 
              float *Ori,               //out : output orientation, same size as InputData
              int height,               // in : height of the image
              int width,                // in : width of the image
              int dim,                  // in : dim of the image, 1 for gray image, 3 for color image.For color image, 
                                        //      the Mag and Ori will be the biggest one among all channels
              bool full );              // in : true for 0-2pi, false for 0-pi

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gradMaggradMagNorm
 *  Description:  normalize gradient magnitude at each location (uses sse) 
 *                M = M/(S + norm)
 * =====================================================================================
 */
void gradMagNorm( float *M,           // in&out: input Matrix
                  const float *S,     // in  : S, Smoothed matrix
                  int height,         // in  : height of the matrix
                  int width,          // in  : width of the matrix
                  float norm );       // in  : norm factor


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  gradQuantize
 *  Description:  helper for gradHist, quantize Ori and Maginto Ori0, Ori1 and Mag0, Mag1( interpolated) (uses sse)
 * =====================================================================================
 */
void gradQuantize( const float *Ori,        //in : orientation matrix
                   const float *Mag,        //in : magnitude matrix
                   int *Ori0,               //out: quantized orientation 1             
                   int *Ori1,               //out: quantized oritentation 2
                   float *Mag0,             //out: quantized magnitude 1
                   float *Mag1,             //out: quantized magnitude 2
                   int numberOfBlock,       //in : number of block
                   int numberOfElement,     //in : number of element of the row 
                   float norm,              //in : optional normlized value 
                   int nOrients,            //in : number of orientaton
                   bool full,               //in : ture for 0-2pi, false for 0-pi
                   bool interpolate );      //in : use interpolated or not


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  rgb2luv_setup
 *  Description:  Constants for rgb2luv conversion and lookup table for y-> l conversion
 *       return: the L value lookup table
 * =====================================================================================
 */
template<class oT> oT* rgb2luv_setup( oT z,     //in : base factor value, default 1, otherwise euqals the norm factor 
                                      oT *mr,   //out: red
                                      oT *mg,   //out: green
                                      oT *mb,   //out: blue
                                      oT &minu, //out: min value of the U channel
                                      oT &minv, //out: min value of the V channel
                                      oT &un,   //out: U channel shift
                                      oT &vn )  //out: V channel shift
{
    // set constants for conversion
    const oT y0=(oT) ((6.0/29)*(6.0/29)*(6.0/29));
    const oT a= (oT) ((29.0/3)*(29.0/3)*(29.0/3));
    un=(oT) 0.197833; vn=(oT) 0.468331;
    mr[0]=(oT) 0.430574*z; mr[1]=(oT) 0.222015*z; mr[2]=(oT) 0.020183*z;
    mg[0]=(oT) 0.341550*z; mg[1]=(oT) 0.706655*z; mg[2]=(oT) 0.129553*z;
    mb[0]=(oT) 0.178325*z; mb[1]=(oT) 0.071330*z; mb[2]=(oT) 0.939180*z;
    oT maxi=(oT) 1.0/270; minu=-88*maxi; minv=-134*maxi;
    // build (padded) lookup table for y->l conversion assuming y in [0,1]
    static oT lTable[1064]; static bool lInit=false;
    if( lInit ) return lTable; oT y, l;
    for(int i=0; i<1025; i++) {
        y = (oT) (i/1024.0);
        l = y>y0 ? 116*(oT)pow((double)y,1.0/3.0)-16 : y*a;
        lTable[i] = l*maxi;
    }
    for(int i=1025; i<1064; i++) lTable[i]=lTable[i-1];
    lInit = true; 
    return lTable;
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  rgb2luv
 *  Description:  Convert from rgb to luv
 * =====================================================================================
 */
template<class iT, class oT> void rgb2luv( const iT *InputData,     //in : inputData
                                           oT *OutputData,          //out: outputData
                                           int n,                   //in : number of the element
                                           oT nrm )                 //in : normlize value,( output factor)
{
    oT minu, minv, un, vn, mr[3], mg[3], mb[3];
    oT *lTable = rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn);
    oT *L=OutputData, *U=L+n, *V=U+n;
    const iT *R=InputData+2, *G=InputData+1, *B=InputData;			// opencv , B,G,R,B,G,R..
    for( int i=0; i<n; i++ )
    {
        oT r, g, b, x, y, z, l;
        r=(oT)*R; R=R+3;
        g=(oT)*G; G=G+3;
        b=(oT)*B; B=B+3;
        x = mr[0]*r + mg[0]*g + mb[0]*b;
        y = mr[1]*r + mg[1]*g + mb[1]*b;
        z = mr[2]*r + mg[2]*g + mb[2]*b;
        l = lTable[(int)(y*1024)];
        *(L++) = l; z = 1/(x + 15*y + 3*z + (oT)1e-35);
        *(U++) = l * (13*4*x*z - 13*un) - minu;
        *(V++) = l * (13*9*y*z - 13*vn) - minv;
    }
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  rgb2luv_sse
 *  Description:  Convert from rgb to luv using sse
 * =====================================================================================
 */
template<class iT> void rgb2luv_sse( const iT *I,   // in : input_image's header
                                     float *J,      //out : output image's header
                                     int n,         // in : number of the elements
                                     float nrm )    // in : normlized value( scale factor )
{
    const int k=256; float R[k], G[k], B[k];
    if( (size_t(R)&15||size_t(G)&15||size_t(B)&15||size_t(I)&15||size_t(J)&15)
            || n%4>0 )
    {
        rgb2luv(I,J,n,nrm); return;
    }                      // data not align
    int i=0, i1, n1; float minu, minv, un, vn, mr[3], mg[3], mb[3];
    float *lTable = rgb2luv_setup(nrm,mr,mg,mb,minu,minv,un,vn);
    while( i<n )
    {
        n1 = i+k; if(n1>n) n1=n; float *J1=J+i; float *R1, *G1, *B1;
        /* ------------ RGB is now RRRRRGGGGGBBBBB ----------*/
        // convert to floats (and load input into cache)
        R1=R; G1=G; B1=B;
        const iT *Bi=I+i*3, *Gi=Bi+1, *Ri=Bi+2;
        for( i1=0; i1<(n1-i); i1++ )
        {
            R1[i1] = (float) (*Ri);Ri = Ri+3;
            G1[i1] = (float) (*Gi);Gi = Gi+3;
            B1[i1] = (float) (*Bi);Bi = Bi+3;
        }
        /* ------------ RGB is now RRRRRGGGGGBBBBB ----------*/
        // compute RGB -> XYZ
        for( int j=0; j<3; j++ )
        {
            __m128 _mr, _mg, _mb, *_J=(__m128*) (J1+j*n);
            __m128 *_R=(__m128*) R1, *_G=(__m128*) G1, *_B=(__m128*) B1;
            _mr=SET(mr[j]); _mg=SET(mg[j]); _mb=SET(mb[j]);
            for( i1=i; i1<n1; i1+=4 )
            {
                *(_J++) = ADD( ADD(MUL(*(_R++),_mr),MUL(*(_G++),_mg)),MUL(*(_B++),_mb));
            }
        }
        /* ---------------XXXXXXXYYYYYYYZZZZZZZZ now --------------- */

        { // compute XZY -> LUV (without doing L lookup/normalization)
            __m128 _c15, _c3, _cEps, _c52, _c117, _c1024, _cun, _cvn;
            _c15=SET(15.0f); _c3=SET(3.0f); _cEps=SET(1e-35f);
            _c52=SET(52.0f); _c117=SET(117.0f), _c1024=SET(1024.0f);
            _cun=SET(13*un); _cvn=SET(13*vn);
            __m128 *_X, *_Y, *_Z, _x, _y, _z;
            _X=(__m128*) J1; _Y=(__m128*) (J1+n); _Z=(__m128*) (J1+2*n);
            for( i1=i; i1<n1; i1+=4 )
            {
                _x = *_X; _y=*_Y; _z=*_Z;
                _z = RCP(ADD(_x,ADD(_cEps,ADD(MUL(_c15,_y),MUL(_c3,_z)))));
                *(_X++) = MUL(_c1024,_y);
                *(_Y++) = SUB(MUL(MUL(_c52,_x),_z),_cun);
                *(_Z++) = SUB(MUL(MUL(_c117,_y),_z),_cvn);
            }
        }
        { // perform lookup for L and finalize computation of U and V
            for( i1=i; i1<n1; i1++ ) J[i1] = lTable[(int)J[i1]];
            __m128 *_L, *_U, *_V, _l, _cminu, _cminv;
            _L=(__m128*) J1; _U=(__m128*) (J1+n); _V=(__m128*) (J1+2*n);
            _cminu=SET(minu); _cminv=SET(minv);
            for( i1=i; i1<n1; i1+=4 ) {
                _l = *(_L++);
                *_U = SUB(MUL(_l,*_U),_cminu); _U++;
                *_V = SUB(MUL(_l,*_V),_cminv); _V++;
            }
        }
        i = n1;
    }
}


/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  ssefhog
 *  Description: compute FHOG features giving magnitude and oritentation, using sse
 * =====================================================================================
 */
void ssefhog( const float *Mag,     // in : magnitude
              const float *Ori,     // in : orientation
              float *feature,       // out: computed feature
              int height,           // in : height of the image
              int width,            // in : width of the image
              int binSize,          // in : binSize of the cell, eg 8
              int nOrients,         // in : number of orientation, eg 9
              float clip );         // in : clip value, if mag > clip, then mag = clip, eg 0.2

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  ssehog
 *  Description:  compute the hog features;
 * =====================================================================================
 */
void ssehog(const float *Mag,   //in : magnitude
            const float *Ori,   //in : orientation
            float *feature,     //out: computed feature
            int height,          //in : height of the image
            int width,          //in : width of the image
            int binSize,        //in : binSize of the cell, eg 8
            int nOrients,       //in : number of the orientation, eg 9
            bool full,          //in : true - [0, 2pi], false -> [0, pi]
            float clip );       //in : clip value, mag = (mag> clip?clip:mag)

#endif
