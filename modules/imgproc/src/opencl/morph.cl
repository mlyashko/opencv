//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Zero Lin, zero.lin@amd.com
//    Yao Wang, bitwangyaoyao@gmail.com
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#endif

#ifdef DEPTH_0
#define MIN_VAL 0
#define MAX_VAL UCHAR_MAX
#elif defined DEPTH_1
#define MIN_VAL SCHAR_MIN
#define MAX_VAL SCHAR_MAX
#elif defined DEPTH_2
#define MIN_VAL 0
#define MAX_VAL USHRT_MAX
#elif defined DEPTH_3
#define MIN_VAL SHRT_MIN
#define MAX_VAL SHRT_MAX
#elif defined DEPTH_4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif defined DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#ifdef OP_ERODE
#define VAL MAX_VAL
#elif defined OP_DILATE
#define VAL MIN_VAL
#else
#error "Unknown operation"
#endif

#ifdef OP_ERODE
#if defined INTEL_DEVICE && defined DEPTH_0
// workaround for bug in Intel HD graphics drivers (10.18.10.3496 or older)
#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
#define WA_CONVERT_1 CAT(convert_uint, cn)
#define WA_CONVERT_2 CAT(convert_, T)
#define convert_uint1 convert_uint
#define MORPH_OP(A, B) WA_CONVERT_2(min(WA_CONVERT_1(A), WA_CONVERT_1(B)))
#else
#define MORPH_OP(A, B) min((A), (B))
#endif
#endif
#ifdef OP_DILATE
#define MORPH_OP(A, B) max((A), (B))
#endif

#define PROCESS(y, x) \
    res = MORPH_OP(res, LDS_DAT[mad24(l_y + y, width, l_x + x)]);

// BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii
#define ELEM(i, l_edge, r_edge, elem1, elem2) (i) < (l_edge) | (i) >= (r_edge) ? (elem1) : (elem2)

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
#define EXTRA_PARAMS , __global const uchar * matptr, int mat_step, int mat_offset
#else
#define EXTRA_PARAMS
#endif

__kernel void morph(__global const uchar * srcptr, int src_step, int src_offset,
                    __global uchar * dstptr, int dst_step, int dst_offset,
                    int src_offset_x, int src_offset_y, int cols, int rows,
                    int src_whole_cols, int src_whole_rows EXTRA_PARAMS)
{
    int gidx = get_global_id(0), gidy = get_global_id(1);
    int l_x = get_local_id(0), l_y = get_local_id(1);
    int x = get_group_id(0) * LSIZE0, y = get_group_id(1) * LSIZE1;
    int start_x = x + src_offset_x - RADIUSX;
    int width = mad24(RADIUSX, 2, LSIZE0 + 1);
    int start_y = y + src_offset_y - RADIUSY;
    int point1 = mad24(l_y, LSIZE0, l_x);
    int point2 = point1 + LSIZE0 * LSIZE1;
    int tl_x = point1 % width, tl_y = point1 / width;
    int tl_x2 = point2 % width, tl_y2 = point2 / width;
    int cur_x = start_x + tl_x, cur_y = start_y + tl_y;
    int cur_x2 = start_x + tl_x2, cur_y2 = start_y + tl_y2;
    int start_addr = mad24(cur_y, src_step, cur_x * TSIZE);
    int start_addr2 = mad24(cur_y2, src_step, cur_x2 * TSIZE);

    __local T LDS_DAT[2 * LSIZE1 * LSIZE0];

    // read pixels from src
    int end_addr = mad24(src_whole_rows - 1, src_step, src_whole_cols * TSIZE);
    start_addr = start_addr < end_addr && start_addr > 0 ? start_addr : 0;
    start_addr2 = start_addr2 < end_addr && start_addr2 > 0 ? start_addr2 : 0;

    T temp0 = loadpix(srcptr + start_addr);
    T temp1 = loadpix(srcptr + start_addr2);

    // judge if read out of boundary
    temp0 = ELEM(cur_x, 0, src_whole_cols, (T)(VAL), temp0);
    temp0 = ELEM(cur_y, 0, src_whole_rows, (T)(VAL), temp0);

    temp1 = ELEM(cur_x2, 0, src_whole_cols, (T)(VAL), temp1);
    temp1 = ELEM(cur_y2, 0, src_whole_rows, (T)(VAL), temp1);

    LDS_DAT[point1] = temp0;
    LDS_DAT[point2] = temp1;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gidx < cols && gidy < rows)
    {
        T res = (T)(VAL);
        PROCESS_ELEMS;

        int dst_index = mad24(gidy, dst_step, mad24(gidx, TSIZE, dst_offset));

#if defined OP_GRADIENT || defined OP_TOPHAT || defined OP_BLACKHAT
        int mat_index =  mad24(gidy, mat_step, mad24(gidx, TSIZE, mat_offset));
        T value = loadpix(matptr + mat_index);

#ifdef OP_GRADIENT
        storepix(convertToT(convertToWT(res) - convertToWT(value)), dstptr + dst_index);
#elif defined OP_TOPHAT
        storepix(convertToT(convertToWT(value) - convertToWT(res)), dstptr + dst_index);
#elif defined OP_BLACKHAT
        storepix(convertToT(convertToWT(res) - convertToWT(value)), dstptr + dst_index);
#endif
#else // erode or dilate
        storepix(res, dstptr + dst_index);
#endif
    }
}

//this kernel works for odd kwidth only! and single-channel (8UC1) only
/*
__kernel void hor_vHGW(__global const uchar * srcptr, int src_step, int src_offset,
    __global uchar * dstptr, int dst_step, int dst_offset,
    __constant uchar * mat_kernel, int kwidth)
{
    //int x0 = get_group_id(0);
    //int y0 = get_group_id(1);
    int xgl = get_global_id(0);
    int ygl = get_global_id(1);
    //apron size, left and right from the tile
    int apron = (kwidth - 1) / 2;
    int xpix = xgl*kwidth - apron;
    int xmaxpix = xgl*kwidth + apron;

    //if xpix < 0 - we got the first window, and do need to compute less from left side
    //if xmaxpix > src.size().x - we got the last window, and do need to compute less from right side

    //special case for the first window

    //special case for the last window

    __global const int* ptr = (__global const int*)(srcptr + src_offset + src_step*ygl + xpix);

    __local int suf[2*kwidth-1];
//    int curVal;
    suf[kwidth - 1] = *(__global const int*)(ptr + (kwidth-1)*4);
    for (int i = kwidth - 2; i > 0; i--)
    {
        suf[i] = max(suf[i + 1], *(__global const int*)(ptr + i * 4));
        suf[2 * kwidth - i] = max(suf[2*kwidth-i+1], *(__global const int*)(ptr + (kwidth - i) * 4));
    }

}
*/
    /*
    __global__ void _horizontalVHGWKernel(const dataType *img, int imgStep, dataType *result,
                                    int resultStep, unsigned int width, unsigned int height,
                                        unsigned int size, rect2d borderSize)
    {
    const unsigned int step = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const unsigned int startx = __umul24(step, size);

    if (y >= height || startx > width)
        return;

    const dataType *lineIn = img + y*imgStep;
    dataType *lineOut = result + y*resultStep;
    const unsigned int center = startx + (size - 1);

    dataType minarray[512];
    minarray[size - 1] = lineIn[center];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for (k = 1; k<size; ++k) {
            nextMin = lineIn[center - k];
            minarray[size - 1 - k] = min(minarray[size - k], nextMin);

            nextMin = (center + k < width + size - 1) ? lineIn[center + k] : 255;
            minarray[size - 1 + k] = min(minarray[size + k - 2], nextMin);
        }
    }
    else {
        for (k = 1; k<size; ++k) {
            nextMin = lineIn[__umul24(center - k, imgStep)];
            minarray[size - 1 - k] = max(minarray[size - k], nextMin);

            nextMin = lineIn[__umul24(center + k, imgStep)];
            minarray[size - 1 + k] = max(minarray[size + k - 2], nextMin);
        }
    }

    int diff = width - startx;
    if (diff > 0) {
        lineOut += startx;
        lineOut[0] = minarray[0];

        for (k = 1; k < size - 1; ++k) {
            if (diff > k) {
                lineOut[k] = minMax<dataType, MOP>(minarray[k], minarray[k + size - 1]);
            }
        }

        if (diff > size - 1) {
            lineOut[size - 1] = minarray[2 * (size - 1)];
        }
    }
}
*/
}
