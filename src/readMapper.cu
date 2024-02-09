#include "readMapper.cuh"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/**
 * Prints information for each available GPU device on stdout
 */
void printGpuProperties () {
    int nDevices;

    // Store the number of available GPU device in nDevicess
    cudaError_t err = cudaGetDeviceCount(&nDevices);

    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaGetDeviceCount failed!\n");
        exit(1);
    }

    // For each GPU device found, print the information (memory, bandwidth etc.)
    // about the device
    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Device memory: %lu\n", prop.totalGlobalMem);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }
}


GpuReadMapper::ReadBatch::ReadBatch(uint64_t len, uint64_t bs) {
    readLen = len;
    batchSize = bs;
    readDesc.reserve(batchSize);

    uint64_t compressedReadLen = (readLen+15)/16;
    compressedReadBatch = new uint32_t[compressedReadLen*batchSize];
    
    cudaError_t err;
    err = cudaMalloc(&d_compressedReadBatch, compressedReadLen*batchSize*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    mappingScores = new uint32_t[batchSize];
    mappingStartCoords = new uint32_t[batchSize];
    mappingEndCoords = new uint32_t[batchSize];
    
    err = cudaMalloc(&d_mappingScores, batchSize*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    err = cudaMalloc(&d_mappingStartCoords, batchSize*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    err = cudaMalloc(&d_mappingEndCoords, batchSize*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
}

void GpuReadMapper::transferReadBatch(GpuReadMapper::ReadBatch* readBatch) {
    cudaError_t err;

    uint64_t numReads = readBatch->readDesc.size();
    uint64_t compressedReadLen = (readBatch->readLen+15)/16;
    err = cudaMemcpy(readBatch->d_compressedReadBatch, readBatch->compressedReadBatch, compressedReadLen*numReads*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }
}

/* 
 * Print mapping scores and coordinates for each read in the batch
 *
 * ASSIGNMENT 4 TASK: Ensure print is atomic
 *
 * HINT: consider using a tbb::mutex and defining it in readMapper.cuh 
 * BONUS: Also maintain the relative order of batches during the printing 
 */
void GpuReadMapper::printReadBatch(GpuReadMapper::ReadBatch* readBatch) {
    if (!GpuReadMapper::headerPrinted) {
        printf("READ_ID\tMAP_SCORE\tSTART_COORD\tEND_COORD\n");
        GpuReadMapper::headerPrinted = true;
    }
    for (uint64_t i=0; i < readBatch->readDesc.size(); i++) {
        printf("%s\t%u\t%u\t%u\n", readBatch->readDesc[i].c_str(), readBatch->mappingScores[i],
                readBatch->mappingStartCoords[i], readBatch->mappingEndCoords[i]);
    }
}

/**
 * Free allocated host and GPU device memory for the different arrays
 */
void GpuReadMapper::clearReadBatch(GpuReadMapper::ReadBatch* readBatch) {
    readBatch->readDesc.clear();
    delete[] readBatch->compressedReadBatch;
    delete[] readBatch->mappingScores;
    delete[] readBatch->mappingStartCoords;
    delete[] readBatch->mappingEndCoords;
    
    cudaFree(readBatch->d_compressedReadBatch);
    cudaFree(readBatch->d_mappingScores);
    cudaFree(readBatch->d_mappingStartCoords);
    cudaFree(readBatch->d_mappingEndCoords);

    delete readBatch;
}


/**
 * Allocates arrays on the GPU device for (i) storing the compressed reference
 * (ii) kmer offsets of the seed table (iii) kmer positions of the seed table
 * Size of the arrays depends on the input reference length and kmer size
 */
void GpuReadMapper::ReferenceArrays::allocateReferenceArrays(char* r, uint32_t rLen, uint32_t kmerSize) {
    refLen = rLen;

    // Compute the 2bit-compressed sequence for the reference
    uint32_t twoBitCompressedSize = (refLen+15)/16;
    uint32_t * twoBitCompressed = new uint32_t[twoBitCompressedSize];
    twoBitCompress(r, rLen, twoBitCompressed);
    
    uint32_t maxKmers = (uint32_t) pow(4,kmerSize)+1;

    cudaError_t err;
    // Only (1)allocate and (2)transfer the 2-bit compressed sequence to GPU.
    // This reduces the memory transfer and storage overheads
    // 1. Allocate memory
    err = cudaMalloc(&d_compressedRef, twoBitCompressedSize*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // 2. Transfer compressed sequence
    err = cudaMemcpy(d_compressedRef, twoBitCompressed, twoBitCompressedSize*sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    // Allocate memory on GPU device for storing the kmer offset array
    err = cudaMalloc(&d_kmerOffset, maxKmers*sizeof(uint32_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }

    // Allocate memory on GPU device for storing the kmer position array
    // Each element is uint64_t (64-bit) because an intermediate step uses the
    // first 32-bits for kmer value and the last 32-bits for kmer positions
    err = cudaMalloc(&d_kmerPos, (rLen-kmerSize+1)*sizeof(uint64_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMalloc failed!\n");
        exit(1);
    }
    cudaDeviceSynchronize();
    
    seedTableOnGpu(d_compressedRef, rLen, kmerSize, d_kmerOffset, d_kmerPos);
    
    uint64_t numValues = 10;
    uint32_t* kmerOffset = new uint32_t[numValues];
    uint64_t* kmerPos = new uint64_t[numValues];

    err = cudaMemcpy(kmerOffset, d_kmerOffset, numValues*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    err = cudaMemcpy(kmerPos, d_kmerPos, numValues*sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    printf("i\tkmerOffset[i]\tkmerPos[i]\n");
    for (int i=0; i<numValues; i++) {
        printf("%i\t%u\t%zu\n", i, kmerOffset[i], kmerPos[i]);
    }

    delete[] kmerOffset;
    delete[] kmerPos;

    delete[] twoBitCompressed;
}

/**
 * Free allocated GPU device memory for different arrays
 */
void GpuReadMapper::ReferenceArrays::clearAll() {
    cudaFree(d_compressedRef);
    cudaFree(d_kmerOffset);
    cudaFree(d_kmerPos);
}

void GpuReadMapper::clearReferenceArrays(GpuReadMapper::ReferenceArrays referenceArrays) {
    referenceArrays.clearAll();
}

/*
 * Finds kmers for the compressed sequence creates an array with elements
 * containing the 64-bit concatenated value consisting of the kmer value in the
 * first 32 bits and the kmer position in the last 32 bits. The values are
 * stored in the arrary kmerPos, with i-th element corresponding to the i-th
 * kmer in the sequence
 *
 * ASSIGNMENT 3 BONUS: parallelize this function using the same strategy as
 * ASSIGNMENT 2
 * */
__global__ void kmerPosConcat(
    uint32_t* d_compressedSeq,
    uint32_t d_seqLen,
    uint32_t kmerSize,
    uint64_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t N = d_seqLen;
    uint32_t compressedSeqLen = (N+15)/16;
    uint32_t k = kmerSize;

    // Helps mask the non kmer bits from compressed sequence. E.g. for k=2,
    // mask=0x1111 and for k=3, mask=0x111111
    uint32_t mask = (1 << 2*k)-1;
    uint64_t kmer = 0;

    if ((bx == 0) && (tx == 0)) {
        for (uint32_t i = 0; i <= N-k; i++) {
            uint32_t index = i/16;
            uint32_t shift1 = 2*(i%16);
            uint64_t val1 = d_compressedSeq[index];
            uint64_t val2 = (index+1 < compressedSeqLen) ? d_compressedSeq[index+1] : 0;
            if (shift1 > 0) {
                uint32_t shift2 = (32-shift1);
                kmer = ((val1 >> shift1) | (val2 << shift2)) & mask;
            } else {
                kmer = val1 & mask;
            }

            uint64_t kPosConcat = (kmer << 32) + i;
            d_kmerPos[i] = kPosConcat;
        }
    }
}

/**
 * Generates the kmerOffset array using the sorted kmerPos array consisting of
 * the kmer and positions. Requires iterating through the kmerPos array and
 * finding indexes where the kmer values change, depending on which the
 * kmerOffset values are determined.
 *
 * ASSIGNMENT 3 BONUS: parallelize this function using the same strategy as
 * ASSIGNMENT 2
 */
__global__ void kmerOffsetFill(
    uint32_t seqLen,
    uint32_t kmerSize,
    uint32_t numKmers,
    uint32_t* d_kmerOffset,
    uint64_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t N = seqLen;
    uint32_t k = kmerSize;

    uint64_t mask = ((uint64_t) 1 << 32)-1;
    uint32_t kmer = 0;
    uint32_t lastKmer = 0;

    if ((bx == 0) && (tx == 0)) {
        for (uint32_t i = 0; i <= N-k; i++) {
            kmer = (d_kmerPos[i] >> 32) & mask;
            if (kmer != lastKmer) {
                for (auto j=lastKmer; j<kmer; j++) {
                    d_kmerOffset[j] = i;
                }
            }
            lastKmer = kmer;
        }

        // For all kmers lexicographically larger than the lexicographically
        // largest kmer in the sequence, set offset to N-k
        for (auto j=lastKmer; j<numKmers; j++) {
            d_kmerOffset[j] = N-k;
        }
    }
}

/**
 * Masks the first 32 bits of the elements in the kmerPos array
 *
 */
__global__ void kmerPosMask(
    uint32_t seqLen,
    uint32_t kmerSize,
    uint64_t* d_kmerPos) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint32_t N = seqLen;
    uint32_t k = kmerSize;

    uint64_t mask = ((uint64_t) 1 << 32)-1;
    if ((bx == 0) && (tx == 0)) {
        for (uint32_t i = 0; i <= N-k; i++) {
            d_kmerPos[i] = d_kmerPos[i] & mask;
        }
    }
}

/**
 * Constructs seed table, consisting of kmerOffset and kmerPos arrrays
 * on the GPU.
*/
void GpuReadMapper::seedTableOnGpu (
    uint32_t* compressedSeq,
    uint32_t seqLen,

    uint32_t kmerSize,

    uint32_t* kmerOffset,
    uint64_t* kmerPos) {

    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    kmerPosConcat<<<numBlocks, blockSize>>>(compressedSeq, seqLen, kmerSize, kmerPos);

    // Parallel sort the kmerPos array on the GPU device using the thrust
    // library (https://thrust.github.io/)
    thrust::device_ptr<uint64_t> kmerPosPtr(kmerPos);
    thrust::sort(kmerPosPtr, kmerPosPtr+seqLen-kmerSize+1);

    uint32_t numKmers = pow(4, kmerSize);;
    kmerOffsetFill<<<numBlocks, blockSize>>>(seqLen, kmerSize, numKmers, kmerOffset, kmerPos);
    kmerPosMask<<<numBlocks, blockSize>>>(seqLen, kmerSize, kmerPos);

    // Wait for all computation on GPU device to finish. Needed to ensure
    // correct runtime profiling results for this function.
    cudaDeviceSynchronize();
}

/* 
 * Main kernel for mapping reads to reference
 *
 * ASSIGNMENT 3 TASKS: Efficiently parallelize this function using the following
 * strategies:
 * 
 *  1. Multiple threads of a thread block map one read at a time
 *  2. Coalescing memory accesses for reading read and reference sequences
 *  3. Parallel reduction for finding the minimizer seed in a window 
 *  4. Parallel prefix scan for finding the mappingScore in a query block
 *  5. Exploit reuse of sequences using shared memory
 *  6. Any additional optimization you can think of that improves performance
 *  7. You may assume a read length (readSize) of 256
 * */
__global__ void readMapper(
        uint64_t batchSize,
        uint32_t readLen,
        uint32_t* d_compressedReadBatch,
        uint32_t* d_kmerOffset,
        uint64_t* d_kmerPos,
        uint32_t kmerSize,
        uint32_t kmerWindow,
        uint32_t* d_compressedRef,
        uint32_t refLen,
        uint32_t* d_mappingScores,
        uint32_t* d_mappingStartCoords,
        uint32_t* d_mappingEndCoords) {

    int tx = threadIdx.x;
    int bx = blockIdx.x;

    __shared__ uint64_t windowKmerPos[32];
    uint32_t kmerMask = (1 << 2*kmerSize) - 1;
    uint32_t posMask = (1 << 30) - 1;
    uint32_t twoBitMask = 3;
    uint64_t lastKmerPos = 16;
    uint64_t currKmerPos = 0;

    uint32_t compressedReadLen = (readLen+15)/16;
    if ((bx == 0) && (tx == 0)) {
        for (uint64_t readNum=0; readNum<batchSize; readNum++) {
            uint32_t startAddress = readNum*compressedReadLen;
            uint32_t windowIdx = 0;
            uint32_t bestMappingScore = 0;
            uint32_t bestMappingStartCoords = 0;
            uint32_t bestMappingEndCoords = 0;
    
            uint32_t numKmers = readLen-kmerSize+1;
            
            for (uint32_t i=0; i<numKmers; i++) {
                uint32_t index = i/16;
                uint32_t shift1 = 2*(i%16);
                uint64_t val1 = d_compressedReadBatch[startAddress+index];
                uint64_t val2 = (index+1 < compressedReadLen) ? d_compressedReadBatch[startAddress+index+1] : 0;
                if (shift1 > 0) {
                    uint32_t shift2 = (32-shift1);
                    currKmerPos = ((val1 >> shift1) + (val2 << shift2)) & kmerMask;
                } else {
                    currKmerPos = val1 & kmerMask;
                }
                currKmerPos = (currKmerPos << 32) + i;
                windowKmerPos[windowIdx] = currKmerPos;
                windowIdx = (windowIdx+1)%kmerWindow;

                if (i >= kmerWindow-1) {
                    for (uint32_t w=0; w<kmerWindow; w++) {
                        if (windowKmerPos[w] < currKmerPos) {
                            currKmerPos = windowKmerPos[w];
                        }
                    }

                    if (currKmerPos != lastKmerPos) {
                        uint32_t kmer = (currKmerPos >> 32) & kmerMask; 
                        uint32_t pos = currKmerPos & posMask;
                        uint32_t e = d_kmerOffset[kmer];
                        uint32_t s = 0;
                        if (kmer > 0) {
                            s = d_kmerOffset[kmer-1];
                        }

                        for (uint32_t p=s; p<e; p++) {
                            uint32_t hitPos = d_kmerPos[p];
                            uint32_t refStart=0, qStart=0;
                            uint32_t alignLen = readLen;
                                
                            if (hitPos > pos) {
                                refStart = hitPos-pos;
                            }
                            else {
                                qStart = pos-hitPos;
                            }
                            if (refStart+readLen > refLen) {
                                alignLen = refLen-refStart;
                            }

                            if (qStart > 0) {
                                alignLen = min(alignLen , readLen-qStart);
                            }

                            uint32_t mappingScore=0;
                            for (uint32_t j=0; j < alignLen; j++) {
                                uint32_t rIndex = (refStart+j)/16;
                                uint32_t rShift = 2*((refStart+j)%16);
                                uint32_t qIndex = (qStart+j)/16;
                                uint32_t qShift = 2*((qStart+j)%16);

                                if (((d_compressedRef[rIndex] >> rShift) & twoBitMask) == 
                                    ((d_compressedReadBatch[startAddress+qIndex] >> qShift) & twoBitMask)) {
                                    mappingScore += 1;
                                }
                            }

                            if (mappingScore > bestMappingScore) {
                                bestMappingScore = mappingScore;
                                bestMappingStartCoords = refStart; 
                                bestMappingEndCoords = refStart+alignLen; 
                            }
                        }
                    }

                    lastKmerPos = currKmerPos;
                }

            }

            d_mappingScores[readNum] =  bestMappingScore;
            d_mappingStartCoords[readNum] =  bestMappingStartCoords;
            d_mappingEndCoords[readNum] =  bestMappingEndCoords;
        }
    }
}


/**
 * Accepts (i) referenceArrays containing the reference sequence and its seed
 * table, (ii) a batch of reads, (iii) minimizer seed parameters (k-mer size and
 * minimizer window) and maps each read to a single region in the reference
 * where the match score around the seed hit is the highest 
 * */
void GpuReadMapper::mapReadBatch (
        GpuReadMapper::ReferenceArrays referenceArrays,
        GpuReadMapper::ReadBatch* readBatch,
        uint32_t kmerSize,
        uint32_t kmerWindow) {

    // 
    int numBlocks = 1; // i.e. number of thread blocks on the GPU
    int blockSize = 1; // i.e. number of GPU threads per thread block

    uint64_t numReads = readBatch->readDesc.size();

    readMapper<<<numBlocks, blockSize>>>(numReads, readBatch->readLen, readBatch->d_compressedReadBatch, 
            referenceArrays.d_kmerOffset, referenceArrays.d_kmerPos, kmerSize, kmerWindow, 
            referenceArrays.d_compressedRef, referenceArrays.refLen,
            readBatch->d_mappingScores, readBatch->d_mappingStartCoords, readBatch->d_mappingEndCoords); 
    
    cudaError_t err;
    
    err = cudaMemcpy(readBatch->mappingScores, readBatch->d_mappingScores, numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }
    
    err = cudaMemcpy(readBatch->mappingStartCoords, readBatch->d_mappingStartCoords, numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }
    
    err = cudaMemcpy(readBatch->mappingEndCoords, readBatch->d_mappingEndCoords, numReads*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU_ERROR: cudaMemCpy failed!\n");
        exit(1);
    }

    cudaDeviceSynchronize();
}


