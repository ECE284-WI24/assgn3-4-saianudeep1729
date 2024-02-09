#include <stdint.h>
#include <vector>
#include <string>
#include <assert.h>
#include "timer.hpp"
#include "twoBitCompressor.hpp"

void printGpuProperties();

namespace GpuReadMapper {
    struct ReferenceArrays {
        uint32_t refLen;
        uint32_t* d_compressedRef;
        uint32_t* d_kmerOffset;
        size_t* d_kmerPos;

        void allocateReferenceArrays (char* ref, uint32_t refLen, uint32_t kmerSize);
        void clearAll();
    };

    struct ReadBatch {
        uint32_t readLen;
        uint32_t batchSize;
        std::vector<std::string> readDesc;
        uint32_t* compressedReadBatch;
        uint32_t* d_compressedReadBatch;
        uint32_t* mappingScores;
        uint32_t* mappingStartCoords;
        uint32_t* mappingEndCoords;
        uint32_t* d_mappingScores;
        uint32_t* d_mappingStartCoords;
        uint32_t* d_mappingEndCoords;

        ReadBatch(size_t len, size_t bs);

        void addRead(char* desc, size_t descLen, char* seq, size_t seqLen) {
            size_t currNumReads = readDesc.size();
            assert (currNumReads < batchSize);
            readDesc.push_back(std::string(desc, descLen));

            size_t compressedSeqLen = (seqLen+15)/16;
            size_t startOffset = currNumReads*compressedSeqLen;

            for (size_t i=0; i < compressedSeqLen; i++) {
                compressedReadBatch[startOffset+i] = 0;

                size_t start = 16*i;
                size_t end = std::min(seqLen, start+16);

                uint32_t twoBitVal = 0;
                uint32_t shift = 0;
                for (auto j=start; j<end; j++) {
                    switch(seq[j]) {
                        case 'A':
                            twoBitVal = 0;
                            break;
                        case 'C':
                            twoBitVal = 1;
                            break;
                        case 'G':
                            twoBitVal = 2;
                            break;
                        case 'T':
                            twoBitVal = 3;
                            break;
                        default:
                            twoBitVal = 0;
                            break;
                    }

                    compressedReadBatch[startOffset+i] |= (twoBitVal << shift);
                    shift += 2;
                }
            }

        }
    };

    static bool headerPrinted = false;
    void transferReadBatch (ReadBatch* readBatch);
    void mapReadBatch(ReferenceArrays referenceArrays, ReadBatch* readBatch, uint32_t kmerSize, uint32_t kmerWindow);
    void printReadBatch(ReadBatch* readBatch);
    void clearReadBatch(ReadBatch* readBatch);


    void clearReferenceArrays(ReferenceArrays referenceArrays);

    void seedTableOnGpu (uint32_t* compressedSeq,
            uint32_t seqLen,
            uint32_t kmerSize,
            uint32_t* kmerOffset,
            size_t* kmerPos);

}
