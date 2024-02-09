#include <iostream>
#include <vector>
#include <boost/program_options.hpp>
#include <tbb/task_scheduler_init.h>
#include "readMapper.cuh"
#include "twoBitCompressor.hpp"
#include "kseq.h"
#include "zlib.h"

// For parsing the command line values
namespace po = boost::program_options;

// For reading in the FASTA file
KSEQ_INIT2(, gzFile, gzread)

int main(int argc, char** argv) {
    // Timer below helps with the performance profiling (see timer.hpp for more
    // details)
    Timer timer;

    std::string refFilename;
    std::string readsFilename;
    uint32_t readSize;
    uint32_t kmerSize;
    uint32_t kmerWindow;
    uint32_t maxReads;
    uint32_t batchSize;
    uint32_t numThreads;

    // Parse the command line options
    po::options_description desc{"Options"};
    desc.add_options()
    ("reference,r", po::value<std::string>(&refFilename)->required(), "Input reference sequence in FASTA file format [REQUIRED].")
    ("reads,q", po::value<std::string>(&readsFilename)->required(), "Input read sequences in FASTA file format [REQUIRED].")
    ("readSize,s", po::value<uint32_t>(&readSize)->default_value(256), "Size of the read in number of bases [DO NOT CHANGE THE DEFAULT OF 256!]")
    ("kmerSize,k", po::value<uint32_t>(&kmerSize)->default_value(14), "Minimizer seed size (range: 2-15)")
    ("kmerWindow,w", po::value<uint32_t>(&kmerWindow)->default_value(16), "Minimizer window size (range: 1-32)")
    ("maxReads,N", po::value<uint32_t>(&maxReads)->default_value(1e6), "Maximum number of reads to read from the input read sequence file")
    ("batchSize,b", po::value<uint32_t>(&batchSize)->default_value(32), "Number of reads in a batch")
    ("numThreads,T", po::value<uint32_t>(&numThreads)->default_value(4), "Number of Threads (range: 1-8)")
    ("help,h", "Print help messages");

    po::options_description allOptions;
    allOptions.add(desc);

    po::variables_map vm;
    try {
        po::store(po::command_line_parser(argc, argv).options(allOptions).run(), vm);
        po::notify(vm);
    } catch(std::exception &e) {
        std::cerr << desc << std::endl;
        exit(1);
    }

    // Check input values
    if ((kmerSize < 2) || (kmerSize > 15)) {
        std::cerr << "ERROR! kmerSize should be between 2 and 15." << std::endl;
        exit(1);
    }
    if ((kmerWindow < 1) || (kmerWindow > 32)) {
        std::cerr << "ERROR! kmerWindow should be between 1 and 64." << std::endl;
        exit(1);
    }
    if ((numThreads < 1) || (numThreads > 8)) {
        std::cerr << "ERROR! numThreads should be between 1 and 8." << std::endl;
        exit(1);
    }

    // Print GPU information
    timer.Start();
    fprintf(stdout, "Setting CPU threads to %u and printing GPU device properties.\n", numThreads);
    tbb::task_scheduler_init init(numThreads);
    printGpuProperties();
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Read reference sequence as kseq_t object
    timer.Start();
    fprintf(stdout, "Reading reference sequence and compressing to two-bit encoding.\n");
    gzFile fp = gzopen(refFilename.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", refFilename.c_str());
        exit(1);
    }
    kseq_t *record = kseq_init(fp);
    int n;
    if ((n = kseq_read(record)) < 0) {
        fprintf(stderr, "ERROR: No reference sequence found!\n");
        exit(1);
    }
    printf("Sequence name: %s\n", record->name.s);
    printf("Sequence size: %zu\n", record->seq.l);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // compress reference on CPU, transfer to GPU and build the seed table
    timer.Start();
    fprintf(stdout, "Compressing reference, transferring to GPU device and constructing the seed table.\n");
    GpuReadMapper::ReferenceArrays referenceArrays;
    referenceArrays.allocateReferenceArrays(record->seq.s, record->seq.l, kmerSize);
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    // Read in the reference sequence
    timer.Start();
    fprintf(stdout, "Reading read file %s and mapping the reads to the reference.\n", readsFilename.c_str());
    gzFile qp = gzopen(readsFilename.c_str(), "r");
    if (!qp) {
        fprintf(stderr, "ERROR: Cannot open file: %s\n", readsFilename.c_str());
        exit(1);
    }

    kseq_t *read = kseq_init(qp);
    uint32_t totReads = 0;
    uint32_t batchReads = 0;
    //size_t compressedReadSize = (readSize+15/16);
    GpuReadMapper::ReadBatch* readBatch;
    while ((kseq_read(read) >= 0) && (totReads < maxReads)) {
        totReads++;
        batchReads++;

        // If first read in the batch, allocate memory for a new read batch
        if (batchReads == 1) {
            readBatch = new GpuReadMapper::ReadBatch(readSize, batchSize);
        }

        // Add read to the batch
        readBatch->addRead(read->name.s, read->name.l, read->seq.s, read->seq.l);
        
        // If the number of reads in the batch equals batchSize, do the
        // following:
        // 1. transfer the batch of reads
        // 2. map each read to highest-scoring location in the reference
        // 3. print the details of the mapped reads
        // 4. clear the memory allocated for the read batch
        // ASSIGNMENT 4 TASK: Implement pipeline parallelism for the 4 stages
        // HINT: Use tbb::pipeline (remember to set the tokens to numThreads and
        // be careful about serial and parallel mode for filter chain) for the
        // while loop
        if (batchReads == batchSize) {
            GpuReadMapper::transferReadBatch(readBatch);
            GpuReadMapper::mapReadBatch(referenceArrays, readBatch, kmerSize, kmerWindow);
            GpuReadMapper::printReadBatch(readBatch);
            GpuReadMapper::clearReadBatch(readBatch);
            batchReads = 0;
        }
    }

    // Map the residual batch of reads
    if (batchReads > 0) {
        GpuReadMapper::transferReadBatch(readBatch);
        GpuReadMapper::mapReadBatch(referenceArrays, readBatch, kmerSize, kmerWindow);
        GpuReadMapper::printReadBatch(readBatch);
        GpuReadMapper::clearReadBatch(readBatch);
    }

    if (totReads == 0) {
        fprintf(stderr, "ERROR: No reads found!\n");
        exit(1);
    }
    fprintf(stdout, "Completed in %ld msec \n\n", timer.Stop());

    fprintf(stdout, "Clearing reference arrays and exiting.\n");
    GpuReadMapper::clearReferenceArrays(referenceArrays);

    return 0;
}

