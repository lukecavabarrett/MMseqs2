#ifndef COUNTIN32ARRAY_H
#define COUNTIN32ARRAY_H

#include "IndexTable.h"

namespace detail {
constexpr unsigned long bits_to_represent(const unsigned long N) {
    if(N==0)return 1;
    unsigned long result = 31;
    for(unsigned long D=1;D<=32;++D)result+=((N >= (1UL << (D - 1)) && N < (1UL << D)) ? D : -1);
    return result;
};

}

struct __attribute__((__packed__)) CounterResult {
    unsigned int id;
    unsigned short diagonal;
    unsigned char count;

    static bool sortById(const CounterResult &first, const CounterResult &second) {
        if (first.id < second.id)
            return true;
        if (second.id < first.id)
            return false;
        return false;
    }

    static bool sortScore(const CounterResult &first, const CounterResult &second) {
        if (first.count > second.count)
            return true;
        if (second.count > first.count)
            return false;
        return false;
    }
};

template<unsigned int BINSIZE>
class CacheFriendlyOperations {
public:
    // 00000000000000000000000111111111
    static constexpr unsigned int MASK_0_5 = BINSIZE - 1;
    static constexpr unsigned int MASK_0_5_BIT = detail::bits_to_represent(MASK_0_5);

    CacheFriendlyOperations(size_t maxElement, size_t initBinSize);
    ~CacheFriendlyOperations();

    size_t findDuplicates(IndexEntryLocal **input, CounterResult *output, size_t outputSize, unsigned short indexFrom, unsigned short indexTo, bool computeTotalScore);

    // merge elements in CounterResult assuming that each element (diagonalMatcher.id) exist at most twice
    size_t mergeElementsByScore(CounterResult *inputOutputArray, const size_t N);

    // merge elements in CounterResult by diagonal, combines elements with same ids that occur after each other
    size_t mergeElementsByDiagonal(CounterResult *inputOutputArray, const size_t N, const bool keepScoredHits = false);

    size_t keepMaxScoreElementOnly(CounterResult *inputOutputArray, const size_t N);

private:
    // this bit array should fit in L1/L2
    size_t duplicateBitArraySize;
    unsigned char *duplicateBitArray;
    // needed for lower bit hashing function
    const static unsigned int BINCOUNT = MASK_0_5 + 1;
    size_t binSize;
    // pointer for hashing
    CounterResult **bins;
    // array to keep the bin elements
    CounterResult *binDataFrame;

    struct __attribute__((__packed__)) TmpResult {
        unsigned int id;
        unsigned short diagonal;
    };
    // needed to temporary keep ids
    TmpResult *tmpElementBuffer;

    // detect if overflow occurs
    bool checkForOverflowAndResizeArray(bool includeTmpResult);

    // reset pointer to the bin start pos
    void setupBinPointer();

    // hash input array based on MASK_0_5
    void hashElements(CounterResult *inputArray, size_t N);

    // hash index entry and compute diagonal
    void hashIndexEntry(unsigned short position_i, IndexEntryLocal *inputArray, size_t N, CounterResult *lastPosition);

    // detect duplicates in diagonal
    size_t findDuplicates(CounterResult *output, size_t outputSize, bool computeTotalScore);

    // merge by id and combine score
    size_t mergeScoreDuplicates(CounterResult *output);

    size_t mergeDiagonalDuplicates(CounterResult *output);

    size_t mergeDiagonalKeepScoredHitsDuplicates(CounterResult *output);

    size_t keepMaxElement(CounterResult *output);
};

#endif
