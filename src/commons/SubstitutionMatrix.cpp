#include "SubstitutionMatrix.h"
#include "Util.h"



SubstitutionMatrix::SubstitutionMatrix(const char *scoringMatrixFileName_,
                                       float bitFactor, float scoreBias = 0.2) :
        scoringMatrixFileName(scoringMatrixFileName_)
{
    // read amino acid substitution matrix from file
    std::string fileName(scoringMatrixFileName);
    matrixName = Util::base_name(fileName, "/\\");
    matrixName = Util::remove_extension(matrixName);
    if (fileName.substr(fileName.length()-4, 4).compare(".out") == 0)
        readProbMatrix();
    else{
        std::cerr << "Invalid format of the substitution matrix input file! Only .out files are accepted.\n";
        EXIT(1);
    }

    generateSubMatrix(this->probMatrix, this->subMatrixPseudoCounts, this->subMatrix, this->subMatrix2Bit, this->alphabetSize, bitFactor, scoreBias);
    this->bitFactor = bitFactor;
}


void SubstitutionMatrix::calcLocalAaBiasCorrection(const BaseMatrix *m,
                                                   const int * int_sequence,
                                                   const int N,
                                                   float * compositionBias){
    const int windowSize = 40;
    for (int i = 0; i < N; i++){
        const int minPos = std::max(0, (i - windowSize/2));
        const int maxPos = std::min(N, (i + windowSize/2));
        const int windowLength = maxPos - minPos;
        // negative score for the amino acids in the neighborhood of i
        int sumSubScores = 0;
        short    * subMat = m->subMatrix[int_sequence[i]];
        for (int j = minPos; j < maxPos; j++){
            sumSubScores += subMat[int_sequence[j]];
        }
        // remove own amino acid
        sumSubScores -= subMat[int_sequence[i]];
        float deltaS_i = (float) sumSubScores;
        // negative avg.
        deltaS_i /= -1.0 * static_cast<float>(windowLength);
        // positive score for the background score distribution for i
        for (int a = 0; a < m->alphabetSize; a++){
            deltaS_i += m->pBack[a] * static_cast<float>(subMat[a]);
        }
        compositionBias[i] = deltaS_i;
//        std::cout << i << " " << compositionBias[i] << std::endl;
    }
}

/* Compute aa correction
   => p(a) =  ( \prod_{i=1}^L pi(a) )^(1/L)
   => p(a) = 2^[ (1/L) * \log2 ( \prod_{i=1}^L pi(a) )
   => p(a) = 2^[ (1/L) * \sum_{i=1}^L  \log2 pi(a) ]
   => p(a) = 2^[ (1/L) * \sum_{i=1}^L  \log2 ( pi(a) / f(a) )  + log2 f(a) ]
   => p(a) = f(a) * 2^[ (1/L) * \sum_{i=1}^L  S(pi, a) ]
 */

void SubstitutionMatrix::calcGlobalAaBiasCorrection(const BaseMatrix *m,
                                                    const short *profileScores,
                                                    const unsigned int *profileIndex,
                                                    const size_t profileAASize,
                                                    const int N,
                                                    float * compositionBias){

    const int windowSize = 40;
    for (int i = 0; i < N; i++){
        const int minPos = std::max(0, (i - windowSize/2));
        const int maxPos = std::min(N, (i + windowSize/2));
        const int windowLength = maxPos - minPos;
        // negative score for the amino acids in the neighborhood of i
        int aaSum[20];
        for (int j = minPos; j < maxPos; j++){
            const short * subMat = profileScores + (j * profileAASize);
            const unsigned int * aaIdx = profileIndex + (j * profileAASize);
            if(i == j )
                continue;
            for(size_t aa = 0; aa < 20; aa++){
                const unsigned int aaPos = aaIdx[aa];
                aaSum[aaPos] += subMat[aaPos];
            }
        }
        const unsigned int * aaIdx = profileIndex + (i * profileAASize);
        const short * subMat = profileScores + (i * profileAASize);
        for(size_t aa = 0; aa < 20; aa++) {
            const unsigned int aaPos = aaIdx[aa];
            float deltaS_i_aa = (float) aaSum[aaPos];
            // negative avg.
            deltaS_i_aa /= -1.0 * static_cast<float>(windowLength);
            // positive score for the background score distribution for i
            deltaS_i_aa += m->pBack[aaPos] * static_cast<float>(subMat[aaPos]);
            compositionBias[i*20 +aaIdx[aa]] = deltaS_i_aa;
        }
//        std::cout << i << " " << compositionBias[i] << std::endl;
    }

}



SubstitutionMatrix::~SubstitutionMatrix(){
}

void SubstitutionMatrix::readProbMatrix(){

    std::ifstream in(scoringMatrixFileName);
    if( in.fail() ) {
        std::cerr << "Cannot read " << scoringMatrixFileName << "\n";
        EXIT(1);
    }
    int row    = 0;
    int column = 0;
    std::string line;
    bool capture = false;
    unsigned char aa_lookup[20];
    while( in.good() ){
        getline( in, line );
        if( line.length()>11 && line.substr(0, 11)!="Frequencies" && !capture )
            continue;
        if( line.length()>11 && line.substr(0, 11)=="Frequencies"){
            capture=true;
            continue;
        }
        // all are read amino acids
        if( row == 20 ) break;
        std::stringstream stream(line);
        std::string h;
        stream >> h;
        if( h=="" ) continue;

        if (!isalpha(h.at(0))){
            column = 0;
            stream.clear();
            stream.str(line);
            float f;
            size_t row_aa_index = aa2int[aa_lookup[row]];
            while( stream >> f ){
                size_t column_aa_index = aa2int[aa_lookup[column]];
                probMatrix[row_aa_index][column_aa_index] = f;
                probMatrix[column_aa_index][row_aa_index] = f;
                ++column;
            }
            ++row;
        }else{
            char * words[20];
            char * data = (char *)line.c_str();
            if(Util::getWordsOfLine(data, words, 20) != 20){
                std::cerr << "Not enough AminoAcids in Substituon matrix, please check format.\n";
                EXIT(-1);
            }else{
                for(size_t i = 0; i < 20; i++){
                    aa_lookup[i] = *words[i];
                }
            }
        }
    }
    in.close();

    double sum=0.0;
    for(int i=0; i<alphabetSize; ++i)
        for(int j=0; j<alphabetSize; ++j){
            if( i==j ) pBack[i] += probMatrix[i][j];
            else       pBack[i] += (probMatrix[i][j]/2.0f);
            if( j<=i ) sum += probMatrix[i][j];
        }

    const double _2sum = 2.0*sum;
    double pbsum = 0.0;
    for(int i=0; i<alphabetSize; ++i){
        pbsum += pBack[i];
        for(int j=0; j<alphabetSize; ++j)
            if( i==j ) probMatrix[i][j] = probMatrix[i][j] / sum;
            else       probMatrix[i][j] = probMatrix[i][j] / _2sum;
    }

    for(int i=0; i<alphabetSize; ++i)pBack[i] /= sum;




}

